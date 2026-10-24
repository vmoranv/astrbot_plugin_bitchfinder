import asyncio

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Node, Nodes, Plain, Reply
from astrbot.api.star import Context, Star

from . import engines
from .engines import PostCandidate, search_yandex
from .fuse import format_verdicts, fuse
from .sauce import _hash_similarity, _perceptual_hashes

# aiohttp 复用 engines 的会话工厂
import aiohttp


class SauceFinderPlugin(Star):
    """反向图片搜索：多引擎查找图片出处与作者（含 X 原帖）。

    用法：
    - 同一条消息：找鸡 + 图片
    - 回复一张包含图片的消息，输入 找鸡

    引擎：Yandex(.com/.ru 双域名)、SauceNAO(需 api_key)、
    ascii2d(需 cf_clearance)、Google Vision WEB_DETECTION(需 GCP key)。
    缺 key / 被风控的引擎自动跳过，不影响其他引擎。
    """

    def __init__(self, context: Context):
        super().__init__(context)
        self.cfg: dict = {
            "max_results": 8,
            "timeout": 20,
            "proxy": "",
            "saucenao_api_key": "",
            "gcv_api_key": "",
            "ascii2d_cf_clearance": "",
            "send_as_forward": True,
        }

    async def initialize(self):
        cfg = getattr(self.context, "config", None)
        if cfg:
            for key in self.cfg:
                try:
                    self.cfg[key] = cfg.get(key, self.cfg[key])
                except Exception:
                    pass
        enabled = [
            "yandex",
            "saucenao" if self.cfg["saucenao_api_key"] else "-saucenao(无key)",
            "ascii2d" if self.cfg["ascii2d_cf_clearance"] else "-ascii2d(无cookie)",
            "gcv" if self.cfg["gcv_api_key"] else "-gcv(无key)",
        ]
        logger.info(f"[saucefinder] 引擎: {enabled}, 配置: {self.cfg}")

    @filter.command("找鸡", alias={"kfc"})
    async def find_sauce(self, event: AstrMessageEvent):
        """反搜图片出处（需同时发送图片，或回复一张图片）。别名为 kfc。"""
        image_bytes = await self._extract_image(event)
        if image_bytes is None:
            yield event.plain_result(
                "请在同一条消息中附带图片，或回复一条包含图片的消息后使用 找鸡。"
            )
            return

        yield event.plain_result("正在多引擎反搜，请稍候……")
        proxy = self.cfg["proxy"]
        timeout = self.cfg["timeout"]

        # 1) 先做 Yandex 上传，拿到公网可访问的原图 URL（供 SauceNAO 用）
        candidates: list[PostCandidate] = []
        yandex_url = ""
        yandex_error = ""
        try:
            yandex_url, yx = await search_yandex(
                image_bytes, timeout=timeout, proxy=proxy
            )
            candidates.extend(yx)
        except Exception as e:
            yandex_error = str(e)
            logger.warning(f"[saucefinder] yandex 失败: {e}")

        # 2) 其余引擎并发（SauceNAO 需要图片 URL）
        tasks: dict[str, asyncio.Task] = {}
        if self.cfg["saucenao_api_key"] and yandex_url:
            img_public_url = self._extract_public_url(yandex_url)
            if img_public_url:
                tasks["saucenao"] = asyncio.create_task(
                    engines.search_saucenao(
                        img_public_url,
                        api_key=self.cfg["saucenao_api_key"],
                        timeout=timeout,
                        proxy=proxy,
                    )
                )
        if self.cfg["ascii2d_cf_clearance"]:
            tasks["ascii2d"] = asyncio.create_task(
                engines.search_ascii2d(
                    image_bytes,
                    timeout=timeout,
                    proxy=proxy,
                    cf_clearance=self.cfg["ascii2d_cf_clearance"],
                )
            )
        if self.cfg["gcv_api_key"]:
            tasks["gcv"] = asyncio.create_task(
                engines.search_gcv(
                    image_bytes,
                    api_key=self.cfg["gcv_api_key"],
                    timeout=timeout,
                    proxy=proxy,
                )
            )
        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, res in zip(tasks.keys(), results):
                if isinstance(res, BaseException):
                    logger.warning(f"[saucefinder] {name} 失败: {res}")
                else:
                    logger.info(f"[saucefinder] {name}: {len(res)} 条")
                    candidates.extend(res)

        if not candidates:
            msg = "所有引擎均未返回结果。"
            if yandex_error:
                msg += f" Yandex: {yandex_error}"
            yield event.plain_result(msg)
            return

        # 3) Yandex 候选做感知哈希硬确认（同图 +20% 置信度）
        phash_scores = await self._phash_verify_all(
            candidates, image_bytes, proxy, timeout
        )

        # 4) 融合评分
        verdicts = fuse(candidates, phash_scores)
        text = format_verdicts(verdicts, limit=self.cfg["max_results"])

        # 5) 附 Yandex 出处列表（无署名证据时兜底人工排查）
        top_pages = [c.page_url for c in candidates if c.engine == "yandex"][
            : self.cfg["max_results"]
        ]
        if top_pages:
            lines = [f"Yandex 出处（前 {len(top_pages)} 条）："]
            for i, u in enumerate(top_pages, 1):
                lines.append(f"{i}. {u}")
            text += "\n" + "\n".join(lines)
        if yandex_url:
            text += f"\n\n完整结果页：{yandex_url}"

        # 6) 发送方式：合并转发（可复制文本）或普通纯文本
        if self.cfg.get("send_as_forward", True):
            try:
                node = Node(
                    content=[Plain(text)],
                    uin=event.get_self_id(),
                    name="找鸡结果",
                )
                yield event.chain_result([Nodes(nodes=[node])])
                return
            except Exception as e:
                logger.warning(f"[saucefinder] 合并转发失败，回退纯文本: {e}")
        yield event.plain_result(text)

    def _extract_public_url(self, yandex_result_url: str) -> str:
        """从 Yandex 结果页 URL 中解出 Yandex 托管的公网原图 URL。

        SauceNAO 需要一个它能抓到的图片 URL；Yandex 的
        avatars.mds.yandex.net 托管 URL 可公开访问，符合要求。
        """
        import re
        from urllib.parse import unquote

        m = re.search(r"url=(https[^&]+)", yandex_result_url)
        if m:
            return unquote(m.group(1))
        return ""

    async def _phash_verify_all(
        self, candidates: list[PostCandidate], query: bytes, proxy: str, timeout: int
    ) -> dict[str, float]:
        """对所有带原图 URL 的候选按署名分组下载，计算与查询图的相似度。

        覆盖全部引擎（Yandex/SauceNAO/ascii2d/gcv），同图硬确认的
        +20% 置信度加成不再漏掉非 Yandex 引擎命中的作者。
        """
        scores: dict[str, float] = {}
        qh = _perceptual_hashes(query)
        if qh is None:
            return scores
        # 只对有 handle 的条目做
        per_handle: dict[str, list[str]] = {}
        for c in candidates:
            if not c.image_url:
                continue
            # 用 fuse 中相同的署名提取（简化：title + desc）
            from .fuse import _DESC_CREDIT_RE, _TITLE_HANDLE_RE, _X_URL_RE, _clean

            handles = set()
            if c.author_handle:
                handles.add(c.author_handle.lower())
            for m in _DESC_CREDIT_RE.finditer(c.description or ""):
                h = _clean(m.group(1) or m.group(2) or m.group(3))
                if h:
                    handles.add(h.lower())
            m = _TITLE_HANDLE_RE.search(c.title) or _X_URL_RE.search(c.page_url)
            if m:
                h = _clean(m.group(1))
                if h:
                    handles.add(h.lower())
            for h in handles:
                per_handle.setdefault(h, []).append(c.image_url)
        if not per_handle:
            return scores
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            connector=aiohttp.TCPConnector(ssl=False),
            trust_env=False,
        ) as session:
            for h, urls in per_handle.items():
                for u in urls[:3]:
                    try:
                        async with session.get(u, proxy=proxy or None) as resp:
                            if resp.status != 200:
                                continue
                            ch = _perceptual_hashes(await resp.read())
                    except Exception:
                        continue
                    if ch is None:
                        continue
                    sim = _hash_similarity(qh, ch)
                    if sim > scores.get(h, 0.0):
                        scores[h] = sim
        return scores

    async def _extract_image(self, event: AstrMessageEvent) -> bytes | None:
        """从当前消息或被回复消息中提取第一张图片的字节。"""
        chain = event.get_messages() or []
        img = next((c for c in chain if isinstance(c, Image)), None)
        if img is None:
            reply = next((c for c in chain if isinstance(c, Reply)), None)
            if reply:
                rchain = getattr(reply, "chain", None) or []
                img = next((c for c in rchain if isinstance(c, Image)), None)
        if img is None:
            return None
        try:
            local_path = await img.convert_to_file_path()
            with open(local_path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"[saucefinder] 图片提取失败: {e}")
            return None

    async def terminate(self):
        pass
