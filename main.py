import asyncio

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Node, Nodes, Plain, Reply
from astrbot.api.star import Context, Star

from . import engines
from .engines import PostCandidate, search_yandex
from .fuse import format_verdicts, fuse
from .llm_candidates import SYSTEM_PROMPT, build_prompt, build_sources, parse_candidates
from .preprocess import prepare_image
from .sauce import _hash_similarity, _perceptual_hashes
from .social import fetch_followers, pick_by_followers

# aiohttp 复用 engines 的会话工厂
import aiohttp


class SauceFinderPlugin(Star):
    """反向图片搜索：多引擎查找图片出处与作者（含 X 原帖）。

    用法：
    - 同一条消息：找鸡 + 图片
    - 回复一张包含图片的消息，输入 找鸡

    引擎：Yandex(.com/.ru 双域名)、Baidu、Copyseeker、SauceNAO(需 key)、
    ascii2d(需 cf_clearance)、Google Vision(需 GCP key)、
    Google Lens(需 serpapi_api_key)。
    缺 key / 被风控的引擎自动跳过，不影响其他引擎。

    兜底链：无可靠署名时 1) LLM 解析来源文字候选（evidence 验证）；
    2) 候选并列时按 fxTwitter 粉丝数取多者。
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
            "serpapi_api_key": "",
            "serpapi_visual_matches": False,
            "baidu_enabled": True,
            "copyseeker_enabled": False,
            "use_llm_candidates": True,
            "follower_tiebreak": True,
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
            "baidu" if self.cfg["baidu_enabled"] else "-baidu(关)",
            "copyseeker" if self.cfg["copyseeker_enabled"] else "-copyseeker(关)",
            "saucenao" if self.cfg["saucenao_api_key"] else "-saucenao(无key)",
            "ascii2d" if self.cfg["ascii2d_cf_clearance"] else "-ascii2d(无cookie)",
            "gcv" if self.cfg["gcv_api_key"] else "-gcv(无key)",
            "gcl" if self.cfg["serpapi_api_key"] else "-gcl(无serpapi key)",
            "llm" if self.cfg["use_llm_candidates"] else "-llm候选(关)",
        ]
        logger.info(f"[saucefinder] 引擎: {enabled}")

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

        # 0) 预处理：EXIF 纠向/缩放/透明填白；合格图保留原始字节
        image_bytes = prepare_image(image_bytes)

        # 1) 先做 Yandex 上传，拿到公网可访问的原图 URL（供 SauceNAO/GCL 用）
        candidates: list[PostCandidate] = []
        engine_errors: dict[str, str] = {}  # 引擎名 -> 失败原因（区分失败语义）
        yandex_url = ""
        try:
            yandex_url, yx = await search_yandex(
                image_bytes, timeout=timeout, proxy=proxy
            )
            candidates.extend(yx)
        except Exception as e:
            engine_errors["yandex"] = str(e)
            logger.warning(f"[saucefinder] yandex 失败: {e}")

        img_public_url = self._extract_public_url(yandex_url) if yandex_url else ""

        # 2) 其余引擎并发
        tasks: dict[str, asyncio.Task] = {}
        if self.cfg["saucenao_api_key"] and img_public_url:
            tasks["saucenao"] = asyncio.create_task(
                engines.search_saucenao(
                    img_public_url,
                    api_key=self.cfg["saucenao_api_key"],
                    timeout=timeout,
                    proxy=proxy,
                )
            )
        if self.cfg["serpapi_api_key"] and img_public_url:
            tasks["gcl"] = asyncio.create_task(
                engines.search_google_lens(
                    img_public_url,
                    api_key=self.cfg["serpapi_api_key"],
                    visual_matches=bool(self.cfg.get("serpapi_visual_matches")),
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
        if self.cfg["baidu_enabled"]:
            tasks["baidu"] = asyncio.create_task(
                engines.search_baidu(image_bytes, timeout=timeout, proxy=proxy)
            )
        if self.cfg["copyseeker_enabled"]:
            tasks["copyseeker"] = asyncio.create_task(
                engines.search_copyseeker(image_bytes, timeout=timeout + 20, proxy=proxy)
            )
        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, res in zip(tasks.keys(), results):
                if isinstance(res, BaseException):
                    reason = str(res)
                    engine_errors[name] = reason
                    logger.warning(f"[saucefinder] {name} 失败: {reason}")
                else:
                    logger.info(f"[saucefinder] {name}: {len(res)} 条")
                    candidates.extend(res)

        if not candidates:
            # 失败语义区分：引擎全挂 ≠ 图片没有出处
            msg = "本次检索未取得任何结果。"
            if engine_errors:
                msg += "\n各引擎状态：" + "；".join(
                    f"{k}: {v}" for k, v in engine_errors.items()
                )
            msg += "\n提示：引擎被风控/验证页拦截不等于图片没有出处，可稍后重试。"
            if yandex_url:
                msg += f"\nYandex 结果页: {yandex_url}"
            yield event.plain_result(msg)
            return

        # 3) 感知哈希硬确认（同图加成）
        phash_scores = await self._phash_verify_all(
            candidates, image_bytes, proxy, timeout
        )

        # 4) 融合评分
        verdicts = fuse(candidates, phash_scores)

        # 5) 粉丝数 tie-break：最优先两名置信度并列且都是 X 账号时取粉丝多者
        strong = [v for v in verdicts if v.confidence >= 0.30]
        if self.cfg["follower_tiebreak"] and len(strong) >= 2:
            top2 = strong[:2]
            if top2[0].confidence == top2[1].confidence and all(
                v.platform == "X/Twitter" for v in top2
            ):
                handles = [v.handle for v in top2]
                followers = await fetch_followers(
                    handles, timeout=10, proxy=proxy
                )
                winner = pick_by_followers(handles, followers, handles)
                logger.info(f"[saucefinder] 粉丝 tie-break: {followers} -> @{winner}")
                if winner.lower() == handles[1].lower():
                    verdicts.remove(top2[1])
                    verdicts.insert(verdicts.index(top2[0]), top2[1])

        text = format_verdicts(verdicts, limit=self.cfg["max_results"])

        # 6) LLM 文字候选兜底：无可靠署名时从来源文字提取账号线索
        if self.cfg["use_llm_candidates"] and not strong:
            llm_cands = await self._run_llm_candidates(
                candidates, event.unified_msg_origin
            )
            if llm_cands:
                lines = ["LLM 文字线索（待核实，仅提取来源页中出现过的标识）："]
                for c in llm_cands:
                    plat = f"[{c['platform']}]" if c["platform"] != "unknown" else ""
                    lines.append(
                        f"  ? @{c['handle']} {plat}  依据: “…{c['evidence'][:60]}…”"
                    )
                    if c["platform"] == "X/Twitter":
                        lines.append(
                            f"      X 搜索验证: https://x.com/search?q=from%3A{c['handle']}"
                        )
                text += "\n" + "\n".join(lines)

        # 7) 附 Yandex 出处列表（兜底人工排查）
        top_pages = [c.page_url for c in candidates if c.engine == "yandex"][
            : self.cfg["max_results"]
        ]
        if top_pages:
            lines = [f"Yandex 出处（前 {len(top_pages)} 条）："]
            for i, u in enumerate(top_pages, 1):
                lines.append(f"{i}. {u}")
            text += "\n" + "\n".join(lines)
        # 部分引擎失败时附状态，帮助区分"无署名"与"引擎被拦"
        if engine_errors:
            text += "\n引擎状态：" + "；".join(
                f"{k}: {v}" for k, v in engine_errors.items()
            )
        if yandex_url:
            text += f"\n完整结果页：{yandex_url}"

        # 8) 发送方式：合并转发（可复制文本）或普通纯文本
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

    async def _run_llm_candidates(
        self, candidates: list[PostCandidate], umo: str
    ) -> list[dict]:
        """调用已配置的文本模型解析候选；任何失败返回空列表。"""
        sources = build_sources(candidates)
        if len(sources) < 2:
            return []
        try:
            prov_id = await self.context.get_current_chat_provider_id(umo)
            resp = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=prov_id,
                    prompt=build_prompt(sources),
                    system_prompt=SYSTEM_PROMPT,
                ),
                timeout=30,
            )
            return parse_candidates(resp.completion_text, sources)
        except Exception as e:
            logger.warning(f"[saucefinder] LLM 候选解析失败(降级): {e}")
            return []

    def _extract_public_url(self, yandex_result_url: str) -> str:
        """从 Yandex 结果页 URL 中解出 Yandex 托管的公网原图 URL。"""
        import re
        from urllib.parse import unquote

        m = re.search(r"url=(https[^&]+)", yandex_result_url)
        if m:
            return unquote(m.group(1))
        return ""

    async def _phash_verify_all(
        self, candidates: list[PostCandidate], query: bytes, proxy: str, timeout: int
    ) -> dict[str, float]:
        """对所有带原图 URL 的候选按署名分组下载，计算与查询图的相似度。"""
        scores: dict[str, float] = {}
        qh = _perceptual_hashes(query)
        if qh is None:
            return scores
        from .fuse import _DESC_CREDIT_RE, _TITLE_HANDLE_RE, _X_URL_RE, _clean

        per_handle: dict[str, list[str]] = {}
        for c in candidates:
            if not c.image_url:
                continue
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
