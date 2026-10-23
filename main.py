from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import Image, Reply
from astrbot.api.star import Context, Star

from .sauce import (
    format_authors,
    format_entries,
    rank_authors_by_image,
    search_by_image,
)


class SauceFinderPlugin(Star):
    """反向图片搜索：查找图片出处（原图/画师/Coser 署名）。

    用法：
    - 同一条消息：sauce + 图片
    - 回复一张包含图片的消息，输入 sauce
    """

    def __init__(self, context: Context):
        super().__init__(context)
        self.cfg = {
            "max_results": 8,
            "timeout": 20,
            "proxy": "",
        }

    async def initialize(self):
        cfg = getattr(self.context, "config", None)
        if cfg:
            for key in self.cfg:
                try:
                    self.cfg[key] = cfg.get(key, self.cfg[key])
                except Exception:
                    pass
        logger.info(f"[saucefinder] 配置: {self.cfg}")

    @filter.command("sauce")
    async def sauce(self, event: AstrMessageEvent):
        """反搜图片出处（需同时发送图片，或回复一张图片）"""
        image_bytes = await self._extract_image(event)
        if image_bytes is None:
            yield event.plain_result(
                "请在同一条消息中附带图片，或回复一条包含图片的消息后使用 sauce。"
            )
            return

        yield event.plain_result("正在反搜，请稍候……")
        try:
            page_url, entries = await search_by_image(
                image_bytes,
                timeout=self.cfg["timeout"],
                proxy=self.cfg["proxy"],
            )
        except Exception as e:
            logger.warning(f"[saucefinder] 搜索失败: {e}")
            yield event.plain_result(f"搜索失败：{e}")
            return

        text = format_entries(entries, limit=self.cfg["max_results"])
        try:
            # 用候选原图与查询图做感知哈希比对，账号按图片相似度排序
            authors = await rank_authors_by_image(
                entries,
                image_bytes,
                timeout=self.cfg["timeout"],
                proxy=self.cfg["proxy"],
            )
            if authors:
                text = format_authors(authors) + "\n\n" + text
        except Exception as e:
            logger.warning(f"[saucefinder] 账号比对失败（退化为未排序输出）: {e}")
        yield event.plain_result(f"{text}\n\n完整结果页：{page_url}")

    async def _extract_image(self, event: AstrMessageEvent) -> bytes | None:
        """从当前消息或被回复消息中提取第一张图片的字节。

        统一走 Image.convert_to_file_path()：无论图片是 URL、本地路径
        还是 base64，SDK 都会下载/解析为本地文件路径。
        """
        chain = event.get_messages() or []
        img = next((c for c in chain if isinstance(c, Image)), None)
        if img is None:
            # 回复消息场景：图片在被引用消息的 chain 里
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
