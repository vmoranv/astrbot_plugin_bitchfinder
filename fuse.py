"""跨引擎证据融合与置信度评分。

输入各引擎的 PostCandidate 列表，输出按置信度排序的作者及其
最佳原帖 URL。置信度模型（0-1，封顶）：

- description/正文署名（Yandex 转发者正文中注明原作者）：+0.35/次
- SauceNAO 返回的作者（官方索引，直接给 status URL）：+0.45，且附原帖
- ascii2d 结果作者（链接直达 status/artwork）：+0.30，且附原帖
- 标题/URL handle（可能是转发者）：+0.10/次
- 感知哈希同图硬确认（>=0.90）：+0.20；0.75-0.90 可疑：+0.05

X 原帖 URL 解析优先级：SauceNAO ext_urls > ascii2d status 链接 >
从 Yandex 镜像页 URL 提取的用户页（非具体帖）。
"""

import re
from dataclasses import dataclass, field

from .engines import PostCandidate

_TITLE_HANDLE_RE = re.compile(r"@([A-Za-z0-9_]{1,20})")
_X_URL_RE = re.compile(
    r"(?:x\.com|twitter\.com|sotwe\.com|urlebird\.com)/(?:user/)?@?([A-Za-z0-9_]{2,20})(?:[/?]|$)"
)
# 强署名：明确 credit 模式（"sc: handle" / "handle's tweet"）
_DESC_CREDIT_RE = re.compile(
    r"(?:sc|credit|cr|source|via|cn|photo|drawn\s+by|tw)\s*[:：]\s*@?([A-Za-z0-9_]{2,20})"
    r"|([A-Za-z0-9_]{2,20})'(?:s|’s)\s+(?:tweet|post)"
)
# 弱证据：裸 @mention（可能是转发者自引用）
_DESC_MENTION_RE = re.compile(r"@([A-Za-z0-9_]{2,20})")
_STATUS_RE = re.compile(r"(?:x\.com|twitter\.com)/(\w{2,20})/status/(\d+)")

_STOPWORDS = {"status", "search", "user", "home", "intent", "sunna", "sc", "i", "web"}


def _clean(h: str | None) -> str:
    if not h or h.lower() in _STOPWORDS:
        return ""
    return h.lstrip("@").strip()


@dataclass
class AuthorVerdict:
    """一个作者的融合判定。"""

    handle: str
    platform: str
    confidence: float  # 0-1
    post_url: str = ""  # 最佳已知原帖（具体 status URL 优先）
    page_url: str = ""  # 证据页面
    evidence: list[str] = field(default_factory=list)  # 人类可读证据链


def fuse(
    candidates: list[PostCandidate],
    phash_scores: dict[str, float] | None = None,
) -> list[AuthorVerdict]:
    """融合所有引擎候选。phash_scores: handle.lower() -> 相似度。"""
    phash_scores = phash_scores or {}
    authors: dict[tuple[str, str], AuthorVerdict] = {}

    def _author(handle: str, platform: str) -> AuthorVerdict:
        key = (handle.lower(), platform)
        if key not in authors:
            authors[key] = AuthorVerdict(
                handle=handle, platform=platform, confidence=0.0
            )
        return authors[key]

    for c in candidates:
        # --- 具体原帖 URL（X status / pixiv artwork）---
        status_m = _STATUS_RE.search(c.post_url or "") or _STATUS_RE.search(
            c.page_url or ""
        )
        pixiv_m = re.search(
            r"pixiv\.net/artworks/(\d+)", c.post_url or c.page_url or ""
        )

        # --- 作者证据 ---
        if c.engine == "saucenao" and c.author_handle:
            a = _author(c.author_handle, c.author_platform or "X/Twitter")
            a.confidence = min(1.0, a.confidence + 0.45)
            a.evidence.append(f"SauceNAO 索引命中: {c.page_url[:70]}")
            if status_m:
                a.post_url = status_m.group(0)
            elif pixiv_m:
                a.post_url = f"https://www.pixiv.net/artworks/{pixiv_m.group(1)}"
        elif c.engine == "ascii2d" and c.author_handle:
            a = _author(c.author_handle, c.author_platform or "X/Twitter")
            a.confidence = min(1.0, a.confidence + 0.30)
            a.evidence.append(f"ascii2d 结果: {c.page_url[:70]}")
            if status_m and not a.post_url:
                a.post_url = status_m.group(0)
        elif c.engine == "yandex":
            platform = _platform_of(c)
            if not platform:
                continue
            title_m = _TITLE_HANDLE_RE.search(c.title) or _X_URL_RE.search(c.page_url)
            title_h = _clean(title_m.group(1)) if title_m else ""
            # 1) description 强署名（credit 模式，原作者）
            for m in _DESC_CREDIT_RE.finditer(c.description or ""):
                h = _clean(m.group(1) or m.group(2))
                if h:
                    a = _author(h, platform)
                    a.confidence = min(1.0, a.confidence + 0.35)
                    a.evidence.append(
                        f"转发正文署名: “…{c.description[max(0, m.start() - 15) : m.end() + 10]}…”"
                    )
                    a.page_url = a.page_url or c.page_url
                    if status_m and not a.post_url:
                        a.post_url = status_m.group(0)
            # 2) description 裸 @mention：只有与页面标题 handle 不同才算独立弱证据
            for m in _DESC_MENTION_RE.finditer(c.description or ""):
                h = _clean(m.group(1))
                if h and h.lower() != title_h.lower():
                    a = _author(h, platform)
                    a.confidence = min(1.0, a.confidence + 0.10)
                    a.evidence.append(f"正文提及: @​{h}")
                    a.page_url = a.page_url or c.page_url
            # 3) 标题/URL handle（转发者）
            if title_h:
                a = _author(title_h, platform)
                a.confidence = min(1.0, a.confidence + 0.10)
                a.evidence.append(f"镜像页署名(可能为转发者): {c.page_url[:60]}")
                a.page_url = a.page_url or c.page_url
        elif c.engine == "gcv":
            # GCV 只给页面不给作者；作为页面证据挂在未知作者上无效，
            # 但 URL 若是 X 用户页可提取 handle
            m = re.search(r"(?:x\.com|twitter\.com)/(\w{2,20})/?$", c.page_url)
            h = _clean(m.group(1)) if m else ""
            if h:
                a = _author(h, "X/Twitter")
                a.confidence = min(1.0, a.confidence + 0.15)
                a.evidence.append(f"Google Vision 匹配页: {c.page_url[:70]}")

    # --- 感知哈希硬确认加成 ---
    for (h_lower, _), a in authors.items():
        sim = phash_scores.get(h_lower, 0.0)
        if sim >= 0.90:
            a.confidence = min(1.0, a.confidence + 0.20)
            a.evidence.append(f"感知哈希同图确认: {sim:.0%}")
        elif sim >= 0.75:
            a.confidence = min(1.0, a.confidence + 0.05)
            a.evidence.append(f"感知哈希可疑相似: {sim:.0%}")

    ranked = sorted(authors.values(), key=lambda a: -a.confidence)
    return [a for a in ranked if a.confidence >= 0.10]


def _platform_of(c: PostCandidate) -> str:
    d = c.page_url.split("/")[2] if "://" in c.page_url else ""
    table = [
        ("x.com", "X/Twitter"),
        ("twitter.com", "X/Twitter"),
        ("sotwe.com", "X/Twitter"),
        ("urlebird.com", "X/Twitter"),
        ("weibo.com", "微博"),
        ("weibo.cn", "微博"),
        ("xiaohongshu.com", "小红书"),
        ("instagram.com", "Instagram"),
        ("pixiv.net", "Pixiv"),
        ("danbooru.donmai.us", "Danbooru"),
        ("coser.one", "Cosplay站"),
        ("cosppi.net", "Cosplay站"),
    ]
    for prefix, name in table:
        if d == prefix or d.endswith("." + prefix):
            return name
    return ""


def format_verdicts(verdicts: list[AuthorVerdict], limit: int = 5) -> str:
    """格式化融合结果。"""
    if not verdicts:
        return "未能确定作者候选（各引擎均无署名证据）。"
    lines = [f"作者候选（融合 {len(verdicts)} 位，按置信度排序）："]
    for v in verdicts[:limit]:
        pct = f"{v.confidence:.0%}"
        lines.append(f"  @{v.handle}  [{v.platform}]  置信度 {pct}")
        if v.post_url:
            lines.append(f"      原帖: {v.post_url}")
        elif v.page_url:
            lines.append(f"      证据页: {v.page_url}")
        for ev in v.evidence[:3]:
            lines.append(f"      · {ev}")
    return "\n".join(lines)
