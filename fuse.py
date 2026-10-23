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
# 强署名：明确 credit 模式（"sc: handle" / "handle's tweet photo/video"）
# 实测：sotwe 转发推文描述为 "pon9019's tweet photo."，必须是 photo/video/status 全集
_DESC_CREDIT_RE = re.compile(
    r"(?:sc|credit|cr|source|via|cn|photo|drawn\s+by|tw)\s*[:：]\s*@?([A-Za-z0-9_]{2,20})"
    r"|([A-Za-z0-9_]{2,20})'(?:s|’s)\s+(?:tweet\s+)?(?:photo|video|status|post)"
)
# 弱证据：裸 @mention（可能是转发者自引用）
_DESC_MENTION_RE = re.compile(r"@([A-Za-z0-9_]{2,20})")
_STATUS_RE = re.compile(r"(?:x\.com|twitter\.com)/(\w{2,20})/status/(\d+)")

_STOPWORDS = {"status", "search", "user", "home", "intent", "sunna", "sc", "i", "web"}

# handle 形态：4-20 位字母数字下划线，至少含一个字母（排除纯数字 id/日期）
_HANDLE_LIKE_RE = re.compile(r"[A-Za-z0-9_]{4,20}")
_LETTER_RE = re.compile(r"[A-Za-z]")

# 聚合站标题常见通用词，不是人名
_TITLE_NOISE = {
    "nude",
    "leaks",
    "leaked",
    "onlyfans",
    "patreon",
    "photo",
    "photos",
    "video",
    "videos",
    "naked",
    "sexy",
    "cosplay",
    "forum",
    "wall",
    "free",
    "top",
    "best",
    "girl",
    "girls",
    "asian",
    "japanese",
    "leakedmodels",
    "fappening",
    "fapeza",
    "fapello",
    "nudostar",
    "nudogram",
    "fapodrop",
    "tokyomotion",
    "sotwe",
    "urlebird",
    "twitter",
    "profile",
    "голая",
    "фото",
    "сливы",
    # 英文站结构性词（标题分页/站点名）
    "page",
    "pages",
    "index",
    "home",
    "main",
    "list",
    "category",
    "tag",
    "search",
    "archive",
    "post",
    "posts",
    "comments",
    "login",
    "update",
    # 站名/品牌（会跨站出现在标题里）
    "aliexpress",
    "imall",
    "telegra",
    "telegram",
    "telegraph",
    "vikacg",
    "alibaba",
    "amazon",
    "ebay",
    "reddit",
    "google",
    "yandex",
    "bing",
    "kitty",
    "welfare",
    "silver",
    "misty",
    "models",
    "model",
}


def _is_handle_like(tok: str) -> bool:
    """判断标题 token 是否像账号名：形态匹配、含字母、非通用词。"""
    if not _HANDLE_LIKE_RE.fullmatch(tok):
        return False
    if not _LETTER_RE.search(tok):
        return False
    if tok.lower() in _TITLE_NOISE:
        return False
    return True


def _domain_root(url: str) -> str:
    """域名根（剥离语言前缀 ru./de. 等），用于跨站一致计分。"""
    host = url.split("/")[2] if "://" in url else ""
    parts = host.split(".")
    if len(parts) > 2 and len(parts[0]) <= 3:
        parts = parts[1:]
    return ".".join(parts)


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
    domains: set[str] = field(default_factory=set)  # 引用该 handle 的独立域名


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
            domain_root = _domain_root(c.page_url)
            title_m = _TITLE_HANDLE_RE.search(c.title) or _X_URL_RE.search(c.page_url)
            title_h = _clean(title_m.group(1)) if title_m else ""
            # 1) description 强署名（credit 模式，原作者）
            for m in _DESC_CREDIT_RE.finditer(c.description or ""):
                h = _clean(m.group(1) or m.group(2))
                if h:
                    a = _author(h, platform)
                    a.confidence = min(1.0, a.confidence + 0.35)
                    a.domains.add(domain_root)
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
                    a.domains.add(domain_root)
                    a.evidence.append(f"正文提及: @​{h}")
                    a.page_url = a.page_url or c.page_url
            # 3) 标题/URL handle（转发者）—— 但跨域名一致性提权
            if title_h:
                a = _author(title_h, platform)
                if domain_root not in a.domains:
                    # 同一域名重复出现只 +0.05（同站多页信息量低）；
                    # 新域名出现该 handle +0.15（跨站一致指向同一身份）
                    a.confidence = min(
                        1.0, a.confidence + (0.15 if a.domains else 0.10)
                    )
                    a.domains.add(domain_root)
                else:
                    a.confidence = min(1.0, a.confidence + 0.05)
                a.evidence.append(f"站点署名({domain_root}): {c.page_url[:55]}")
                a.page_url = a.page_url or c.page_url
            # 4) 聚合站标题开头的 handle 形态词（"pon9019 Nude Leaks..."）
            #    非社媒域名也会以人名为页面标题，是跨域一致性的重要来源
            if not platform:
                for raw_tok in (c.title or "").split():
                    # "pon9019-nude/7159524" 类 slug 只取第一段
                    tok = raw_tok.strip(",.;:!?()[]").split("-")[0]
                    if not _is_handle_like(tok):
                        continue
                    a = _author(tok, "未知")
                    if domain_root not in a.domains:
                        a.domains.add(domain_root)
                        if len(a.domains) >= 2:
                            # 第二个独立域名确认后才计分（单域噪声不注册）；
                            # 封顶 0.30：跨站人名只说明"此人的图在流通"，
                            # 不能压过正文强署名（0.35，直接指明作者）
                            bonus = 0.10 if a.confidence == 0 else 0.10
                            a.confidence = min(0.30, a.confidence + bonus)
                            a.evidence.append(
                                f"聚合站人名(跨站一致 x{len(a.domains)}): {c.title[:50]}"
                            )
                            a.page_url = a.page_url or c.page_url
                    else:
                        a.confidence = min(0.30, a.confidence + 0.05)
                # URL 路径中的 handle 形态段（fappeningbook.com/pon9019-nude/）
                for seg in c.page_url.split("/")[3:]:
                    seg = seg.split("?")[0]
                    m2 = _HANDLE_LIKE_RE.match(seg)
                    if (
                        m2
                        and _is_handle_like(m2.group(0))
                        and (m2.group(0).lower(), "未知") in authors
                    ):
                        # 只为已被其他证据登记的 handle 记域名（避免噪声 slug 注册新作者）
                        a = authors[(m2.group(0).lower(), "未知")]
                        if domain_root not in a.domains:
                            a.domains.add(domain_root)
                            if len(a.domains) >= 2:
                                a.confidence = min(0.30, a.confidence + 0.10)
                                a.evidence.append(f"URL 路径含该名({domain_root})")
        elif c.engine == "gcv":
            # GCV 只给页面不给作者；作为页面证据挂在未知作者上无效，
            # 但 URL 若是 X 用户页可提取 handle
            m = re.search(r"(?:x\.com|twitter\.com)/(\w{2,20})/?$", c.page_url)
            h = _clean(m.group(1)) if m else ""
            if h:
                a = _author(h, "X/Twitter")
                a.confidence = min(1.0, a.confidence + 0.15)
                a.evidence.append(f"Google Vision 匹配页: {c.page_url[:70]}")

    # --- 感知哈希加成 ---
    # 实测教训：转发页同样含同图，"页面含同图"≠"该页账号是作者"。
    # 跨域名（>=2 站点引用）的 handle 才能给完整作者确认加成；
    # 单域名命中只给图源页加成（信息量："同图在该账号页面出现过"）。
    for (h_lower, _), a in authors.items():
        sim = phash_scores.get(h_lower, 0.0)
        if sim >= 0.90:
            if len(a.domains) >= 2:
                a.confidence = min(1.0, a.confidence + 0.20)
                a.evidence.append(f"感知哈希同图确认(跨{len(a.domains)}域): {sim:.0%}")
            else:
                a.confidence = min(1.0, a.confidence + 0.10)
                a.evidence.append(f"同图出现在其页面(单域,或为转发): {sim:.0%}")
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


# 强证据阈值：有正文署名/官方引擎命中/同图确认之一才作为"作者候选"展示
_STRONG_CONFIDENCE = 0.30


def format_verdicts(verdicts: list[AuthorVerdict], limit: int = 5) -> str:
    """格式化融合结果。

    实测教训：镜像页标题 handle 大多是转发者。仅靠弱证据（+0.10
    累加）的作者不允许以"作者候选"名义出现，折叠进明确标注低可信
    的列表，避免误导。无任何强证据时明确说明。
    """
    if not verdicts:
        return (
            "未能确定作者候选（各引擎均无署名证据）。\n"
            "提示：配置 SauceNAO api_key 可显著提升原帖命中率。"
        )

    strong = [v for v in verdicts if v.confidence >= _STRONG_CONFIDENCE]
    weak = [v for v in verdicts if v.confidence < _STRONG_CONFIDENCE]

    lines = []
    if strong:
        lines.append(f"作者候选（{len(strong)} 位，按置信度排序）：")
        for v in strong[:limit]:
            lines.append(f"  @{v.handle}  [{v.platform}]  置信度 {v.confidence:.2f}")
            if v.post_url:
                lines.append(f"      原帖: {v.post_url}")
            elif v.page_url:
                lines.append(f"      证据页: {v.page_url}")
            for ev in v.evidence[:3]:
                lines.append(f"      · {ev}")
    else:
        lines.append(
            "未找到可靠署名证据（无正文 credit、无官方引擎命中、无同图确认）。"
        )
        lines.append("以下仅为结果页中出现过的账号，多为转发者，低可信：")
        for v in weak[:limit]:
            lines.append(f"  ? @{v.handle}  [{v.platform}]  置信度 {v.confidence:.2f}")
    if strong and weak:
        lines.append(f"（另有 {len(weak)} 个仅弱证据账号，多为转发者，已折叠）")
    if not strong:
        lines.append("提示：配置 SauceNAO api_key 可显著提升原帖命中率。")
    return "\n".join(lines)
