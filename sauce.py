"""Yandex Images 反向搜索客户端与结果解析。

仅返回图片出处条目（源页面 URL、标题、域名、原图 URL）。
网络层使用 aiohttp（AstrBot 核心自带依赖）。
"""

import json
import re
from dataclasses import dataclass
from urllib.parse import quote

import aiohttp

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_UPLOAD_URL = "https://yandex.com/images/search"
_RESULT_URL = (
    "https://yandex.com/images/search?rpt=imageview&cbir_page=similar&cbir_id={}"
)
# block 名带 __link 后缀才会返回 cbirId；不带后缀的旧写法返回 blocks:[null]
_UPLOAD_REQUEST = json.dumps(
    {"blocks": [{"block": "b-page_type_search-by-image__link"}]}
)


@dataclass
class SauceEntry:
    """一条相似图结果：源页面 + 原图。"""

    title: str
    domain: str
    page_url: str
    image_url: str
    width: int = 0
    height: int = 0


# ---------------------------------------------------------------------------
# 社媒账号提取
#
# 原理：Yandex 结果的标题/URL 经常直接携带作品作者的署名，典型模式：
#   - "Coser 名 @xxx - Twitter Profile Sotwe"（sotwe.com 等 X 镜像站）
#   - "xxx @yyy - Urlebird"
#   - cosppi.net/user/<id>、x.com/<handle> 等页面 URL
# 这里只做"出处条目里的作者署名解析"，不做跨条目身份聚合。
# ---------------------------------------------------------------------------

# 标题中的 @handle 模式（X 镜像站标题格式："昵称 @handle - ... "）
_TITLE_HANDLE_RE = re.compile(r"@([A-Za-z0-9_]{1,20})")
# URL 中的 X handle
_X_URL_RE = re.compile(
    r"(?:x\.com|twitter\.com|sotwe\.com|urlebird\.com)/(?:user/)?@?([A-Za-z0-9_]{2,20})(?:[/?]|$)"
)

# 平台展示名（按域名归类，用于输出）
_DOMAIN_PLATFORM = [
    ("x.com", "X/Twitter"),
    ("twitter.com", "X/Twitter"),
    ("sotwe.com", "X/Twitter"),
    ("urlebird.com", "X/Twitter"),
    ("weibo.com", "微博"),
    ("weibo.cn", "微博"),
    ("sinaimg.cn", "微博"),
    ("xiaohongshu.com", "小红书"),
    ("xhscdn.com", "小红书"),
    ("instagram.com", "Instagram"),
    ("pixiv.net", "Pixiv"),
    ("danbooru.donmai.us", "Danbooru"),
    ("coser.one", "Cosplay站"),
    ("cosppi.net", "Cosplay站"),
]


@dataclass
class AuthorHandle:
    """从出处条目中解析出的作者署名。"""

    handle: str  # @xxx 的 handle（不含 @）
    platform: str  # 平台名
    source_domain: str  # 从哪个结果域名提取的
    page_url: str  # 对应的出处页面


def _platform_of(domain: str) -> str:
    d = domain.lower()
    for prefix, name in _DOMAIN_PLATFORM:
        if d == prefix or d.endswith("." + prefix):
            return name
    return ""


def extract_authors(entries: list[SauceEntry]) -> list[AuthorHandle]:
    """从出处条目的标题/URL 中提取作者 handle，按出现次数排序去重。

    只保留 X 镜像站与已知社媒域名携带的署名；普通站点标题里的
    @xxx（如邮件地址、正文提及）不算作者。
    """
    counts: dict[tuple[str, str], AuthorHandle] = {}
    order: dict[tuple[str, str], int] = {}
    for e in entries:
        platform = _platform_of(e.domain)
        if not platform:
            continue
        # 1) 标题中的 @handle（镜像站标题固定带作者 handle）
        handle = ""
        m = _TITLE_HANDLE_RE.search(e.title)
        if m:
            handle = m.group(1)
        # 2) 标题没有时，尝试页面 URL 路径中的 handle
        if not handle:
            m = _X_URL_RE.search(e.page_url)
            if m:
                handle = m.group(1)
        if not handle:
            continue
        # 过滤明显的非账号词
        if handle.lower() in {"status", "search", "user", "home", "intent"}:
            continue
        key = (handle.lower(), platform)
        if key not in counts:
            counts[key] = AuthorHandle(
                handle=handle,
                platform=platform,
                source_domain=e.domain,
                page_url=e.page_url,
            )
            order[key] = 0
        order[key] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-order[kv[0]], order[kv[0]]))
    return [v for _, v in ranked]


def format_authors(authors: list[AuthorHandle]) -> str:
    """格式化作者段落。"""
    if not authors:
        return ""
    lines = ["可能的作者账号（按命中次数排序）："]
    for a in authors[:5]:
        lines.append(f"  @{a.handle}  [{a.platform}]  (via {a.source_domain})")
    return "\n".join(lines)


async def search_by_image(
    image_bytes: bytes,
    *,
    timeout: int = 20,
    proxy: str = "",
) -> tuple[str, list[SauceEntry]]:
    """上传图片到 Yandex 并解析相似图结果。

    返回 (result_page_url, entries)。解析失败时 entries 可能为空，
    result_page_url 仍可用于人工查看。
    """
    timeout_cfg = aiohttp.ClientTimeout(total=timeout)
    connector = aiohttp.TCPConnector(ssl=False)  # Yandex 证书链在国内环境偶有不完整
    async with aiohttp.ClientSession(
        timeout=timeout_cfg,
        headers={"User-Agent": _UA},
        connector=connector,
        trust_env=False,  # 不读系统代理环境变量，统一走显式 proxy 配置
    ) as session:
        # 第一步：上传图片（multipart），拿到 cbirId 与 Yandex 托管的原图 URL
        form = aiohttp.FormData()
        form.add_field(
            "upfile", image_bytes, filename="blob", content_type="image/jpeg"
        )
        async with session.post(
            _UPLOAD_URL,
            params={"rpt": "imageview", "format": "json", "request": _UPLOAD_REQUEST},
            data=form,
            proxy=proxy or None,
        ) as resp:
            resp.raise_for_status()
            upload_data = await resp.json(content_type=None)

        blocks = upload_data.get("blocks") or []
        params = (blocks[0] or {}).get("params", {}) if blocks else {}
        cbir_id = params.get("cbirId")
        original_url = params.get("originalImageUrl", "")
        if not cbir_id:
            raise RuntimeError(
                f"Yandex 上传响应中未找到 cbirId: {str(upload_data)[:200]}"
            )

        # 第二步：用 cbir_id 拉取相似图结果页（URL 结构与网页端一致）
        result_url = _RESULT_URL.format(
            quote(f"url={original_url}&cbir_id={cbir_id}", safe="=&")
        )
        async with session.get(result_url, proxy=proxy or None) as resp:
            resp.raise_for_status()
            html = await resp.text()

    entries = _parse_cbir_sites(html)
    return result_url, entries


def _parse_cbir_sites(html: str) -> list[SauceEntry]:
    """从结果页 HTML 中提取 cbirSites JSON 数组。

    Yandex 将结果以转义 JSON 嵌在页面中，需先反转义 HTML 实体，
    再按括号配对切出数组。结构变更时返回空列表而不是抛异常。
    """
    txt = html.replace("&quot;", '"').replace("&#39;", "'").replace("&amp;", "&")
    marker = '"cbirSites":{"sites"'
    i = txt.find(marker)
    if i == -1:
        return []
    j = txt.find("[", i)
    if j == -1:
        return []
    depth = 0
    end = -1
    for k in range(j, len(txt)):
        c = txt[k]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = k
                break
    if end == -1:
        return []
    try:
        raw_list = json.loads(txt[j : end + 1])
    except json.JSONDecodeError:
        return []

    entries: list[SauceEntry] = []
    seen: set[str] = set()
    for item in raw_list:
        page_url = item.get("url", "")
        # 去掉 Yandex 追踪参数（可能以 ?、& 或 URL 编码的 & 连接）
        page_url = re.sub(
            r"[?&]utm_medium=organic(&|%26|&amp;)?utm_source=yandexsmartcamera.*$",
            "",
            page_url,
        )
        page_url = page_url.rstrip("&?")
        if not page_url or page_url in seen:
            continue
        seen.add(page_url)
        orig = item.get("originalImage", {}) or {}
        thumb = item.get("thumb", {}) or {}
        entries.append(
            SauceEntry(
                title=item.get("title", ""),
                domain=item.get("domain", ""),
                page_url=page_url,
                image_url=orig.get("url", "") or thumb.get("url", ""),
                width=orig.get("width", 0),
                height=orig.get("height", 0),
            )
        )
    return entries


def format_entries(entries: list[SauceEntry], limit: int = 8) -> str:
    """格式化结果为纯文本输出（含作者账号段落）。"""
    if not entries:
        return "未找到相似图结果（图片可能未被公开索引过）。"
    authors = extract_authors(entries)
    lines = []
    author_block = format_authors(authors)
    if author_block:
        lines.append(author_block)
    lines.append(f"共 {len(entries)} 条出处（显示前 {min(limit, len(entries))} 条）：")
    for idx, e in enumerate(entries[:limit], 1):
        title = e.title.strip() or "(无标题)"
        lines.append(f"{idx}. [{e.domain}] {title}\n    {e.page_url}")
    return "\n".join(lines)
