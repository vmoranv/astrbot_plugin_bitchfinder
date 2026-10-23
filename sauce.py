"""Yandex Images 反向搜索客户端与结果解析。

仅返回图片出处条目（源页面 URL、标题、域名、原图 URL）。
网络层使用 aiohttp（AstrBot 核心自带依赖）。
"""

import json
import re
from dataclasses import dataclass
from urllib.parse import quote

import aiohttp
from PIL import Image

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
    description: str = ""  # 结果条目的描述文本（推文正文摘要等）
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
# 推文正文（description）中的原作者署名，实测常见形式：
#   "Sunna sc: mamamia1008" / "sc: @yyy" / "📷: @yyy" / "mamamia1008's tweet video"
# 以及直接提及的 @handle
_DESC_CREDIT_RE = re.compile(
    r"(?:sc|credit|cr|source|via|cn|photo|drawn\s+by|tw)\s*[:：]\s*@?([A-Za-z0-9_]{2,20})"
    r"|@([A-Za-z0-9_]{2,20})'(?:s|’s)\s+(?:tweet|post)"
    r"|@([A-Za-z0-9_]{2,20})"
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
    similarity: float = 0.0  # 候选原图与查询图的感知哈希相似度（0-1）
    match_url: str = ""  # 相似度对应的候选原图所在页面
    ranked: bool = False  # 是否已完成图片比对排序
    credit_weight: int = 0  # description 署名权重（>0 表示被正文署名引用过）


def _platform_of(domain: str) -> str:
    d = domain.lower()
    for prefix, name in _DOMAIN_PLATFORM:
        if d == prefix or d.endswith("." + prefix):
            return name
    return ""


def _clean_handle(h: str | None) -> str | None:
    """过滤明显的非账号词，返回规范化 handle 或 None。"""
    if not h:
        return None
    if h.lower() in {"status", "search", "user", "home", "intent", "sunna", "sc"}:
        return None
    return h


def extract_authors(entries: list[SauceEntry]) -> list[AuthorHandle]:
    """从出处条目中提取作者 handle，带来源权重。

    实测教训（mamamia1008 案例）：镜像站标题里的 @handle 往往是
    **转发者/补档号**，真正的原作者署名在推文正文（description）里，
    典型形式："Sunna sc: mamamia1008"、"xxx's tweet video"、
    "📷: @yyy"。因此 description 提权（权重 3），标题/URL 的
    handle 降权（权重 1）。
    """
    weights: dict[tuple[str, str], int] = {}
    samples: dict[tuple[str, str], AuthorHandle] = {}
    for e in entries:
        platform = _platform_of(e.domain)
        if not platform:
            continue
        # 1) description 中的署名（原作者，最高优先）
        #    形式："sc: handle"、"credit: @handle"、"handle's tweet"、"@handle"
        for m in _DESC_CREDIT_RE.finditer(e.description or ""):
            h = _clean_handle(m.group(1))
            if h:
                key = (h.lower(), platform)
                weights[key] = weights.get(key, 0) + 3
                if key not in samples:
                    samples[key] = AuthorHandle(
                        handle=h,
                        platform=platform,
                        source_domain=e.domain,
                        page_url=e.page_url,
                        credit_weight=weights[key],
                    )
                else:
                    samples[key].credit_weight = weights[key]
        # 2) 标题 @handle / URL handle（可能是转发者，低权重）
        m = _TITLE_HANDLE_RE.search(e.title)
        if not m:
            m = _X_URL_RE.search(e.page_url)
        if m:
            h = _clean_handle(m.group(1))
            if h:
                key = (h.lower(), platform)
                weights[key] = weights.get(key, 0) + 1
                samples.setdefault(
                    key,
                    AuthorHandle(
                        handle=h,
                        platform=platform,
                        source_domain=e.domain,
                        page_url=e.page_url,
                    ),
                )
    ranked = sorted(samples.items(), key=lambda kv: -weights[kv[0]])
    return [v for _, v in ranked]


# ---------------------------------------------------------------------------
# 图片比对：感知哈希（aHash + dHash）。Pillow 为 AstrBot 自带依赖。
# ---------------------------------------------------------------------------

_LANCZOS = Image.Resampling.LANCZOS


def _perceptual_hashes(data: bytes) -> tuple[int, int] | None:
    """返回 (aHash64, dHash64)，失败返回 None。"""
    import io

    try:
        img = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        return None
    # aHash: 8x8 均值哈希
    small = img.resize((8, 8), _LANCZOS)
    px: list[int] = list(small.tobytes())
    avg = sum(px) / 64
    ahash = 0
    for bit, v in enumerate(px):
        if v > avg:
            ahash |= 1 << bit
    # dHash: 9x8 相邻差分哈希
    small2 = img.resize((9, 8), _LANCZOS)
    px2: list[int] = list(small2.tobytes())
    dhash = 0
    bit = 0
    for row in range(8):
        for col in range(8):
            if px2[row * 9 + col] > px2[row * 9 + col + 1]:
                dhash |= 1 << bit
            bit += 1
    return ahash, dhash


def _hash_similarity(h1: tuple[int, int], h2: tuple[int, int]) -> float:
    """两组哈希的归一化汉明相似度（0-1）。"""

    def _sim(a: int, b: int) -> float:
        return 1.0 - bin(a ^ b).count("1") / 64.0

    return (_sim(h1[0], h2[0]) + _sim(h1[1], h2[1])) / 2


async def rank_authors_by_image(
    entries: list[SauceEntry],
    query_bytes: bytes,
    *,
    timeout: int = 10,
    proxy: str = "",
    max_candidates: int = 12,
) -> list[AuthorHandle]:
    """提取作者并用条目原图做感知哈希比对，按图片相似度排序。

    对每个带署名的条目：下载其 originalImage（多数为 pbs.twimg.com 等
    公开 CDN，无需绕过反爬），与查询图算相似度；同一账号取最高分。
    """
    authors = extract_authors(entries)
    if not authors:
        return []
    # handle -> 该 handle 关联条目的原图候选
    by_handle: dict[tuple[str, str], list[SauceEntry]] = {}
    for e in entries:
        platform = _platform_of(e.domain)
        if not platform or not e.image_url:
            continue
        handles: list[str] = []
        for m in _DESC_CREDIT_RE.finditer(e.description or ""):
            h = _clean_handle(m.group(1) or m.group(2) or m.group(3) or "")
            if h:
                handles.append(h)
        m = _TITLE_HANDLE_RE.search(e.title) or _X_URL_RE.search(e.page_url)
        if m:
            h = _clean_handle(m.group(1))
            if h:
                handles.append(h)
        for h in set(handles):
            by_handle.setdefault((h.lower(), platform), []).append(e)

    query_hash = _perceptual_hashes(query_bytes)
    if query_hash is None:
        # 查询图不可解码，退化为纯出现次数排序
        return authors

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
        headers={"User-Agent": _UA},
        connector=connector,
        trust_env=False,
    ) as session:
        best: dict[tuple[str, str], float] = {}
        best_url: dict[tuple[str, str], str] = {}
        for author in authors[:max_candidates]:
            key = (author.handle.lower(), author.platform)
            for e in by_handle.get(key, [])[:3]:  # 每账号最多比对 3 张候选图
                try:
                    async with session.get(e.image_url, proxy=proxy or None) as resp:
                        if resp.status != 200:
                            continue
                        cand = _perceptual_hashes(await resp.read())
                except Exception:
                    continue
                if cand is None:
                    continue
                sim = _hash_similarity(query_hash, cand)
                if sim > best.get(key, 0.0):
                    best[key] = sim
                    best_url[key] = e.page_url
        for author in authors:
            key = (author.handle.lower(), author.platform)
            author.similarity = best.get(key, 0.0)
            author.match_url = best_url.get(key, "")

    # 排序：正文署名（原作者）优先；相似度只对 >=75% 的硬确认起作用，
    # 更低的分数（同系列/同风格误报）不参与排序——mamamia 案例教训
    authors.sort(
        key=lambda a: (
            -a.credit_weight,
            -a.similarity if a.similarity >= 0.75 else 0.0,
        )
    )
    for a in authors:
        a.ranked = True
    return authors


def format_authors(authors: list[AuthorHandle]) -> str:
    """格式化作者段落：图片相似度评分 + 检索链接。

    相似度来自候选原图（Yandex 条目的 originalImage）与查询图的
    感知哈希比对；≥0.9 视为同图，0.75-0.9 为可疑，低于 0.75 通常
    只是相关图。检索链接仅供人工点开确认，未做自动化验证。
    """
    if not authors:
        return ""
    lines = ["可能的作者账号（正文署名优先，相似度仅在同图时作数）："]
    for a in authors[:5]:
        sim = f" 图像相似度 {a.similarity:.0%}" if a.similarity else ""
        credit = " [正文署名]" if a.credit_weight else ""
        tag = "✅" if a.similarity >= 0.9 else ("🤔" if a.similarity >= 0.75 else "  ")
        lines.append(f"{tag} @{a.handle}  [{a.platform}]{credit}{sim}")
        lines.append(f"      页面: {a.match_url or a.page_url}")
        lines.append(f"      检索: {site_search_links(a)}")
    return "\n".join(lines)


def site_search_links(author: AuthorHandle) -> str:
    """为作者 handle 生成平台内检索链接，用于确认其主页/原帖。"""
    h = author.handle
    plat = author.platform
    if plat == "X/Twitter":
        # from: 操作符直接搜该账号的发帖；附 nitter 镜像兜底
        return f"https://x.com/search?q=from%3A{h}"
    if plat == "微博":
        return f"https://s.weibo.com/user/?q={quote(h)}"
    if plat == "小红书":
        return f"https://www.xiaohongshu.com/search_result?keyword={quote(h)}"
    if plat == "Instagram":
        return f"https://www.instagram.com/{h}/"
    if plat == "Pixiv":
        return f"https://www.pixiv.net/search.php?s_mode=s_usr&word={quote(h)}"
    # 兜底：通用引擎 site 限定
    return f"https://www.google.com/search?q=site:{author.source_domain}+{quote(h)}"


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
                description=item.get("description", ""),
                width=orig.get("width", 0),
                height=orig.get("height", 0),
            )
        )
    return entries


def format_entries(entries: list[SauceEntry], limit: int = 8) -> str:
    """格式化结果为纯文本输出（含作者账号段落）。

    注意：作者排序需要图片比对（rank_authors_by_image），若已调用
    过则 authors 已带相似度；否则退化为出现次数排序。
    """
    if not entries:
        return "未找到相似图结果（图片可能未被公开索引过）。"
    authors = [a for a in extract_authors(entries) if a.ranked]
    lines = []
    author_block = format_authors(authors) if authors else ""
    if author_block:
        lines.append(author_block)
    lines.append(f"共 {len(entries)} 条出处（显示前 {min(limit, len(entries))} 条）：")
    for idx, e in enumerate(entries[:limit], 1):
        title = e.title.strip() or "(无标题)"
        lines.append(f"{idx}. [{e.domain}] {title}\n    {e.page_url}")
    return "\n".join(lines)
