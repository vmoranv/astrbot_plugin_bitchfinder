"""多引擎反搜适配层。

每个引擎独立实现，返回统一的 PostCandidate 列表；任何引擎失败
（网络、风控、缺 key）都只影响自身，不影响其他引擎。

引擎状态（2026-10 实测）：
- Yandex (.com/.ru)：可用，无 key，同一 CBIR 后端，双域名互为备份
- SauceNAO：官方 API，需免费 api_key（100 次/天）；匿名请求当前被 CF 拦
- ascii2d：当前全站 Cloudflare 挑战，纯 HTTP 抓取不可用；支持传入
  cf_clearance cookie，盾放开或自带 cookie 时恢复工作
- Google Cloud Vision（WEB_DETECTION）：官方付费 API，需 GCP key，
  返回 pages_with_matching_images + 匹配页面 URL
"""

import base64
import json
import re
from dataclasses import dataclass
from urllib.parse import quote

import aiohttp

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


@dataclass
class PostCandidate:
    """一个候选出处：可能带作者、可能带具体帖子 URL。"""

    engine: str  # yandex / saucenao / ascii2d / gcv
    page_url: str  # 出处页面（帖子或镜像页）
    title: str = ""
    author_handle: str = ""  # 不含 @
    author_platform: str = ""  # X/Twitter / Pixiv / ...
    post_url: str = ""  # 明确的原帖 URL（区别于镜像页）
    description: str = ""  # 正文摘要（供二次署名提取）
    image_url: str = ""  # 候选原图
    weight: float = 0.0  # 引擎内部证据权重（融合前）


class EngineError(Exception):
    """引擎级失败（被调用方捕获后降级）。"""


def _make_session(timeout: int, proxy: str) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
        headers={"User-Agent": _UA},
        connector=aiohttp.TCPConnector(ssl=False),
        trust_env=False,
    )


# ---------------------------------------------------------------------------
# Yandex（.com 与 .ru 同一 CBIR 后端，双域名互备）
# ---------------------------------------------------------------------------

_UPLOAD_REQUEST = json.dumps(
    {"blocks": [{"block": "b-page_type_search-by-image__link"}]}
)
_YANDEX_HOSTS = ("yandex.com", "yandex.ru")


async def search_yandex(
    image_bytes: bytes,
    *,
    timeout: int = 20,
    proxy: str = "",
    host: str = "yandex.com",
) -> tuple[str, list[PostCandidate]]:
    """上传图片到 Yandex 并解析相似图结果页。

    返回 (result_page_url, candidates)。两个域名同后端，一个失败自动换另一个。
    """
    last_err: Exception | None = None
    hosts = (host,) + tuple(h for h in _YANDEX_HOSTS if h != host)
    for h in hosts:
        try:
            return await _search_yandex_on(image_bytes, h, timeout=timeout, proxy=proxy)
        except Exception as e:  # noqa: BLE001 - 引擎级降级
            last_err = e
    raise EngineError(f"Yandex 双域名均失败: {last_err}")


async def _search_yandex_on(
    image_bytes: bytes, host: str, *, timeout: int, proxy: str
) -> tuple[str, list[PostCandidate]]:
    async with _make_session(timeout, proxy) as session:
        form = aiohttp.FormData()
        form.add_field(
            "upfile", image_bytes, filename="blob", content_type="image/jpeg"
        )
        async with session.post(
            f"https://{host}/images/search",
            params={"rpt": "imageview", "format": "json", "request": _UPLOAD_REQUEST},
            data=form,
            proxy=proxy or None,
        ) as resp:
            resp.raise_for_status()
            blocks = (await resp.json(content_type=None)).get("blocks") or []
        params = (blocks[0] or {}).get("params", {}) if blocks else {}
        cbir_id = params.get("cbirId")
        original_url = params.get("originalImageUrl", "")
        if not cbir_id:
            raise EngineError("上传响应中未找到 cbirId")

        result_url = (
            f"https://{host}/images/search?rpt=imageview&cbir_page=similar&cbir_id="
            + quote(f"url={original_url}&cbir_id={cbir_id}", safe="=&")
        )
        async with session.get(result_url, proxy=proxy or None) as resp:
            resp.raise_for_status()
            html = await resp.text()

    return result_url, _parse_yandex_html(html)


def _parse_yandex_html(html: str) -> list[PostCandidate]:
    txt = html.replace("&quot;", '"').replace("&#39;", "'").replace("&amp;", "&")
    i = txt.find('"cbirSites":{"sites"')
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

    out: list[PostCandidate] = []
    seen: set[str] = set()
    for item in raw_list:
        page_url = re.sub(
            r"[?&]utm_medium=organic(&|%26|&amp;)?utm_source=yandexsmartcamera.*$",
            "",
            item.get("url", ""),
        ).rstrip("&?")
        if not page_url or page_url in seen:
            continue
        seen.add(page_url)
        orig = item.get("originalImage", {}) or {}
        thumb = item.get("thumb", {}) or {}
        out.append(
            PostCandidate(
                engine="yandex",
                page_url=page_url,
                title=item.get("title", ""),
                description=item.get("description", ""),
                image_url=orig.get("url", "") or thumb.get("url", ""),
            )
        )
    return out


# ---------------------------------------------------------------------------
# SauceNAO：官方 API，直接返回作者与原帖 URL（X status / Pixiv artwork）
# ---------------------------------------------------------------------------


async def search_saucenao(
    image_url: str,
    *,
    api_key: str,
    timeout: int = 25,
    proxy: str = "",
) -> list[PostCandidate]:
    """调用 SauceNAO API（db=999 全库）。image_url 需为公网可访问 URL。"""
    if not api_key:
        raise EngineError("SauceNAO 需要 api_key（saucenao.com 免费注册，100 次/天）")
    params = {
        "db": 999,
        "output_type": 2,
        "numres": 16,
        "api_key": api_key,
        "url": image_url,
    }
    async with _make_session(timeout, proxy) as session:
        async with session.get(
            "https://saucenao.com/search.php", params=params, proxy=proxy or None
        ) as resp:
            if resp.status != 200:
                raise EngineError(
                    f"SauceNAO HTTP {resp.status}（可能被 CF 拦或 key 无效）"
                )
            data = await resp.json(content_type=None)

    if data.get("header", {}).get("status") not in (0, None):
        raise EngineError(f"SauceNAO status={data['header'].get('status')}")

    out: list[PostCandidate] = []
    for result in data.get("results", []):
        d = result.get("data", {})
        ext_urls = d.get("ext_urls", []) or []
        handle = (
            d.get("twitter_user_handle")
            or d.get("author_name")
            or d.get("member_name")
            or ""
        )
        platform = ""
        post_url = ""
        for u in ext_urls:
            if "twitter.com" in u or "x.com" in u:
                platform = "X/Twitter"
                post_url = u
                break
            if "pixiv.net" in u:
                platform = "Pixiv"
                post_url = u
        if not platform and ext_urls:
            post_url = ext_urls[0]
        out.append(
            PostCandidate(
                engine="saucenao",
                page_url=post_url or (ext_urls[0] if ext_urls else ""),
                post_url=post_url,
                title=d.get("title", ""),
                author_handle=handle.lstrip("@"),
                author_platform=platform or ("Pixiv" if d.get("pixiv_id") else ""),
                weight=0.5,
            )
        )
    return out


# ---------------------------------------------------------------------------
# ascii2d：无官方 API，解析彩色搜索结果页；当前被 CF 拦，支持 cf_clearance
# ---------------------------------------------------------------------------


async def search_ascii2d(
    image_bytes: bytes,
    *,
    timeout: int = 30,
    proxy: str = "",
    cf_clearance: str = "",
) -> list[PostCandidate]:
    """上传到 ascii2d 并解析彩色结果页中的原帖链接。"""
    headers = {"User-Agent": _UA}
    cookies = {}
    if cf_clearance:
        cookies["cf_clearance"] = cf_clearance
    async with _make_session(timeout, proxy) as session:
        form = aiohttp.FormData()
        form.add_field("file", image_bytes, filename="blob", content_type="image/jpeg")
        async with session.post(
            "https://ascii2d.net/search/file",
            data=form,
            headers=headers,
            cookies=cookies,
            proxy=proxy or None,
        ) as resp:
            if resp.status == 403 or "Just a moment" in (await resp.text())[:2000]:
                raise EngineError(
                    "ascii2d 被 Cloudflare 拦截：可在配置中提供有效的 cf_clearance cookie"
                )
            resp.raise_for_status()
            html = await resp.text()

    out: list[PostCandidate] = []
    seen: set[str] = set()
    # 结果条目：detail-box 内含外部链接与作者
    for m in re.finditer(r'<div class="detail-box[^"]*"[^>]*>(.*?)</div>', html, re.S):
        block = m.group(1)
        link_m = re.search(r'href="(https?://[^"]+)"[^>]*>', block)
        if not link_m:
            continue
        url = link_m.group(1)
        if url in seen or "ascii2d" in url:
            continue
        seen.add(url)
        handle = ""
        platform = ""
        um = re.search(r"(?:twitter|x)\.com/@?(\w{2,20})(?:/status/(\d+))?", url)
        if um:
            handle = um.group(1)
            platform = "X/Twitter"
        else:
            pm = re.search(r"pixiv\.net/(?:artworks/\d+|users/(\d+))", url)
            if pm:
                handle = pm.group(1) or ""
                platform = "Pixiv"
        title_m = re.search(r"<h4[^>]*>(.*?)</h4>", block, re.S)
        out.append(
            PostCandidate(
                engine="ascii2d",
                page_url=url,
                post_url=url if "/status/" in url or "artworks/" in url else "",
                title=re.sub(r"<[^>]+>", "", title_m.group(1)).strip()
                if title_m
                else "",
                author_handle=handle,
                author_platform=platform,
                weight=0.4,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Google Cloud Vision WEB_DETECTION（用户所说的"旧识图 API"，官方付费）
# ---------------------------------------------------------------------------


async def search_gcv(
    image_bytes: bytes,
    *,
    api_key: str,
    timeout: int = 30,
    proxy: str = "",
) -> list[PostCandidate]:
    """Google Cloud Vision API webDetection：返回完全/部分匹配页面。"""
    if not api_key:
        raise EngineError("GCV 需要 GCP API key（Vision API，webDetection）")
    body = {
        "requests": [
            {
                "image": {"content": base64.b64encode(image_bytes).decode()},
                "features": [{"type": "WEB_DETECTION", "maxResults": 20}],
            }
        ]
    }
    async with _make_session(timeout, proxy) as session:
        async with session.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={quote(api_key)}",
            json=body,
            proxy=proxy or None,
        ) as resp:
            if resp.status != 200:
                raise EngineError(
                    f"GCV HTTP {resp.status}: {(await resp.text())[:150]}"
                )
            data = await resp.json(content_type=None)

    wd = data.get("responses", [{}])[0].get("webDetection", {})
    out: list[PostCandidate] = []
    seen: set[str] = set()
    for section, w in (
        ("pagesWithMatchingImages", 0.3),
        ("fullMatchingImages", 0.2),
        ("partialMatchingImages", 0.15),
    ):
        for item in wd.get(section, []) or []:
            url = item.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(
                PostCandidate(
                    engine="gcv",
                    page_url=url,
                    weight=w,
                )
            )
    return out
