"""fxTwitter 公共接口：读取 X 账号的粉丝数（无需任何 key）。

用途：多个候选账号无法确定谁是作者时，取已取得粉丝数中
最多的一个作为最终展示（仅用于候选排序，不作为同图证明）。
"""

import aiohttp

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


async def fetch_followers(
    handles: list[str],
    *,
    timeout: int = 10,
    proxy: str = "",
) -> dict[str, int | None]:
    """批量查询 handle -> 粉丝数；查询失败/不存在映射为 None。"""
    out: dict[str, int | None] = {}
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
        headers={"User-Agent": _UA},
        connector=connector,
        trust_env=False,
    ) as session:
        for h in handles[:6]:
            try:
                async with session.get(
                    f"https://api.fxtwitter.com/{h}", proxy=proxy or None
                ) as resp:
                    if resp.status != 200:
                        out[h.lower()] = None
                        continue
                    data = await resp.json(content_type=None)
                user = data.get("user") or {}
                followers = user.get("followers")
                out[h.lower()] = int(followers) if isinstance(followers, int) else None
            except Exception:
                out[h.lower()] = None
    return out


def pick_by_followers(
    candidates: list[str],
    followers: dict[str, int | None],
    fallback_order: list[str],
) -> str:
    """在候选中选粉丝数最多的；数量缺失或并列时按 fallback_order。"""
    known = [(h.lower(), followers.get(h.lower())) for h in candidates]
    known = [(h, f) for h, f in known if f is not None]
    if known:
        max_f = max(f for _, f in known)
        winners = [h for h, f in known if f == max_f]
        if len(winners) == 1:
            return winners[0]
    for h in fallback_order:
        if h.lower() in {c.lower() for c in candidates}:
            return h
    return candidates[0]
