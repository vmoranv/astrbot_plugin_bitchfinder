"""真机 smoke 测试：对比预处理前后的反搜命中数，不发任何 QQ 消息。

用法（在 AstrBot 的 Python 环境执行）：
    python scripts/smoke_ab.py /path/to/test.jpg
    python scripts/smoke_ab.py /path/to/test.jpg --timeout 30
    python scripts/smoke_ab.py /path/to/test.jpg --proxy http://127.0.0.1:7890

输出：原始图 vs prepare_image 处理后，各自跑一次 Yandex（+可选百度），
对比命中条数、作者署名提取数、感知哈希确认数。
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 允许从仓库根目录直接 import 包（namespace package）
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from astrbot_plugin_bitchfinder import engines
from astrbot_plugin_bitchfinder.fuse import fuse
from astrbot_plugin_bitchfinder.preprocess import prepare_image
from astrbot_plugin_bitchfinder.sauce import _hash_similarity, _perceptual_hashes


async def run_once(label: str, image: bytes, *, timeout: int, proxy: str) -> dict:
    print(f"\n=== {label} ({len(image) / 1024:.0f}KB) ===")
    yandex_url, candidates = await engines.search_yandex(
        image, timeout=timeout, proxy=proxy
    )
    yx = [c for c in candidates if c.engine == "yandex"]
    print(f"Yandex 结果页: {yandex_url}")
    print(f"出处条目: {len(yx)} 条")

    verdicts = fuse(candidates)
    strong = [v for v in verdicts if v.confidence >= 0.30]
    print(f"融合候选: {len(verdicts)} 个（可靠署名 {len(strong)} 个）")
    for v in strong[:3]:
        print(f"  @{v.handle} [{v.platform}] conf={v.confidence:.2f} {v.post_url or v.page_url}")

    # 感知哈希确认数（下载候选原图比对）
    qh = _perceptual_hashes(image)
    confirmed = 0
    if qh:
        import aiohttp

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            connector=aiohttp.TCPConnector(ssl=False),
            trust_env=False,
        ) as session:
            for c in yx[:12]:
                if not c.image_url:
                    continue
                try:
                    async with session.get(c.image_url, proxy=proxy or None) as resp:
                        if resp.status != 200:
                            continue
                        ch = _perceptual_hashes(await resp.read())
                except Exception:
                    continue
                if ch and _hash_similarity(qh, ch) >= 0.90:
                    confirmed += 1
    print(f"感知哈希同图确认(>=90%): {confirmed} 条")
    return {"entries": len(yx), "strong": len(strong), "confirmed": confirmed}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="测试图片路径")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--proxy", default="")
    args = ap.parse_args()

    raw = Path(args.image).read_bytes()
    processed = prepare_image(raw)
    same = raw == processed
    print(f"预处理: {'未改变(保留原字节)' if same else '已处理'} "
          f"{len(raw) / 1024:.0f}KB -> {len(processed) / 1024:.0f}KB")

    a = await run_once("原始图", raw, timeout=args.timeout, proxy=args.proxy)
    b = await run_once("处理后", processed, timeout=args.timeout, proxy=args.proxy)

    print("\n=== 对比 ===")
    for k, name in (("entries", "出处条目"), ("strong", "可靠署名"), ("confirmed", "同图确认")):
        print(f"{name}: {a[k]} -> {b[k]}  ({'↑' if b[k] > a[k] else '↓' if b[k] < a[k] else '='})")


if __name__ == "__main__":
    asyncio.run(main())
