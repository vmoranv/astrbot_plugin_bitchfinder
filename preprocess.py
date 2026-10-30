"""图片预处理：提升反搜命中率。

参考 flj_finder 的做法：
- 符合条件的 JPEG/PNG/WebP（无需纠正方向、最长边 <=1600px）保留
  原始字节——重新编码会改变图像内容，伤害反向搜索命中率；
- 其他图片纠正 EXIF 方向、按最长边 1600 缩放、透明区填白转 JPEG、
  动图只取第一帧。
"""

import io

from PIL import Image, ImageOps

_MAX_SIDE = 1600
_JPEG_QUALITY = 92

_EXIF_ORIENTATION_TAG = 274  # 0x0112 Orientation


def prepare_image(data: bytes) -> bytes:
    """返回适合送入反搜引擎的字节；不可解码时原样返回。"""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return data

    fmt = (img.format or "").upper()
    orientation = 1
    try:
        orientation = int(img.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
    except Exception:
        orientation = 1

    if (
        fmt in ("JPEG", "PNG", "WEBP")
        and max(img.width, img.height) <= _MAX_SIDE
        and orientation <= 1
    ):
        return data

    # EXIF 方向纠正
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # 动图取第一帧（seek(0) 即当前帧）
    if getattr(img, "is_animated", False):
        img.seek(0)

    # 透明区填白后转 RGB
    if img.mode in ("RGBA", "LA", "P") or "transparency" in (img.info or {}):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # 最长边缩放
    if max(img.width, img.height) > _MAX_SIDE:
        img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.Resampling.LANCZOS)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=_JPEG_QUALITY)
    return out.getvalue()
