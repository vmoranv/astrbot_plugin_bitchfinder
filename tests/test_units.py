import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from astrbot_plugin_bitchfinder import engines, fuse, llm_candidates, preprocess, social
from astrbot_plugin_bitchfinder.engines import PostCandidate


def _png(w=100, h=100) -> bytes:
    img = Image.new("RGBA", (w, h), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_small_png_kept_as_is():
    data = _png()
    assert preprocess.prepare_image(data) == data


def test_oversized_png_resized_to_jpeg():
    data = _png(2000, 1000)
    out = preprocess.prepare_image(data)
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG"
    assert max(img.width, img.height) <= 1600


def test_gif_takes_first_frame_as_jpeg():
    img = Image.new("RGB", (50, 50), "blue")
    buf = io.BytesIO()
    img.save(buf, format="GIF", save_all=True, append_images=[img])
    out = preprocess.prepare_image(buf.getvalue())
    assert Image.open(io.BytesIO(out)).format == "JPEG"


def test_garbage_bytes_returned_unchanged():
    assert preprocess.prepare_image(b"not-an-image") == b"not-an-image"


# ---------------------------------------------------------------------------
# llm_candidates
# ---------------------------------------------------------------------------


SOURCES = [
    {"title": "Sunna @mamamia1008 - Twitter Profile Sotwe", "source": "sotwe.com", "link": "https://sotwe.com/mamamia1008"},
    {"title": "随手拍", "source": "weibo.com", "link": "https://weibo.com/xxx"},
]


def _cands(json_text):
    return llm_candidates.parse_candidates(json_text, SOURCES)


def test_valid_candidate_accepted():
    raw = json.dumps(
        {"candidates": [{"handle": "mamamia1008", "source_index": 0,
                          "evidence": "@mamamia1008 - Twitter Profile",
                          "confidence": 0.8}]}
    )
    out = _cands(raw)
    assert len(out) == 1
    assert out[0]["handle"] == "mamamia1008"
    assert out[0]["platform"] == "X/Twitter"


def test_fabricated_evidence_rejected():
    raw = json.dumps(
        {"candidates": [{"handle": "fake_user", "source_index": 0,
                          "evidence": "this text never existed",
                          "confidence": 0.9}]}
    )
    assert _cands(raw) == []


def test_bad_handle_rejected():
    raw = json.dumps(
        {"candidates": [{"handle": "bad handle!", "source_index": 0,
                          "evidence": "Sunna @mamamia1008",
                          "confidence": 0.5}]}
    )
    assert _cands(raw) == []


def test_out_of_range_index_rejected():
    raw = json.dumps(
        {"candidates": [{"handle": "mamamia1008", "source_index": 9,
                          "evidence": "Twitter Profile", "confidence": 0.5}]}
    )
    assert _cands(raw) == []


def test_evidence_without_handle_rejected():
    raw = json.dumps(
        {"candidates": [{"handle": "mamamia1008", "source_index": 0,
                          "evidence": "Sunna", "confidence": 0.5}]}
    )
    assert _cands(raw) == []


def test_negative_clause_not_platform():
    raw = json.dumps(
        {"candidates": [{"handle": "someone", "source_index": 1,
                          "evidence": "随手拍 不是 x.com/someone 的",
                          "confidence": 0.5}]}
    )
    out = _cands(raw)
    assert out == [] or out[0]["platform"] == "unknown"


def test_invalid_json_returns_empty():
    assert _cands("not json") == []
    assert _cands("") == []


def test_prompt_is_bounded_text_only():
    prompt = llm_candidates.build_prompt(SOURCES)
    assert "sources" in prompt
    assert len(prompt) < 4000


# ---------------------------------------------------------------------------
# fuse
# ---------------------------------------------------------------------------


def _c(engine="yandex", **kw):
    return PostCandidate(engine=engine, page_url=kw.pop("page_url", "https://example.com/x"), **kw)


def test_credit_description_outranks_title_handle():
    c1 = _c(title="someone - Sotwe", description="sc: real_author", page_url="https://sotwe.com/someone")
    c2 = _c(title="real_author - Sotwe", page_url="https://sotwe2.com/real_author")
    verdicts = fuse.fuse([c1, c2])
    assert verdicts[0].handle.lower() == "real_author"


def test_phash_same_image_bonus():
    c1 = _c(title="aaa - Sotwe", page_url="https://sotwe.com/aaa")
    c2 = _c(title="bbb - Sotwe", page_url="https://other.com/bbb")
    verdicts = fuse.fuse([c1, c2], {"aaa": 0.95, "bbb": 0.30})
    a = next(v for v in verdicts if v.handle == "aaa")
    assert a.confidence > 0.10


def test_format_shows_single_best_and_folds_rest():
    cs = [
        _c(description=f"sc: user{i}", page_url=f"https://s{i}.com/u{i}")
        for i in range(3)
    ]
    verdicts = fuse.fuse(cs)
    text = fuse.format_verdicts(verdicts)
    assert "最优先候选" in text
    assert text.count("@user") <= 1 or "折叠" in text or "其余" in text


# ---------------------------------------------------------------------------
# engines：yandex HTML 解析（fixture 内嵌 cbirSites JSON）
# ---------------------------------------------------------------------------


def test_parse_yandex_html():
    page = {"url": "https://sotwe.com/abc?utm_medium=organic&utm_source=yandexsmartcamera",
            "title": "x @user1", "description": "sc: user2",
            "domain": "sotwe.com",
            "originalImage": {"url": "https://img/1.jpg", "width": 10, "height": 10},
            "thumb": {"url": "https://img/t.jpg"}}
    dup = {"url": "https://sotwe.com/abc?utm_medium=organic&utm_source=yandexsmartcamera",
           "title": "dup", "domain": "sotwe.com"}
    # 还原 _parse_yandex_html 期望的转义形态："cbirSites":{"sites":[...]}
    sites_json = json.dumps({"sites": [page, dup]})
    html = "&quot;cbirSites&quot;:" + sites_json.replace('"', "&quot;")
    out = engines._parse_yandex_html(html)
    assert len(out) == 1
    assert out[0].page_url == "https://sotwe.com/abc"
    assert out[0].image_url == "https://img/1.jpg"


def test_captcha_html_raises_in_search_but_parser_safe():
    # 解析器对验证页返回空列表（不抛异常）；验证页判定在 search_yandex 层
    assert engines._parse_yandex_html("<html>ShowCaptcha</html>") == []


# ---------------------------------------------------------------------------
# engines：saucenao 解析（离线构造响应）
# ---------------------------------------------------------------------------


def test_saucenao_requires_key():
    import asyncio

    with pytest.raises(engines.EngineError):
        asyncio.run(engines.search_saucenao("https://x/y.jpg", api_key=""))


# ---------------------------------------------------------------------------
# social
# ---------------------------------------------------------------------------


def test_pick_by_followers():
    followers = {"a": 100, "b": 5000, "c": None}
    assert social.pick_by_followers(["a", "b", "c"], followers, ["a", "b"]) == "b"
    # 都查不到时按 fallback 顺序
    assert social.pick_by_followers(["a", "c"], followers, ["a", "c"]) == "a"
    # 并列时按 fallback
    assert social.pick_by_followers(["a", "a"], followers, ["a"]) == "a"


def test_google_lens_requires_key():
    import asyncio

    with pytest.raises(engines.EngineError):
        asyncio.run(engines.search_google_lens("https://x/y.jpg", api_key=""))


# ---------------------------------------------------------------------------
# fuse：聚合站通用词过滤（真机 smoke 暴露的回归）
# ---------------------------------------------------------------------------


def test_generic_title_words_not_registered():
    # APK/工具站标题 "Best Chat Apps 2025" 类通用词不应成为"可靠署名"
    cs = [
        _c(title="Chat", page_url="https://site1.com/chat-app"),
        _c(title="Download", page_url="https://site2.com/download"),
        _c(title="Android", page_url="https://site3.com/android"),
    ]
    verdicts = fuse.fuse(cs)
    handles = {v.handle.lower() for v in verdicts if v.platform != "未知"}
    unknown = {v.handle.lower() for v in verdicts if v.platform == "未知"}
    assert not ({"chat", "android", "download"} & handles)
    assert not ({"chat", "android", "download"} & unknown)


def test_digit_handle_still_registered():
    cs = [
        _c(title="pon9019 Nude Leaks Photo #1", page_url="https://a.com/pon9019"),
        _c(title="pon9019 Gallery", page_url="https://b.com/pon9019-pics"),
    ]
    verdicts = fuse.fuse(cs)
    assert any(v.handle == "pon9019" and v.platform == "未知" for v in verdicts)


def test_noise_word_blocklist_expanded():
    for w in ("Chat", "Android", "Download", "APK", "Online", "Watch"):
        assert not fuse._is_handle_like(w)
    assert fuse._is_handle_like("pon9019")
    assert fuse._is_handle_like("real_author")
