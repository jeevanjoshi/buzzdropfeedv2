"""
Hermetic unit tests for MediaProducer synchronization logic.
No external APIs, no network, no LLM. All data is synthetic (temp files/wave).
Covers: scene-cue parsing, visual-prompt enrichment, subtitle-offset merging
(including crossfade-aware offsets), WAV duration probing, market-symbol
selection, market-quote parsing (monkeypatched fetch), TTS text sanitization,
ass->srt conversion, and renderer/crossfade parameter wiring.
"""
import os
import re
import wave
import struct
import shutil
import tempfile
import asyncio

from src.schemas.state import GlobalState, TopicCandidate
from src.agents.media_producer import (
    MediaProducerAgent,
    parse_scene_visual_cue,
    enrich_visual_prompt,
    merge_ass_subtitle_files,
    _probe_wav_duration,
)
from mcp_servers.audio_edge.server import sanitize_tts_text


def _make_wav(path: str, duration_sec: float, sample_rate: int = 24000):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * int(sample_rate * duration_sec))


def _make_ass(path: str, dialogues):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        for start, end, text in dialogues:
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")


def _topic(keywords, region="global") -> TopicCandidate:
    return TopicCandidate(
        candidate_id="synth-topic",
        headline="Synthetic Headline",
        summary="Synthetic summary.",
        source_url="https://example.invalid/synth",
        keywords=keywords,
        tvs_score=50.0, rpm_score=0.5, idi_score=0.5, sdi_score=0.5, sat_score=1.0,
        region=region,
    )


def test_scene_cue_parsing():
    narr = "Markets are moving [Scene: epic city skyline at night] fast."
    clean, cue = parse_scene_visual_cue(narr)
    assert "Scene:" not in clean
    assert "epic city skyline at night" in cue
    assert "Markets are moving" in clean and "fast." in clean
    # no cue present -> clean unchanged, empty cue
    clean2, cue2 = parse_scene_visual_cue("No cue here.")
    assert clean2 == "No cue here." and cue2 == ""
    print("✓ scene-cue parsing (with/without [Scene:])")


def test_visual_prompt_enrichment():
    out = enrich_visual_prompt("A cool market shot", 1, 1)
    assert "16:9 widescreen" in out
    assert "8k photorealistic" in out
    assert "dramatic cinematic opening shot" in out  # Act 1 = hook tone
    # already-has anchors are not duplicated
    out2 = enrich_visual_prompt("16:9 widescreen 8k photorealistic scene", 6, 15)
    assert out2.count("16:9") == 1 and out2.count("8k") == 1
    assert "final frame cinematic outro" in out2  # Act 6 = verdict tone
    print("✓ visual prompt enrichment (act-tones + no dup anchors)")


def test_subtitle_merge_offsets():
    d = tempfile.mkdtemp()
    a1 = os.path.join(d, "s1.ass"); a2 = os.path.join(d, "s2.ass")
    _make_ass(a1, [("0:00:00.20", "0:00:02.20", "FIRST")])
    _make_ass(a2, [("0:00:00.20", "0:00:02.20", "SECOND")])
    out = os.path.join(d, "master.ass")
    merge_ass_subtitle_files([a1, a2], [3.0, 4.0], out, crossfade=0.0)
    text = open(out).read()
    starts = re.findall(r"Dialogue: 0,([0-9:.]+),", text)
    assert starts == ["0:00:00.20", "0:00:03.20"], starts  # shot2 = 3.0 + 0.20
    assert "SECOND" in text
    print("✓ subtitle merge offsets (no crossfade)")


def test_subtitle_merge_crossfade():
    d = tempfile.mkdtemp()
    a1 = os.path.join(d, "s1.ass"); a2 = os.path.join(d, "s2.ass")
    _make_ass(a1, [("0:00:00.20", "0:00:02.20", "FIRST")])
    _make_ass(a2, [("0:00:00.20", "0:00:02.20", "SECOND")])
    out = os.path.join(d, "master.ass")
    merge_ass_subtitle_files([a1, a2], [3.0, 4.0], out, crossfade=0.5)
    starts = re.findall(r"Dialogue: 0,([0-9:.]+),", open(out).read())
    # shot2 offset = (3.0 - 0.5) + 0.20 = 2.70 (timeline compressed by the dissolve)
    assert starts == ["0:00:00.20", "0:00:02.70"], starts
    print("✓ subtitle merge offsets (crossfade-compressed)")


def test_probe_wav_duration():
    d = tempfile.mkdtemp()
    wav = os.path.join(d, "t.wav")
    _make_wav(wav, 2.5)
    dur = _probe_wav_duration(wav)
    assert dur is not None and abs(dur - 2.5) < 0.05, dur
    assert _probe_wav_duration(os.path.join(d, "missing.wav")) is None
    print("✓ wav duration probing (wave module)")


def test_market_symbol_picker():
    agent = MediaProducerAgent(storage_dir=tempfile.mkdtemp())
    state = GlobalState(pipeline_id="p", timestamp="t", selected_topic=_topic(["NVDA", "chips"]))
    assert agent._pick_symbol(state) == "NVDA"
    state = GlobalState(pipeline_id="p", timestamp="t", region="india",
                        selected_topic=_topic(["reliance", "markets"]))
    assert agent._pick_symbol(state) == "RELIANCE.NS"
    state = GlobalState(pipeline_id="p", timestamp="t", region="global",
                        selected_topic=_topic(["banks"]))
    assert agent._pick_symbol(state) == "SPY"
    print("✓ market symbol picker (keyword ticker / region defaults)")


def test_market_quote_parse():
    import src.engine.external_apis as ea
    orig = ea.ExternalAPIManager.fetch_alpha_vantage_stock_quote
    ea.ExternalAPIManager.fetch_alpha_vantage_stock_quote = (
        lambda self, symbol: {"symbol": symbol, "price": "$650.00", "change": "+3.50%"}
    )
    try:
        agent = MediaProducerAgent(storage_dir=tempfile.mkdtemp())
        state = GlobalState(pipeline_id="p", timestamp="t", selected_topic=_topic(["NVDA"]))
        q = asyncio.run(agent._get_market_quote(state))
        assert q["symbol"] == "NVDA" and q["price"] == 650.0 and q["change"] == 3.5, q
        # cached -> no second fetch
        q2 = asyncio.run(agent._get_market_quote(state))
        assert q2 is q
        # fallback when fetch raises
        ea.ExternalAPIManager.fetch_alpha_vantage_stock_quote = lambda self, s: (_ for _ in ()).throw(RuntimeError("boom"))
        agent2 = MediaProducerAgent(storage_dir=tempfile.mkdtemp())
        qf = asyncio.run(agent2._get_market_quote(state))
        assert qf["price"] > 0, qf  # graceful fallback numbers
    finally:
        ea.ExternalAPIManager.fetch_alpha_vantage_stock_quote = orig
    print("✓ market quote parse (live + fallback, monkeypatched fetch)")


def test_tts_text_sanitize():
    assert sanitize_tts_text("hello <break time='300ms'/> world") == "hello world"
    assert sanitize_tts_text("cost $100 billion") == "cost 100 billion dollars"
    assert "?" in sanitize_tts_text("Really? Yes!")
    assert sanitize_tts_text("SEO and API") == "search engine optimization and A P I"
    assert "break time" not in sanitize_tts_text("a <break/> forward slash")
    print("✓ tts text sanitization (SSML stripped, numbers/acronyms expanded, punctuation kept)")


def test_ass_to_srt():
    agent = MediaProducerAgent(storage_dir=tempfile.mkdtemp())
    d = tempfile.mkdtemp()
    ass = os.path.join(d, "m.ass"); srt = os.path.join(d, "m.srt")
    _make_ass(ass, [("0:00:00.50", "0:00:02.50", "HELLO WORLD")])
    agent._ass_to_srt(ass, srt)
    text = open(srt).read()
    assert "00:00:00,500 --> 00:00:02,500" in text and "HELLO WORLD" in text
    print("✓ ass -> srt conversion")


def test_producer_renderer_params():
    agent = MediaProducerAgent(storage_dir=tempfile.mkdtemp(), renderer="moviepy", crossfade=1.2)
    assert agent.renderer == "moviepy" and agent.crossfade == 1.2
    agent2 = MediaProducerAgent(storage_dir=tempfile.mkdtemp(), crossfade=-3)
    assert agent2.crossfade == 0.0  # negative clamped
    print("✓ producer renderer/crossfade parameter wiring")


def test_subtitle_merge_empty_edge():
    """Empty ass list / missing files -> valid header-only master, no crash."""
    d = tempfile.mkdtemp()
    out = os.path.join(d, "master.ass")
    merge_ass_subtitle_files([], [], out, crossfade=0.5)
    assert os.path.exists(out)
    assert "[Events]" in open(out).read()
    assert "Dialogue:" not in open(out).read()
    merge_ass_subtitle_files([os.path.join(d, "missing.ass")], [3.0], out, crossfade=0.5)
    assert os.path.exists(out)
    print("✓ subtitle merge edge: empty/missing inputs produce valid header-only master")
