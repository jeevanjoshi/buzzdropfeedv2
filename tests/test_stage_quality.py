"""
Hermetic unit tests for pipeline quality gates (Gate 4, Gate 7) and
video-assembly filtergraph correctness.
No external APIs, no network. All media is synthesised locally with ffmpeg
(if available; tests skip gracefully when ffmpeg is absent).
Covers: Gate 7 good-pass, black-frame fail, mute-audio fail, duration-drift
fail, crossfade-range pass; Gate 4 resolution check + measured-duration skip;
crossfade filtergraph offset math.
"""
import os
import re
import shutil
import subprocess
import tempfile

from src.schemas.state import GlobalState, AssetPaths, ScriptData
from src.engine.quality_verifier import StageQualityVerifier


_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _skip(msg="ffmpeg not available"): pass


def _color_clip(path, color, dur=3.0, audio_dur=3.0):
    """Generate a synthetic 1920×1080 H.264 MP4 with a solid colour + sine tone."""
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:size=1920x1080:rate=25:duration={dur}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={audio_dur}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", path,
    ], capture_output=True)


def _mute_clip(path, color="red", dur=3.0):
    """Same as above but with anullsrc (silent audio)."""
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:size=1920x1080:rate=25:duration={dur}",
        "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=stereo:duration={dur}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", path,
    ], capture_output=True)


def _concat_clips(paths, output):
    """Concatenate a list of MP4 clips with ffmpeg copy concat."""
    tmp = tempfile.mktemp(suffix=".txt")
    with open(tmp, "w") as f:
        for p in paths:
            f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", tmp,
                    "-c", "copy", output], capture_output=True)
    os.remove(tmp)


def _probe_dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", path],
                       capture_output=True, text=True, timeout=30)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Gate 7 — render integrity
# ---------------------------------------------------------------------------

def test_gate7_passes_good():
    if not _HAS_FFMPEG: return
    d = tempfile.mkdtemp()
    c = [os.path.join(d, f"g_{i}.mp4") for i in range(3)]
    for p in c:
        _color_clip(p, "blue", dur=3.0)
    final = os.path.join(d, "final.mp4")
    _concat_clips(c, final)
    st = GlobalState(pipeline_id="g7ok", timestamp="x",
                     script_data=ScriptData(title="t", target_shots=3, shots=[],
                                            estimated_runtime_seconds=9))
    st.asset_paths = AssetPaths(
        visuals={f"shot_{i}": c[i] for i in range(3)},
        audio={f"shot_{i}": c[i] for i in range(3)},
        final_video=final,
        measured_durations=[_probe_dur(p) for p in c],
        crossfade_used=0.0,
    )
    ok, issues = StageQualityVerifier().verify_gate7_render_integrity(st)
    assert ok, f"expected pass, got issues: {issues}"
    print("✓ Gate 7 passes good media (no black frames, no silence, duration matched)")


def test_gate7_black_frames():
    if not _HAS_FFMPEG: return
    d = tempfile.mkdtemp()
    good = os.path.join(d, "good.mp4"); black = os.path.join(d, "black.mp4")
    _color_clip(good, "blue", dur=3.0)
    _color_clip(black, "black", dur=3.0)
    st = _dummy_state(d, {"shot_1": good, "shot_2": black},
                      {"shot_1": good, "shot_2": black},
                      measured=[3.0, 3.0], cf=0.0)
    ok, issues = StageQualityVerifier().verify_gate7_render_integrity(st)
    assert not ok, "expected black-frame fail"
    assert any("black" in i.lower() for i in issues), issues
    print("✓ Gate 7 fails black frame shots")


def test_gate7_mute_audio():
    if not _HAS_FFMPEG: return
    d = tempfile.mkdtemp()
    good = os.path.join(d, "good.mp4"); mute = os.path.join(d, "mute.mp4")
    _color_clip(good, "blue", dur=3.0)
    _mute_clip(mute, "red", dur=3.0)
    st = _dummy_state(d, {"shot_1": good, "shot_2": mute},
                      {"shot_1": good, "shot_2": mute},
                      measured=[3.0, 3.0], cf=0.0)
    ok, issues = StageQualityVerifier().verify_gate7_render_integrity(st)
    assert not ok, "expected mute fail"
    assert any("silent" in i.lower() for i in issues), issues
    print("✓ Gate 7 fails mute narration shots")


def test_gate7_duration_drift():
    if not _HAS_FFMPEG: return
    d = tempfile.mkdtemp()
    c = [os.path.join(d, f"d_{i}.mp4") for i in range(3)]
    for p in c:
        _color_clip(p, "red", dur=3.0)
    short = os.path.join(d, "short.mp4")
    _concat_clips(c[:2], short)  # only 2 clips => ~6s, not 9s
    st = _dummy_state(d, {}, {}, final=short, measured=[3.0, 3.0, 3.0], cf=0.0)
    ok, issues = StageQualityVerifier().verify_gate7_render_integrity(st)
    assert not ok, "expected duration drift fail"
    assert any("outside" in i.lower() for i in issues), issues
    print("✓ Gate 7 fails duration drift (final shorter than measured sum)")


def test_gate7_crossfade_range():
    """Crossfade-compressed timeline vs hard-cut fallback — both pass."""
    if not _HAS_FFMPEG: return
    d = tempfile.mkdtemp()
    c = [os.path.join(d, f"x_{i}.mp4") for i in range(3)]
    for p in c:
        _color_clip(p, "green", dur=3.0)
    final = os.path.join(d, "final_x.mp4")
    _concat_clips(c, final)  # hard-cut concat, not crossfade => 9s
    st = _dummy_state(d, {}, {}, final=final,
                      measured=[3.0, 3.0, 3.0], cf=0.5)
    ok, issues = StageQualityVerifier().verify_gate7_render_integrity(st)
    assert ok, f"expected pass (fallback concat within range), got issues: {issues}"
    print("✓ Gate 7 crossfade range accommodates both xfade and fallback concat")


def test_gate7_dark_but_valid_no_false_positive():
    """A legitimately dark but non-black clip must NOT be flagged as a failed render."""
    if not _HAS_FFMPEG: return
    d = tempfile.mkdtemp()
    clip = os.path.join(d, "dark.mp4")
    # solid very-dark grey (mean ~17, above the mean<8 black threshold) + audible tone
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=#111111:size=1920x1080:rate=25:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", clip,
    ], capture_output=True)
    st = _dummy_state(d, {"shot_1": clip}, {"shot_1": clip},
                      measured=[3.0], cf=0.0)
    ok, issues = StageQualityVerifier().verify_gate7_render_integrity(st)
    assert ok, f"dark-but-valid clip should pass, got issues: {issues}"
    assert not any("black" in i.lower() for i in issues), issues
    print("✓ Gate 7 no false positive on dark-but-valid clip")


def test_gate7_partial_black_tolerated():
    """A single black frame among many must NOT trip the gate (documented 60% threshold)."""
    if not _HAS_FFMPEG: return
    d = tempfile.mkdtemp()
    clip = os.path.join(d, "partial.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:size=1920x1080:rate=25:duration=1",
        "-f", "lavfi", "-i", "color=c=red:size=1920x1080:rate=25:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", clip,
    ], capture_output=True)
    st = _dummy_state(d, {"shot_1": clip}, {"shot_1": clip},
                      measured=[3.0], cf=0.0)
    ok, issues = StageQualityVerifier().verify_gate7_render_integrity(st)
    assert ok, f"1 black frame / 3 sampled should pass (tolerance), got: {issues}"
    print("✓ Gate 7 documents black-frame tolerance (single black frame not a false fail)")


def test_gate7_empty_assets_edge():
    """Dry-run: no visuals/audio/final -> gate passes without crashing."""
    d = tempfile.mkdtemp()
    st = _dummy_state(d, {}, {}, final=None, measured=[], cf=0.0)
    ok, issues = StageQualityVerifier().verify_gate7_render_integrity(st)
    assert ok, f"empty-asset edge should pass, got issues: {issues}"
    print("✓ Gate 7 edge: empty asset maps pass cleanly (dry-run)")


def test_crossfade_single_clip_edge():
    """N<2 -> build returns None (falls back to hard-cut concat)."""
    import importlib.util as iu
    spec = iu.spec_from_file_location("mc", "mcp_servers/media_cloud/server.py")
    mc = iu.module_from_spec(spec)
    spec.loader.exec_module(mc)
    out = tempfile.mktemp(suffix=".mp4")
    assert mc._build_crossfade_cmd(["/tmp/only.mp4"], [5.0], 0.5, "fade", "", out, bgm_path=None) is None
    assert mc._build_crossfade_cmd([], [], 0.5, "fade", "", out, bgm_path=None) is None
    print("✓ crossfade edge: single/zero clips return None (safe fallback)")


def _dummy_state(d, visuals, audio, final=None, measured=None, cf=0.0):
    st = GlobalState(pipeline_id="g7", timestamp="x",
                     script_data=ScriptData(title="t", target_shots=len(visuals or {}),
                                            shots=[],
                                            estimated_runtime_seconds=sum(measured or [0])))
    fv = final or os.path.join(d, "final.mp4")
    if not os.path.exists(fv) and measured:
        _concat_clips(list(visuals.values()), fv)
    st.asset_paths = AssetPaths(
        visuals=visuals, audio=audio,
        final_video=fv,
        measured_durations=list(measured or []),
        crossfade_used=cf,
    )
    return st


# ---------------------------------------------------------------------------
# Gate 4 — resolution + measured-duration skip
# ---------------------------------------------------------------------------

def test_gate4_resolution():
    if not _HAS_FFMPEG: return
    d = tempfile.mkdtemp()
    bad = os.path.join(d, "bad_res.mp4")
    # 320x180 is not 1920x1080
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:size=320x180:rate=25:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-c:v", "libx264", "-c:a", "aac", bad], capture_output=True)
    good = os.path.join(d, "good_res.mp4")
    _color_clip(good, "blue", dur=2.0)
    for label, path, measured, expect_pass in [
        ("bad res", bad, [1.0], False),
        ("good res", good, [1.0], True),
    ]:
        st = GlobalState(pipeline_id="g4", timestamp="x",
                         script_data=ScriptData(title="t", target_shots=1, shots=[],
                                                estimated_runtime_seconds=2.0))
        st.asset_paths = AssetPaths(final_video=path, measured_durations=measured)
        ok, issues = StageQualityVerifier().verify_gate4_video_audio_coherence(st)
        if expect_pass:
            assert ok, f"{label}: expected pass, got {issues}"
        else:
            assert not ok and any("resolution" in i.lower() for i in issues), f"{label}: {issues}"
    print("✓ Gate 4 checks resolution and skips estimate-based duration when measured available")


# ---------------------------------------------------------------------------
# Crossfade filtergraph offset math (pure, no ffmpeg needed)
# ---------------------------------------------------------------------------

def test_crossfade_filtergraph_math():
    import importlib.util as iu
    spec = iu.spec_from_file_location("mc", "mcp_servers/media_cloud/server.py")
    mc = iu.module_from_spec(spec)
    spec.loader.exec_module(mc)
    out = tempfile.mktemp(suffix=".mp4")
    cmd = mc._build_crossfade_cmd(
        ["/tmp/a.mp4", "/tmp/b.mp4", "/tmp/c.mp4"],
        [3.0, 4.0, 5.0], 0.5, "fade", "",
        out, bgm_path=None,
    )
    assert cmd is not None, "crossfade-cmd should not be None for N>=2"
    fc = " ".join(cmd)
    assert "xfade" in fc, "xfade filter missing"
    # offset for first transition (between clip0 and clip1)
    # = prefix[0] - 1*cf = 3.0 - 0.5 = 2.5
    assert "offset=2.500" in fc, "xfade offset 2.5 missing"
    # offset for second transition = prefix[1] - 2*cf = 7.0 - 1.0 = 6.0
    assert "offset=6.000" in fc, "xfade offset 6.0 missing"
    assert "acrossfade=d=0.500" in fc, "acrossfade duration missing"
    assert "format=yuv420p" in fc, "pixel format uniform encoding missing"
    assert "fps=25" in fc, "uniform framerate missing"
    # single clip -> no crossfade
    cmd2 = mc._build_crossfade_cmd(["/tmp/single.mp4"], [5.0], 0.5, "fade", "", out, bgm_path=None)
    assert cmd2 is None, "N<2 should return None"
    print("✓ crossfade filtergraph offset math (xfade offsets, acrossfade d, uniform format)")