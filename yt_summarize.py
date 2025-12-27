import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict

from faster_whisper import WhisperModel


def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print("Command failed:\n", " ".join(cmd), "\n\nSTDERR:\n", p.stderr, sep="")
        raise SystemExit(p.returncode)


def sanitize_filename(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    return s.strip()[:120] if s else "video"


def format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def download_audio(youtube_url: str, out_dir: Path) -> Tuple[Path, str]:
    """
    Downloads best audio and converts to wav for transcription.
    Returns (wav_path, title).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get metadata (title)
    meta_cmd = [sys.executable, "-m", "yt_dlp", "-J", youtube_url]
    p = subprocess.run(meta_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print("yt-dlp metadata failed:\n", p.stderr)
        raise SystemExit(p.returncode)
    info = json.loads(p.stdout)
    title = info.get("title", "video")
    safe_title = sanitize_filename(title)

    # Download audio to a known filename
    audio_path = out_dir / f"{safe_title}.m4a"
    run([
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "-o", str(out_dir / f"{safe_title}.%(ext)s"),
        youtube_url
    ])


    # yt-dlp chooses extension; find the actual downloaded file
    candidates = list(out_dir.glob(f"{safe_title}.*"))
    if not candidates:
        raise FileNotFoundError("No downloaded audio found.")
    downloaded = max(candidates, key=lambda p: p.stat().st_size)

    # Convert to wav (16k mono) for Whisper
    wav_path = out_dir / f"{safe_title}.wav"
    run([
        "ffmpeg",
        "-y",
        "-i", str(downloaded),
        "-ac", "1",
        "-ar", "16000",
        str(wav_path)
    ])

    return wav_path, title


def transcribe_with_timestamps(
    wav_path: Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
) -> List[Dict]:
    """
    Returns a list of segments: {start, end, text}
    """
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(wav_path),
        vad_filter=True,
        word_timestamps=False,  # segment-level timestamps are enough for chapters
    )
    out = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            out.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return out


def make_chapters(
    segments: List[Dict],
    chapter_seconds: int = 120,
    min_chars: int = 200,
) -> List[Tuple[int, str]]:
    """
    Simple timestamp chaptering:
    - bucket transcript into fixed time windows (default 2 minutes)
    - use first meaningful line as chapter title
    """
    if not segments:
        return []

    total_end = int(max(s["end"] for s in segments))
    n = math.ceil(total_end / chapter_seconds)

    chapters = []
    seg_idx = 0

    for i in range(n):
        start_t = i * chapter_seconds
        end_t = (i + 1) * chapter_seconds

        bucket = []
        while seg_idx < len(segments) and segments[seg_idx]["start"] < end_t:
            if segments[seg_idx]["end"] > start_t:
                bucket.append(segments[seg_idx]["text"])
            seg_idx += 1

        text = " ".join(bucket).strip()
        if len(text) < min_chars:
            continue

        # crude title: first sentence-ish chunk
        title = re.split(r"[.!?]\s+", text)[0]
        title = title[:90].strip()
        if not title:
            title = "Section"

        chapters.append((start_t, title))

    # de-dup titles
    deduped = []
    seen = set()
    for ts, title in chapters:
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((ts, title))
    return deduped


def save_transcript(segments: List[Dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"[{format_ts(s['start'])} - {format_ts(s['end'])}] {s['text']}\n")


def save_chapters(chapters: List[Tuple[int, str]], out_path: Path, video_title: str, youtube_url: str) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# {video_title}\n\n")
        f.write(f"Source: {youtube_url}\n\n")
        f.write("## Chapters\n\n")
        for ts, title in chapters:
            f.write(f"- {format_ts(ts)} — {title}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python yt_summarize.py <YOUTUBE_URL>")
        raise SystemExit(2)

    youtube_url = sys.argv[1]
    out_dir = Path("artifacts")
    wav_path, title = download_audio(youtube_url, out_dir)

    # For your RTX 3060 6GB, "small" or "medium" is the sweet spot.
    segments = transcribe_with_timestamps(
    wav_path,
    model_size="small",
    device="cpu",
    compute_type="int8"
    )


    transcript_path = out_dir / "transcript.txt"
    save_transcript(segments, transcript_path)

    chapters = make_chapters(segments, chapter_seconds=120)
    chapters_path = out_dir / "chapters.md"
    save_chapters(chapters, chapters_path, title, youtube_url)

    print("Wrote:")
    print(" -", transcript_path)
    print(" -", chapters_path)


if __name__ == "__main__":
    main()
