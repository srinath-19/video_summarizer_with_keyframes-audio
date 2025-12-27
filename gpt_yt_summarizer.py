import os
from dotenv import load_dotenv
import re
from pathlib import Path
from typing import List

from openai import OpenAI

load_dotenv()

ARTIFACTS_DIR = Path("artifacts")
TRANSCRIPT_PATH = ARTIFACTS_DIR / "transcript.txt"
OUT_SUMMARY_PATH = ARTIFACTS_DIR / "summary.md"

# Keep chunks small enough to be safe across models
MAX_CHARS_PER_CHUNK = 12_000  # ~2k-3k tokens depending on text


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def split_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Splits transcript into chunks close to max_chars, preferring to split on blank lines.
    If no blank lines, splits on sentence-ish boundaries.
    """
    text = text.strip()
    if not text:
        return []

    # First try splitting by lines, since your transcript is line-based.
    lines = text.splitlines()
    chunks = []
    buf = []

    cur_len = 0
    for line in lines:
        line_len = len(line) + 1
        if cur_len + line_len > max_chars and buf:
            chunks.append("\n".join(buf).strip())
            buf = []
            cur_len = 0
        buf.append(line)
        cur_len += line_len

    if buf:
        chunks.append("\n".join(buf).strip())

    # Fallback: if something is still too large, hard split
    final = []
    for ch in chunks:
        if len(ch) <= max_chars:
            final.append(ch)
        else:
            # hard split by character count
            for i in range(0, len(ch), max_chars):
                final.append(ch[i : i + max_chars])
    return final


def clean_timestamps(transcript_chunk: str) -> str:
    """
    Optional: remove [MM:SS - MM:SS] so the model focuses on content.
    """
    return re.sub(r"\[\d{2}:\d{2}(?::\d{2})?\s*-\s*\d{2}:\d{2}(?::\d{2})?\]\s*", "", transcript_chunk)


def summarize_chunk(client: OpenAI, chunk_text: str, idx: int, total: int) -> str:
    prompt = f"""
You are summarizing a transcript chunk ({idx}/{total}).
Write:
1) 6-10 bullet key points
2) important names/terms (if any)
3) any decisions, steps, or concrete takeaways
Be faithful to the transcript. If uncertain, say so.
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": "You produce concise, accurate summaries of transcripts.",
            },
            {
                "role": "user",
                "content": prompt + "\n\nTRANSCRIPT CHUNK:\n" + chunk_text,
            },
        ],
    )
    return resp.output_text.strip()


def combine_summaries(client: OpenAI, chunk_summaries: List[str]) -> str:
    joined = "\n\n".join(
        [f"### Chunk {i+1}\n{txt}" for i, txt in enumerate(chunk_summaries)]
    )

    prompt = """
You are given summaries of transcript chunks.
Produce a final structured summary in markdown with:

## Overview (3-5 sentences)
## Key points (8-12 bullets)
## Notable details (optional)
## Action items / next steps (if any)
## Open questions (if any)

Avoid repeating the same point. Keep it crisp.
If the content is mostly narrative, focus on themes and examples.
"""

    resp = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "You write clean, structured markdown summaries."},
            {"role": "user", "content": prompt + "\n\nCHUNK SUMMARIES:\n" + joined},
        ],
    )
    return resp.output_text.strip()


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY env var.")

    client = OpenAI(api_key=api_key)

    transcript = read_text(TRANSCRIPT_PATH)
    if not transcript.strip():
        raise SystemExit(f"Transcript is empty: {TRANSCRIPT_PATH}")

    chunks = split_into_chunks(transcript, MAX_CHARS_PER_CHUNK)

    # Optional: remove timestamps before summarizing
    cleaned_chunks = [clean_timestamps(c) for c in chunks]

    chunk_summaries = []
    for i, ch in enumerate(cleaned_chunks, start=1):
        chunk_summaries.append(summarize_chunk(client, ch, i, len(cleaned_chunks)))

    final_summary = combine_summaries(client, chunk_summaries)

    OUT_SUMMARY_PATH.write_text(final_summary + "\n", encoding="utf-8")
    print("Wrote:", OUT_SUMMARY_PATH)


if __name__ == "__main__":
    main()
