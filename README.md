# Youtube Video Summarizer

A small, end-to-end pipeline that:

- Downloads audio from a YouTube video
- Transcribes it with Faster-Whisper (timestamps included)
- Generates rough, time-based chapters
- Summarizes the transcript with the OpenAI API

The repo also includes a small website scraping utility (`scraper.py`) for fetching page text and links.

## Features

- YouTube audio download via `yt-dlp`
- Speech-to-text using Faster-Whisper (CPU-friendly)
- Timestamped transcript output
- Automatic time-based chapter generation
- Chunked summarization for long videos
- OpenAI-powered final markdown summary

## Project Structure

- `yt_summarize.py`: download audio, transcribe, and create chapters
- `gpt_yt_summarizer.py`: summarize `artifacts/transcript.txt` with OpenAI
- `main.py`: run the full pipeline (download, transcribe, summarize)
- `scraper.py`: fetch website text and links
- `artifacts/`: generated outputs
- `summarizer.ipynb`: notebook exploration

## Requirements

- Python 3.11+
- `ffmpeg` on PATH
- OpenAI API key for the summarization step

## Setup

### Option A: uv (recommended)

```bash
uv sync
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate llms
```

### Environment variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

## Usage

### 1) Download, transcribe, and generate chapters

```bash
uv run python yt_summarize.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Outputs:

- `artifacts/transcript.txt` (timestamped transcript)
- `artifacts/chapters.md` (rough chapters)

Default transcription settings (see `yt_summarize.py`):

- Model: `small`
- Device: `cpu`
- Compute type: `int8`
- VAD enabled

### 2) Generate the AI summary

```bash
uv run python gpt_yt_summarizer.py
```

Outputs:

- `artifacts/summary.md`

The summary includes:

- Overview
- Key points
- Notable details (optional)
- Action items / next steps (if any)
- Open questions (if any)

### One-command pipeline

```bash
uv run python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Website Utilities

The `scraper.py` module provides small helpers to fetch website text and links:

```python
from scraper import fetch_website_contents, fetch_website_links

text = fetch_website_contents("https://example.com")
links = fetch_website_links("https://example.com")
```

## Notes

- Chapter titles are time-based (not semantic).
- Longer videos will require more OpenAI calls; cost scales with length.
- You can safely delete `artifacts/` between runs.

## Troubleshooting

### ffmpeg not found

Install ffmpeg and ensure it is on PATH:

- Windows: https://www.gyan.dev/ffmpeg/builds/
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

### OPENAI_API_KEY not found

- Ensure `.env` exists in the project root.
- Restart your terminal after creating `.env`.
