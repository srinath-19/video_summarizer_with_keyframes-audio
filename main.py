import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, transcribe, and summarize a YouTube video.",
    )
    parser.add_argument("youtube_url", help="YouTube video URL")
    args = parser.parse_args()

    run([sys.executable, "yt_summarize.py", args.youtube_url])
    run([sys.executable, "gpt_yt_summarizer.py"])


if __name__ == "__main__":
    main()
