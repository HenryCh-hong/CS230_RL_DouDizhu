from pathlib import Path
import csv
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "papers"
OUT.mkdir(exist_ok=True)

with (ROOT / "papers.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for i, row in enumerate(rows, 1):
    dest = OUT / row["filename"]
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"[{i}/{len(rows)}] already exists: {dest.name}")
        continue

    print(f"[{i}/{len(rows)}] downloading: {row['title']}")
    req = urllib.request.Request(
        row["pdf_url"],
        headers={"User-Agent": "Mozilla/5.0 CS230-paper-reader/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response, dest.open("wb") as out:
            out.write(response.read())
        print(f"    saved -> {dest}")
    except Exception as exc:
        print(f"    FAILED: {exc}")
        if dest.exists():
            dest.unlink()
    time.sleep(0.5)

print("\nDone. If a source temporarily blocks automated downloads, open its URL from papers.csv in your browser.")
