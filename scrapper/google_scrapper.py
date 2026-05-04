#!/usr/bin/env python3
# pip install requests ddgs Pillow
# (uninstall duckduckgo-search if you still have it: pip uninstall duckduckgo-search)
"""
Download face images of 16 Pakistani politicians into data/raw/<class_name>/.
Uses Google Custom Search API and falls back to DuckDuckGo (via ddgs).
"""

import os
import time
import requests
from io import BytesIO
from PIL import Image
from ddgs import DDGS          # <-- new official package

# Load API keys from environment variables first. Avoid hard-coding secrets in this file.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CX = os.environ.get("GOOGLE_CX") or os.environ.get("GOOGLE_CSE_ID") or os.environ.get("GOOGLE_CUSTOM_SEARCH_CX")

# Optional local secrets file (gitignored). Copy `scrapper/_secrets.example.py` -> `scrapper/_secrets.py` and fill.
_local_api = None
_local_cx = None
try:
    # Try package-relative import (when used as a module)
    from ._secrets import GOOGLE_API_KEY as _LOCAL_GOOGLE_API_KEY, GOOGLE_CX as _LOCAL_GOOGLE_CX
    _local_api = _LOCAL_GOOGLE_API_KEY
    _local_cx = _LOCAL_GOOGLE_CX
except Exception:
    try:
        # Try top-level import (when run as a script)
        from _secrets import GOOGLE_API_KEY as _LOCAL_GOOGLE_API_KEY, GOOGLE_CX as _LOCAL_GOOGLE_CX
        _local_api = _LOCAL_GOOGLE_API_KEY
        _local_cx = _LOCAL_GOOGLE_CX
    except Exception:
        _local_api = None
        _local_cx = None

if not GOOGLE_API_KEY and _local_api:
    GOOGLE_API_KEY = _local_api
if not GOOGLE_CX and _local_cx:
    GOOGLE_CX = _local_cx

TARGET_PER_CLASS = 250
DELAY_BETWEEN_REQUESTS = 2.0   # slightly larger to stay polite
OUTPUT_ROOT = os.path.join("data", "raw")
CLASS_NAMES = [
    "imran_khan", "nawaz_sharif", "asif_ali_zardari", "bilawal_bhutto",
    "shahbaz_sharif", "maryam_nawaz", "fazlur_rehman", "asfandyar_wali",
    "altaf_hussain", "chaudhry_shujaat", "pervez_musharraf", "shehryar_afridi",
    "khawaja_asif", "ahsan_iqbal", "barrister_gohar", "ahmed_sharif_chaudhry"
]

SEARCH_QUERIES_SUFFIXES = [
    "portrait photo",
    "speech Pakistan",
    "press conference",
    "rally",
    "parliament",
    "رسمی تصویر"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ImageDownloader/1.0)"}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def count_existing_images(folder):
    if not os.path.isdir(folder):
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    return sum(1 for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts)


def save_image_bytes(img_bytes, out_path):
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.save(out_path, format="JPEG", quality=85)
        return True
    except Exception:
        return False


def fetch_google_image_links(query, api_key, cx, start=1, num=10):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "searchType": "image",
        "imgType": "face",
        "safe": "active",
        "start": start,
        "num": num,
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    except Exception:
        return None, "request_failed"
    if resp.status_code != 200:
        return None, f"status_{resp.status_code}"
    try:
        data = resp.json()
    except Exception:
        return None, "invalid_json"
    items = data.get("items", []) or []
    links = []
    for it in items:
        link = it.get("link") or it.get("image", {}).get("contextLink")
        if link:
            links.append(link)
    return links, "ok"


def fetch_ddg_image_links(query, max_results=100):
    try:
        # ddgs might also rate-limit, so add a small sleep before calling
        time.sleep(1.0)
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_results))
    except Exception as e:
        print(f"DuckDuckGo error: {e}")
        return []
    links = []
    for r in results or []:
        link = r.get("image")
        if link:
            links.append(link)
    return links


def download_image(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, stream=True)
        if resp.status_code == 200:
            content = resp.content
            return content
    except Exception:
        return None
    return None


def gather_for_class(class_name, target, google_key, google_cx):
    class_folder = ensure_dir(os.path.join(OUTPUT_ROOT, class_name))
    existing = count_existing_images(class_folder)
    if existing >= target:
        print(f"Skipping {class_name}: already has {existing} images (>= {target})")
        return 0
    added = 0
    seq = existing + 1
    full_name = class_name.replace("_", " ")
    queries = [f"{full_name} {suffix}" for suffix in SEARCH_QUERIES_SUFFIXES]
    queries.append(full_name)

    use_google = bool(google_key and google_key != "YOUR_GOOGLE_API_KEY" and google_cx and google_cx != "YOUR_CUSTOM_SEARCH_ENGINE_ID")
    for q in queries:
        if added >= (target - existing):
            break
        remaining = target - (existing + added)

        # Try Google first
        if use_google:
            start = 1
            while remaining > 0 and start <= 91:
                links, status = fetch_google_image_links(q, google_key, google_cx, start=start, num=10)
                time.sleep(DELAY_BETWEEN_REQUESTS)
                if links is None:
                    use_google = False
                    break
                if not links:
                    break
                for link in links:
                    if added >= remaining:
                        break
                    img_bytes = download_image(link)
                    if not img_bytes:
                        continue
                    out_name = f"{seq:04d}.jpg"
                    out_path = os.path.join(class_folder, out_name)
                    if save_image_bytes(img_bytes, out_path):
                        seq += 1
                        added += 1
                start += 10
                remaining = target - (existing + added)

        # Fallback to DuckDuckGo
        if added >= (target - existing):
            break
        ddg_needed = target - (existing + added)
        ddg_links = fetch_ddg_image_links(q, max_results=min(200, ddg_needed * 2 + 50))
        time.sleep(DELAY_BETWEEN_REQUESTS)
        for link in ddg_links:
            if added >= ddg_needed:
                break
            img_bytes = download_image(link)
            if not img_bytes:
                continue
            out_name = f"{seq:04d}.jpg"
            out_path = os.path.join(class_folder, out_name)
            if save_image_bytes(img_bytes, out_path):
                seq += 1
                added += 1
    return added


def main():
    summary = {}
    for cls in CLASS_NAMES:
        print(f"Processing class: {cls}")
        added = gather_for_class(cls, TARGET_PER_CLASS, GOOGLE_API_KEY, GOOGLE_CX)
        summary[cls] = added
        print(f"Added {added} images to {cls}\n")
    print("Summary of new images added:")
    for cls, added in summary.items():
        print(f"- {cls}: {added}")
    total_added = sum(summary.values())
    print(f"Total new images: {total_added}")


if __name__ == "__main__":
    main()