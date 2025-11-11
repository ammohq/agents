"""
Bloxs API Documentation Scraper

Crawls all Bloxs /apidocs/ pages and writes bloxs_docs.jsonl for LLM ingestion.

Usage:
    python bloxs_scraper.py [--output OUTPUT_FILE]

Requirements:
    pip install requests beautifulsoup4
"""

import argparse
import json
import re
from collections import deque
from urllib.parse import urljoin, urldefrag

import requests
from bs4 import BeautifulSoup

START_URL = "https://www.bloxs.io/apidocs/welcome"
ALLOWED_PREFIXES = [
    "https://www.bloxs.io/apidocs/",
    "https://bloxs.document360.io/apidocs/",
]


def create_session():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; BloxsDocScraper/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def normalize_url(base_url, href):
    if not href:
        return None
    href, _ = urldefrag(href)
    abs_url = urljoin(base_url, href)
    for prefix in ALLOWED_PREFIXES:
        if abs_url.startswith(prefix):
            return abs_url
    return None


def extract_content(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""
    article = soup.find("article")
    if not article:
        candidates = soup.find_all("div", class_=re.compile(r"(article|content|kb)", re.I))
        article = candidates[0] if candidates else None
    node = article or soup.body or soup
    text = "\n".join(s.strip() for s in node.stripped_strings)
    return title, text


def crawl(start_url, session):
    visited = set()
    queue = deque([start_url])
    docs = []

    print(f"Starting crawl from: {start_url}")

    while queue:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        print(f"Crawling: {url}")

        try:
            resp = session.get(url, timeout=20)
            if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                print(f"  Skipping (status={resp.status_code})")
                continue
        except requests.RequestException as e:
            print(f"  Error: {e}")
            continue

        title, content = extract_content(resp.text)
        if content.strip():
            docs.append(
                {
                    "url": url,
                    "title": title,
                    "content": content,
                }
            )
            print(f"  Extracted: {title}")

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            next_url = normalize_url(resp.url, a["href"])
            if next_url and next_url not in visited:
                queue.append(next_url)

    return docs


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Bloxs API documentation and generate JSONL file"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="bloxs_docs.jsonl",
        help="Output file path (default: bloxs_docs.jsonl)",
    )
    args = parser.parse_args()

    session = create_session()
    documents = crawl(START_URL, session)

    with open(args.output, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nCrawl complete!")
    print(f"Total pages: {len(documents)}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
