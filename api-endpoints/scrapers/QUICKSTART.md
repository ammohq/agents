# Quick Start: Bloxs Documentation Scraper

Generate `bloxs_docs.jsonl` in 3 steps:

## 1. Setup (one time)

```bash
cd agents/api-endpoints/scrapers

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Run Scraper

```bash
python bloxs_scraper.py
```

This creates `bloxs_docs.jsonl` in the current directory.

## 3. Copy to Your Project

```bash
# Recommended location
cp bloxs_docs.jsonl /path/to/your-project/docs/bloxs/bloxs_docs.jsonl

# Alternative locations
cp bloxs_docs.jsonl /path/to/your-project/docs/bloxs_docs.jsonl
cp bloxs_docs.jsonl /path/to/your-project/bloxs_docs.jsonl
```

## Using the Documentation File

The `bloxs-api-specialist` agent will automatically find and search this file.

### Manual Search Examples

Search for specific endpoints:
```bash
grep "POST /api/leases" docs/bloxs/bloxs_docs.jsonl
```

Extract specific page:
```bash
cat docs/bloxs/bloxs_docs.jsonl | jq 'select(.title | contains("Authentication"))'
```

Search content:
```bash
rg "pagination" docs/bloxs/bloxs_docs.jsonl
```

## Updating Documentation

Re-run the scraper when:
- Bloxs releases API updates
- You need documentation for new endpoints
- Monthly maintenance (recommended)

```bash
cd agents/api-endpoints/scrapers
source .venv/bin/activate
python bloxs_scraper.py
cp bloxs_docs.jsonl /path/to/your-project/docs/bloxs/bloxs_docs.jsonl
```

## Troubleshooting

**Scraper hangs or times out:**
- Check your internet connection
- Verify https://www.bloxs.io/apidocs/welcome is accessible
- Try again later (may be temporary rate limiting)

**Empty or incomplete output:**
- Check if the Bloxs docs site structure changed
- Review scraper output for error messages
- Open an issue in the agents repository

**Agent can't find bloxs_docs.jsonl:**
- Verify file exists in one of these locations:
  - `docs/bloxs/bloxs_docs.jsonl` (recommended)
  - `docs/bloxs_docs.jsonl`
  - `bloxs_docs.jsonl` (root)
- Check file permissions are readable
