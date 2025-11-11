# API Documentation Scrapers

This directory contains utilities for scraping third-party API documentation into JSONL format for LLM consumption.

## Purpose

API endpoint specialists need access to accurate, up-to-date API documentation. Rather than hardcoding API knowledge, we:

1. Scrape official documentation into structured JSONL files
2. Store these files in target project repositories
3. Let the specialist agents search and reference them dynamically

This approach ensures:
- Agents always reference actual documentation
- Documentation can be updated independently
- No hallucination of API endpoints or parameters
- Clear audit trail of information sources

## Available Scrapers

### Bloxs API Scraper

**File:** `bloxs_scraper.py`

Crawls the Bloxs API documentation at https://www.bloxs.io/apidocs/ and generates a JSONL file with all documentation pages.

#### Setup

```bash
cd api-endpoints/scrapers

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install requests beautifulsoup4
```

#### Usage

```bash
# Generate bloxs_docs.jsonl in current directory
python bloxs_scraper.py

# Specify custom output location
python bloxs_scraper.py --output /path/to/project/docs/bloxs/bloxs_docs.jsonl
```

#### Output Format

Each line in `bloxs_docs.jsonl` is a JSON object:

```json
{
  "url": "https://www.bloxs.io/apidocs/endpoint-name",
  "title": "Endpoint Name - Bloxs API",
  "content": "Full text content of the documentation page..."
}
```

#### Integration with Projects

Once you've generated `bloxs_docs.jsonl`, copy it to your target project:

```bash
# Recommended locations (in order of preference):
cp bloxs_docs.jsonl /path/to/project/docs/bloxs/bloxs_docs.jsonl
cp bloxs_docs.jsonl /path/to/project/docs/bloxs_docs.jsonl
cp bloxs_docs.jsonl /path/to/project/bloxs_docs.jsonl
```

The `bloxs-api-specialist` agent will automatically search for this file in these locations.

## Creating New Scrapers

When adding a scraper for a new API:

### 1. Create the Scraper File

```python
# api-endpoints/scrapers/{service}_scraper.py

"""
{Service} API Documentation Scraper

Crawls {Service} documentation and writes {service}_docs.jsonl for LLM ingestion.
"""

import json
import requests
from bs4 import BeautifulSoup

START_URL = "https://api.example.com/docs"
ALLOWED_PREFIXES = ["https://api.example.com/docs/"]

def crawl(start_url):
    # Implementation here
    pass

if __name__ == "__main__":
    documents = crawl(START_URL)
    with open("{service}_docs.jsonl", "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
```

### 2. Key Considerations

**Respect robots.txt**
- Check the service's robots.txt before scraping
- Add appropriate delays between requests
- Use a descriptive User-Agent

**Handle Authentication**
- Some docs require API keys or login
- Document auth requirements clearly
- Store credentials securely (environment variables)

**URL Filtering**
- Only crawl documentation URLs
- Exclude navigation, search, login pages
- Remove URL fragments (#anchors)

**Content Extraction**
- Target main content areas (article, main, content divs)
- Strip scripts, styles, navigation
- Preserve code examples and structure
- Extract meaningful titles

**Error Handling**
- Handle network timeouts gracefully
- Skip non-HTML content
- Log skipped/failed URLs
- Implement retry logic for transient failures

### 3. Testing the Scraper

```bash
# Test on a small subset first
python {service}_scraper.py --limit 10

# Validate output format
cat {service}_docs.jsonl | jq -c 'select(.url == null or .content == "")'

# Check for duplicates
cat {service}_docs.jsonl | jq -r .url | sort | uniq -d
```

### 4. Update the Agent

Create or update `api-endpoints/{service}-api-specialist.md` with:

- Location patterns for `{service}_docs.jsonl`
- Instructions for running the scraper
- Examples of searching the JSONL file
- Fallback strategy if file doesn't exist

## Maintenance

### When to Re-scrape

Re-run scrapers when:
- API documentation is updated (check changelogs)
- New endpoints are added
- Parameters or responses change
- Authentication methods change

### Automation

Consider automating scraper runs:

```bash
# Weekly cron job to update docs
0 2 * * 0 cd /path/to/agents/api-endpoints/scrapers && ./update_all.sh
```

```bash
#!/bin/bash
# update_all.sh

source .venv/bin/activate

python bloxs_scraper.py --output ../../docs/bloxs_docs.jsonl
# Add more scrapers as needed

echo "Documentation updated: $(date)" >> scraper.log
```

## Best Practices

### Storage Strategy

**Option 1: Store in Agent Repository (Current Approach)**
- ✅ Documentation available for reference
- ✅ Version controlled with agent
- ❌ Large files bloat the repo
- ❌ Needs manual updates

**Option 2: Store in Target Project**
- ✅ Project-specific documentation versions
- ✅ Can be updated per project
- ✅ Smaller agent repository
- ❌ Requires scraper distribution

**Option 3: Cloud Storage with Cache**
- ✅ Central documentation source
- ✅ Easy updates
- ✅ Small repositories
- ❌ Requires network access
- ❌ Additional infrastructure

**Recommendation:** Use Option 2 (store in target project) for most cases.

### Search Performance

For large JSONL files:

```bash
# Use ripgrep for fast searches
rg "endpoint pattern" docs/bloxs_docs.jsonl

# Use jq for structured queries
cat docs/bloxs_docs.jsonl | jq 'select(.title | contains("Lease"))'

# Index with grep for common searches
grep -n "POST /api/" docs/bloxs_docs.jsonl > docs/bloxs_endpoints.idx
```

### Documentation Versioning

Track API documentation versions:

```bash
docs/
├── bloxs/
│   ├── bloxs_docs.jsonl          # Current version
│   ├── bloxs_docs_v2.1.0.jsonl   # Versioned backup
│   └── VERSION.txt                # Current API version
```

## Troubleshooting

### Scraper Hangs

- Add timeout to requests: `requests.get(url, timeout=20)`
- Limit crawl depth
- Add visited URL tracking

### Missing Content

- Inspect HTML structure with browser DevTools
- Check for JavaScript-rendered content (may need Selenium)
- Verify content selector logic

### Duplicate URLs

- Normalize URLs (remove trailing slashes, query params)
- Remove URL fragments before comparison
- Track visited URLs by normalized form

### Rate Limiting

- Add delays: `time.sleep(1)` between requests
- Implement exponential backoff
- Cache responses to avoid re-fetching
- Use session pooling for better connection management
