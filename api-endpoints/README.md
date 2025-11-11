# API Endpoint Specialists

This directory contains specialized agents for integrating with specific third-party APIs and services. Each agent is an expert in a particular API's endpoints, authentication, data models, and best practices.

## Purpose

API endpoint specialists differ from generic backend specialists in that they:

- Have deep knowledge of a specific API's documentation and behavior
- Understand the nuances of a particular service's endpoints and data models
- Can fetch and reference live API documentation when needed
- Follow service-specific best practices and patterns
- Handle service-specific error conditions and edge cases

## Structure

Each API specialist agent follows this pattern:

```
api-endpoints/
├── {service}-api-specialist.md     # Agent definition
└── README.md                        # This file
```

## Current API Specialists

### **bloxs-api-specialist**
**Bloxs.io property management API integration expert**
Python/Django integration specialist for the Bloxs API. Handles property management data models including entities, leases, financials, tasks, and cases. Consults live documentation at `https://www.bloxs.io/apidocs/` and local `bloxs_docs.jsonl` files.

**Documentation:** Uses the Bloxs scraper in `scrapers/bloxs_scraper.py` to generate `bloxs_docs.jsonl` for offline reference.

**When to use:**
- Integrating Django applications with Bloxs API
- Mapping Bloxs data models to local Django models
- Debugging Bloxs API calls and responses
- Writing tests for Bloxs integrations

**Setup:**
```bash
cd api-endpoints/scrapers
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python bloxs_scraper.py
```

## Adding New API Specialists

When creating a new API endpoint specialist:

1. **Create the agent file**: `{service}-api-specialist.md`
2. **Include these sections**:
   - YAML frontmatter with metadata
   - Mission statement
   - Documentation sources (live docs URLs, local docs files)
   - When to use this agent
   - Core responsibilities
   - Documentation lookup protocol
   - Standard workflow for implementations
   - Answer style guidelines
   - Safety and compliance rules

3. **Key requirements**:
   - Agent must reference authoritative documentation sources
   - Should specify how to access live API docs
   - Must define local documentation file patterns (e.g., `{service}_docs.jsonl`)
   - Should handle documentation discrepancies gracefully
   - Must never invent endpoints or parameters not in official docs

4. **Update this README** with:
   - Agent name and description
   - When to use guidance
   - Key features or specializations

## Best Practices

### Documentation Strategy
- Always prefer live API documentation as source of truth
- Maintain local documentation snapshots for offline reference
- Make it explicit which documentation source was used
- Handle documentation versioning and changes gracefully

### Implementation Patterns
- Separate configuration from business logic
- Create testable HTTP client wrappers
- Handle authentication consistently
- Implement proper error handling for network and API errors
- Use type hints for API request/response structures

### Testing
- Test with realistic payloads from documentation
- Mock external API calls appropriately
- Validate request structure and headers
- Test error handling for common failure modes

### Security
- Never hardcode credentials or API keys
- Use environment variables or secret managers
- Implement proper credential scoping
- Add audit logging for sensitive operations

## Integration with Other Agents

API endpoint specialists work well with:

- **django-specialist**: For Django-specific integration patterns
- **django-rest-framework-specialist**: For building Django REST APIs that wrap external APIs
- **test-writer**: For comprehensive integration test suites
- **debugger-detective**: For troubleshooting integration issues
- **security-auditor**: For validating API security practices
- **orchestrator-agent**: For complex multi-service integrations
