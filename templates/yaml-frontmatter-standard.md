# YAML Frontmatter Standard for Agents

All agent definitions MUST include a YAML frontmatter block with these fields:

## Required Fields

```yaml
---
name: string           # Lowercase, hyphen-separated identifier (e.g., "api-designer")
version: string        # Semantic version (e.g., "1.0.0", "2.1.3")
description: string    # One-line activation trigger with clear use cases
model: string          # One of: "inherit", "opus", "sonnet", "haiku", or specific model ID
---
```

## Optional Fields

```yaml
---
tools: array          # List of required tools (e.g., ["Read", "Write", "Bash"])
dependencies: array   # Other agents this one depends on (e.g., ["django-specialist"])
tags: array          # Categorization tags (e.g., ["backend", "api", "performance"])
experimental: boolean # Mark as experimental/beta (default: false)
deprecated: boolean   # Mark as deprecated (default: false)
replacement: string   # If deprecated, name of replacement agent
---
```

## Extended Metadata (Optional)

```yaml
---
metadata:
  author: string              # Agent creator
  created: date              # Creation date (ISO 8601)
  modified: date             # Last modification date
  license: string            # License identifier
  documentation: url         # Link to extended docs
  examples: url             # Link to usage examples
  tests: path               # Path to test files
performance:
  context_usage: string      # "low", "medium", "high"
  response_time: string      # "fast", "moderate", "slow"
  parallel_capable: boolean  # Can work in parallel with others
capabilities:
  domains: array            # Primary domains (e.g., ["django", "rest", "api"])
  integrations: array       # Systems it integrates with
  output_formats: array     # Types of outputs produced
---
```

## Examples

### Minimal Agent
```yaml
---
name: simple-formatter
version: 1.0.0
description: Formats code according to project standards
model: inherit
---
```

### Full-Featured Agent
```yaml
---
name: django-specialist
version: 2.3.1
description: Expert in Django, DRF, and async operations for production systems
model: opus
tools: ["Read", "Write", "Edit", "MultiEdit", "Bash", "Grep"]
dependencies: ["database-architect", "api-designer"]
tags: ["backend", "django", "api", "async"]
metadata:
  author: "Engineering Team"
  created: 2024-01-15
  modified: 2024-03-20
  documentation: https://docs.example.com/agents/django
performance:
  context_usage: high
  response_time: moderate
  parallel_capable: true
capabilities:
  domains: ["django", "drf", "celery", "channels"]
  integrations: ["postgresql", "redis", "rabbitmq"]
  output_formats: ["python", "yaml", "json", "markdown"]
---
```

## Validation Rules

1. **name**: Must be lowercase, alphanumeric with hyphens only
2. **version**: Must follow semantic versioning (MAJOR.MINOR.PATCH)
3. **description**: Maximum 200 characters, must include trigger scenarios
4. **model**: Must be a valid model identifier
5. **tools**: If specified, must contain valid tool names
6. **dependencies**: If specified, must reference existing agents
7. **tags**: If specified, must be from approved tag list

## Migration Guide

For existing agents without proper frontmatter:

1. Extract the agent's purpose from the content
2. Assign appropriate version (start with 1.0.0 for stable agents)
3. Determine model requirements based on complexity
4. List actual tools used in the agent's code examples
5. Identify dependencies from integration mentions
6. Add relevant tags for discoverability