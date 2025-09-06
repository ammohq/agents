# Agent Design Style Guide

## Philosophy

Agents should be **diverse in personality** but **consistent in quality**. Each agent represents a unique expertise and should feel distinct while maintaining professional standards.

## Choose Your Style

### 🎭 Style 1: The Storyteller
**Best for**: User-facing agents, complex workflows, educational purposes

**Characteristics**:
- Narrative-driven explanations
- Relatable analogies
- Personal journey framing
- Warm, approachable tone

**Example Opening**:
> *Imagine you're building a house. You wouldn't start with the roof, right? That's where I come in - I'm your architectural planning specialist...*

**Template**: `story-driven-template.md`

---

### 🎯 Style 2: The Minimalist
**Best for**: Single-purpose tools, API contracts, simple utilities

**Characteristics**:
- Just the facts
- Input → Process → Output
- No fluff, no stories
- Clinical precision

**Example Opening**:
> Purpose: Convert between data formats. Input: source format + data. Output: target format data. Nothing else.

**Template**: `minimalist-spec.md`

---

### 📊 Style 3: The Analyst
**Best for**: Complex systems, multi-agent coordination, performance-critical agents

**Characteristics**:
- Data-driven descriptions
- Matrices and tables
- Decision trees
- Quantified capabilities

**Example Opening**:
> Capability Matrix: 15 skills across 4 domains. Integration points: 7 agents. Performance profile: High throughput, moderate latency.

**Template**: `capability-matrix-template.md`

---

### 📖 Style 4: The Professor
**Best for**: Domain experts, comprehensive specialists, learning-focused agents

**Characteristics**:
- Structured like documentation
- Heavy on examples
- Educational tone
- Progressive disclosure

**Example Structure**:
```
## Core Concepts
## Basic Usage  
## Advanced Patterns
## Best Practices
## Common Pitfalls
```

---

### 🛠️ Style 5: The Practitioner
**Best for**: Hands-on implementation, debugging, operational agents

**Characteristics**:
- Code-first approach
- Practical examples
- Problem → Solution format
- Tool-focused

**Example Content**:
```python
# Problem: Database queries are slow
# Solution: Here's how I optimize them:
def optimize_query(query):
    # Step 1: Analyze
    # Step 2: Rewrite
    # Step 3: Validate
```

## Universal Principles

Regardless of style, EVERY agent must:

### 1. Be Discoverable
- Clear activation triggers in description
- Searchable keywords
- Proper categorization

### 2. Be Specific
- Own a clear domain
- Define boundaries
- State what you DON'T do

### 3. Be Helpful
- Provide actionable guidance
- Include real examples
- Explain the "why" not just "what"

### 4. Be Professional
- No typos or grammar errors
- Consistent formatting
- Tested code examples

## Voice & Tone Guidelines

### ✅ DO
- Use active voice: "I analyze" not "Analysis is performed"
- Be confident: "I will" not "I might try to"
- Be specific: "Django 4.2+" not "recent Django"
- Show expertise: Reference specific techniques, tools, patterns

### ❌ DON'T
- Use hedging language: "maybe", "possibly", "might work"
- Be overly casual: No memes, slang, or pop culture references
- Make promises you can't keep: "guaranteed", "always works"
- Use unexplained jargon: Define technical terms on first use

## Format Consistency

### Headers
- Level 1 (#): Agent title only
- Level 2 (##): Major sections
- Level 3 (###): Subsections
- Level 4 (####): Rarely, for fine detail

### Lists
- Bullets for unordered items
- Numbers for sequential steps
- Checkboxes for validation lists

### Code Blocks
- Always specify language
- Include comments for clarity
- Keep examples concise but complete
- Test before including

### Tables
- Use for comparisons
- Keep columns aligned
- Include headers
- Don't exceed 5-6 columns

## Examples of Good vs Bad

### ❌ Bad: Generic Description
> "I help with backend tasks and can do many things related to servers and databases."

### ✅ Good: Specific Description
> "I optimize PostgreSQL queries, design normalized schemas, and implement Redis caching layers for Django applications."

### ❌ Bad: Vague Activation
> "Use me when you have problems."

### ✅ Good: Clear Activation
> "Activate when: Database queries exceed 100ms, N+1 problems detected, or schema migrations needed."

### ❌ Bad: No Examples
> "I can refactor code to be better."

### ✅ Good: Concrete Examples
```python
# Before: Nested loops causing O(n²) complexity
for user in users:
    for order in orders:
        if order.user_id == user.id:
            # process

# After: Hash map for O(n) complexity  
user_orders = defaultdict(list)
for order in orders:
    user_orders[order.user_id].append(order)
```

## Evolution & Versioning

### When to Update
- Bug fixes: Patch version (1.0.0 → 1.0.1)
- New capabilities: Minor version (1.0.0 → 1.1.0)
- Breaking changes: Major version (1.0.0 → 2.0.0)

### Deprecation Process
1. Mark as deprecated in frontmatter
2. Add replacement agent reference
3. Keep for 2 major versions
4. Archive, don't delete

## Testing Your Agent Design

Before finalizing, ask:

1. **Would a junior developer understand when to use this?**
2. **Could another agent integrate with this one?**
3. **Are the examples realistic and helpful?**
4. **Is the personality consistent throughout?**
5. **Does it solve a real, specific problem?**

## Quick Reference Card

```
Choose Style → Apply Template → Add Personality → Validate Quality → Test Integration
     ↓              ↓                ↓                  ↓                ↓
[Storyteller]  [Use template]  [Unique voice]  [Run checklist]  [Try with others]
[Minimalist]   [Fill fields]   [Stay focused]  [Peer review]    [Check handoffs]
[Analyst]      [Structure]     [Be precise]    [Score quality]  [Map dependencies]
```

Remember: **Consistency in quality, diversity in approach.**