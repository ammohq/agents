# Agent Validation Checklist

Every agent, regardless of format, MUST satisfy these criteria:

## ✅ Core Requirements

### Metadata
- [ ] Has valid YAML frontmatter block
- [ ] Includes all required fields (name, version, description, model)
- [ ] Description clearly states activation triggers
- [ ] Version follows semantic versioning

### Identity & Purpose
- [ ] Clear, unique role in the system
- [ ] No significant overlap with existing agents
- [ ] Specific domain expertise defined
- [ ] Activation scenarios are unambiguous

### Content Structure
- [ ] Has at least ONE of the following:
  - [ ] Operational instructions
  - [ ] Capability definitions
  - [ ] Input/output contracts
  - [ ] Integration specifications
- [ ] No redundant or boilerplate content
- [ ] Examples are realistic and tested

## ✅ Technical Requirements

### Integration
- [ ] Tools listed match actual usage
- [ ] Dependencies are declared
- [ ] Output formats are specified
- [ ] Error handling is defined

### Quality
- [ ] Code examples are syntactically correct
- [ ] No placeholder content ([bracketed] items filled in)
- [ ] Language is clear and professional
- [ ] No contradictory instructions

### Performance
- [ ] Resource requirements considered
- [ ] Scalability implications noted
- [ ] Parallel execution capability stated

## ✅ Documentation Standards

### Clarity
- [ ] Target audience is clear
- [ ] Technical level is appropriate
- [ ] Jargon is explained or avoided
- [ ] Instructions are actionable

### Completeness
- [ ] Covers common use cases
- [ ] Includes error scenarios
- [ ] Provides success criteria
- [ ] Has concrete examples

## ✅ Anti-Patterns to Avoid

### Content Issues
- ❌ Generic, could-apply-to-anything descriptions
- ❌ Duplicate functionality of existing agents
- ❌ Overly broad scope ("does everything")
- ❌ No clear activation trigger
- ❌ Untested code examples

### Structure Issues
- ❌ Missing frontmatter
- ❌ Inconsistent formatting
- ❌ Walls of text without structure
- ❌ No examples or use cases
- ❌ Mixing multiple unrelated concerns

### Integration Issues
- ❌ Undefined dependencies
- ❌ Circular dependencies
- ❌ Conflicting with other agents
- ❌ No clear handoff points
- ❌ Hidden tool requirements

## 🎯 Quality Scores

Rate each agent on these dimensions (1-5):

| Dimension | Score | Criteria |
|-----------|-------|----------|
| **Specificity** | ⭐⭐⭐⭐⭐ | How well-defined is the agent's purpose? |
| **Completeness** | ⭐⭐⭐⭐⭐ | Does it cover all necessary aspects? |
| **Clarity** | ⭐⭐⭐⭐⭐ | How easy is it to understand? |
| **Integration** | ⭐⭐⭐⭐⭐ | How well does it work with others? |
| **Examples** | ⭐⭐⭐⭐⭐ | Are examples helpful and realistic? |

**Minimum acceptable score: 3/5 per dimension**

## 📋 Review Process

1. **Self-Review**: Author checks against this checklist
2. **Peer Review**: Another team member validates
3. **Integration Test**: Test with related agents
4. **User Test**: Validate with actual use cases
5. **Documentation Review**: Ensure clarity and completeness

## 🚀 Ready for Production?

An agent is production-ready when:
- All core requirements are met ✅
- All technical requirements are satisfied ✅
- Quality score ≥ 4/5 average ✅
- Peer reviewed and approved ✅
- Integration tested ✅

## 🔄 Continuous Improvement

- Review agents quarterly
- Update based on usage patterns
- Deprecate redundant agents
- Merge similar agents when appropriate
- Version bump for significant changes