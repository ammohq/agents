# Claude Code Subagents

A comprehensive collection of specialized AI agents for software development, each expertly crafted for specific technical domains. These agents follow the principle that specialized expertise delivers superior results compared to general-purpose approaches.

## 📁 Repository Structure

```
agents/
├── django/           # Django framework specialists (7 agents)
├── backend/          # Backend infrastructure and services (10 agents)
├── frontend/         # Frontend and UI specialists (5 agents)
├── ai/               # AI and machine learning specialists (2 agents)
├── testing/          # Testing, debugging, and quality (5 agents)
├── devops/           # DevOps and infrastructure (4 agents)
├── utilities/        # General utilities and orchestration (5 agents)
└── templates/        # Agent design templates and guidelines
```

## Quick Start

When working with Claude Code, use the `Task` tool to delegate work to appropriate specialists:

```bash
# Example: Debug a React component issue
Use the Task tool with the frontend-specialist to investigate why the SearchComponent renders twice

# Example: Optimize a slow Django query
Use the Task tool with the django-specialist to fix the N+1 query problem in the UserViewSet

# Example: Set up CI/CD pipeline
Use the Task tool with the devops-engineer to create GitHub Actions workflow for deployment
```

## 🎯 Core Development Specialists

### **orchestrator-agent**
**Master coordinator for complex multi-step tasks**
Analyzes complex projects and delegates to specialized subagents for coordinated problem-solving. Use when you have multi-faceted tasks requiring expertise across different domains. Has visibility into all 35 specialized agents across the entire ecosystem.

### **debugger-detective** 
**Expert bug hunter and performance profiler**
Root cause analysis, performance profiling, memory leak detection, and distributed tracing. Your go-to agent for investigating mysterious bugs and performance bottlenecks.

### **code-reviewer**
**Quality guardian with meticulous standards**
Performs thorough code reviews for quality, maintainability, and best practices. Evaluates SOLID principles, design patterns, and security considerations.

### **test-writer**
**Comprehensive testing specialist following TDD principles**
Generates complete test suites with unit, integration, and E2E tests. Ensures proper coverage and follows testing best practices.

### **refactoring-agent**
**Code improvement specialist**
Improves code structure and design while preserving behavior through systematic, safe refactoring techniques.

## 🏗️ Backend & Data Specialists

### **django-specialist** ⭐
**Supreme Django + DRF + ORM + Pillow expert**
Production-grade Django applications with comprehensive DRF patterns, async capabilities, advanced admin features, and full Pillow integration for image processing. Must be used for all Django API, backend, async, or data-related tasks. Includes enterprise ViewSets, serializers, permissions, and image optimization.

### **django-admin-specialist** 
**Advanced Django Admin customization expert**
Comprehensive Django admin patterns including raw_id_fields, autocomplete_fields, performance optimizations, custom actions, and full Pillow integration for image management. Expert in admin UI/UX, bulk operations, and advanced filtering.

### **django-unfold-admin-specialist** ⭐
**Ultimate Django Unfold admin interface expert**
Master of Django Unfold's modern Tailwind-powered admin with dynamic navigation, conditional fields, custom dashboards, and rich components. Creates beautiful, production-ready admin interfaces with advanced UX patterns.

### **celery-specialist**
**Django + Celery integration master**
Expert in Celery task patterns, Beat scheduling, django-celery-beat, idempotency, retries, error handling, and monitoring. Includes image processing tasks and async workflows.

### **redis-specialist**
**Django + Redis caching and data structures expert**
Advanced Redis patterns with django-redis, pub/sub, distributed locks, rate limiting, session storage, and cache optimization strategies. Integrates seamlessly with Django applications.

### **file-storage-specialist**
**Cloud storage and Pillow processing expert**
S3/Azure/GCS backends, CDN integration, image processing pipelines with Pillow, WebP conversion, thumbnail generation, and optimized storage strategies. Production-ready file management.

### **monitoring-specialist**
**Application monitoring and observability expert**
Sentry integration, Prometheus metrics, structured logging, health checks, performance monitoring, and alerting. Comprehensive production monitoring setup.

### **full-stack-coder**
**Complete application builder**
Implements features across the entire stack - frontend, backend, database, and infrastructure. Ideal for end-to-end feature development.

### **database-architect**
**Database design and optimization master**
Expert in PostgreSQL, MySQL, query optimization, indexing strategies, migrations, replication, and sharding. Handles complex database challenges.

### **data-engineer**
**Big data and ETL pipeline expert**
Apache Airflow, Spark, data warehousing, Pandas optimization, and big data processing. Builds robust data infrastructure and pipelines.

### **api-designer**
**REST, GraphQL, and gRPC specialist**
Comprehensive API design with OpenAPI documentation, versioning, authentication, and gateway configuration. Focuses on Django REST Framework.

### **migration-specialist**
**Legacy modernization expert**
Legacy code modernization, framework upgrades, database migrations, and zero-downtime deployments. Handles complex system transitions.

## 🎨 Frontend & UX Specialists

### **frontend-specialist**
**Modern JavaScript frameworks expert**
React, Vue, Svelte, state management, modern CSS, build tools, and component libraries. Builds sophisticated user interfaces with performance optimization.

### **htmx-boss**
**Hypermedia-driven architecture master**
HTMX, Alpine.js integration, Django + HTMX patterns, progressive enhancement, and SPA-like experiences without heavy JavaScript frameworks.

### **mobile-developer**
**Cross-platform mobile specialist**
React Native, Flutter, PWAs, mobile optimization, app store deployment, and native integrations. Creates polished mobile experiences.

### **ux-specialist** ⭐
**User experience testing expert with AI-powered automation**
UX/UI testing, accessibility compliance, user interaction flows, visual regression testing using Playwright, Puppeteer, and **Stagehand** for AI-powered natural language automation. Features intelligent bug reproduction, automated issue diagnosis, and context-aware testing. Ensures excellent user experiences through both traditional and AI-enhanced testing methods.

### **accessibility-champion**
**Inclusive design specialist**
WCAG compliance, ARIA, screen readers, keyboard navigation, and inclusive design. Makes applications usable by everyone.

## ⚡ Specialized Technology Experts

### **websocket-specialist**
**Real-time communication expert**
WebSocket implementations across all stacks - Django Channels, Socket.io, native WebSocket API with frontend clients, reconnection strategies, and real-time patterns.

### **search-specialist**
**Search and discovery expert**
Elasticsearch, Algolia, MeiliSearch, full-text search, faceted search, and search relevance tuning. Implements sophisticated search capabilities.

### **n8n-automation-specialist** ⭐
**n8n workflow automation architect**
Production-grade n8n workflow design with comprehensive validation, error handling, and optimization. Builds complex automation workflows with pre and post validation, incremental updates, and security best practices.

### **auth-security-specialist**
**Authentication and security expert**
OAuth2, JWT, OIDC, MFA, session management, CORS, security headers, and authentication best practices. Implements robust security systems.

### **email-specialist**
**Transactional email expert**
Email templates, deliverability, SendGrid/SES integration, and email marketing. Handles all aspects of application email communication.

### **payment-specialist**
**Payment processing expert**
Stripe, PayPal, payment processing, subscriptions, PCI compliance, and webhook handling. Implements secure payment systems.

## 🤖 AI & Machine Learning Specialists

### **chatgpt-specialist** ⭐
**OpenAI and ChatGPT integration master**
Complete OpenAI Python SDK integration (sync/async), Django + Celery patterns, streaming responses with Django Channels, function calling framework, embeddings and vector operations, structured outputs with Pydantic, and production-grade error handling. Expert in GPT-4, GPT-3.5, cost optimization, token management, fine-tuning, and webhook handling. Implements scalable AI features with comprehensive monitoring and security best practices.

### **stockfish-specialist** ⭐
**Stockfish chess engine integration expert**
Complete Stockfish UCI protocol integration, NNUE neural network evaluation, REST API development (FastAPI/Django), python-chess library mastery, Docker/Kubernetes deployment, Syzygy tablebase configuration, analysis caching strategies, and performance optimization. Expert in position analysis, best move calculation, batch processing, real-time streaming, and production chess analysis systems with horizontal scaling.

## 🛠️ Infrastructure & Operations

### **devops-engineer**
**Infrastructure and deployment expert**
Docker, Kubernetes, CI/CD, AWS/GCP/Azure, Terraform, GitHub Actions, monitoring, and production deployment strategies. Handles all DevOps needs.

### **coolify-specialist** ⭐
**Coolify self-hosting platform expert with MCP integration**
Manages Coolify PaaS deployments via coolify-mcp-server. Application deployments, database provisioning (PostgreSQL, MySQL, MongoDB, Redis, etc.), server management, environment configuration, service orchestration, zero-downtime deployments, and resource monitoring. Complete control over self-hosted infrastructure through Coolify API integration.

### **security-auditor**
**Defensive security specialist**
Identifies vulnerabilities and enforces defensive security practices. Focuses only on protective security measures, never offensive techniques.

### **performance-analyzer**
**Optimization and efficiency expert**
Identifies bottlenecks and optimizes code for maximum performance and efficiency. Improves application speed and resource usage.

## 📚 Documentation & Quality

### **documentation-writer**
**Technical writing specialist**
API documentation, README files, architecture docs, user guides, and technical writing. Creates clear, comprehensive documentation.

### **prd-writer**
**Product Requirement Document specialist**
Expert in authoring comprehensive Product Requirement Documents with modern product management frameworks. Creates clear, actionable PRDs that align teams and drive successful product outcomes with structured templates for goals, requirements, user stories, and acceptance criteria.

## 🎯 Agent Selection Guide

### For Immediate Issues:
- **Bug reports**: `debugger-detective`
- **Performance problems**: `performance-analyzer`  
- **UI/UX issues**: `ux-specialist` (with Stagehand AI) or `frontend-specialist`
- **Django/API problems**: `django-specialist`
- **Admin interface issues**: `django-admin-specialist` or `django-unfold-admin-specialist`
- **Image processing problems**: `file-storage-specialist` (Pillow integration)
- **Async task issues**: `celery-specialist`
- **Caching problems**: `redis-specialist`
- **Security concerns**: `security-auditor`
- **AI/ChatGPT integration**: `chatgpt-specialist`
- **Chess engine integration**: `stockfish-specialist`

### For Development Tasks:
- **New features**: `full-stack-coder` or domain-specific specialist
- **Frontend components**: `frontend-specialist`
- **Database design**: `database-architect`
- **API endpoints**: `django-specialist` or `api-designer`
- **Admin interfaces**: `django-admin-specialist` or `django-unfold-admin-specialist`
- **Image handling**: `file-storage-specialist` with Pillow processing
- **Background tasks**: `celery-specialist`
- **Caching strategies**: `redis-specialist`
- **File uploads**: `file-storage-specialist`
- **Mobile apps**: `mobile-developer`
- **AI features**: `chatgpt-specialist` for OpenAI/ChatGPT integration
- **Chess applications**: `stockfish-specialist` for engine integration
- **Natural language testing**: `ux-specialist` with Stagehand

### For Complex Projects:
- **Multi-domain tasks**: `orchestrator-agent`
- **System migrations**: `migration-specialist`
- **Infrastructure setup**: `devops-engineer`
- **Self-hosted deployments**: `coolify-specialist`
- **Data pipelines**: `data-engineer`

### For Quality & Maintenance:
- **Code reviews**: `code-reviewer`
- **Test creation**: `test-writer`
- **Refactoring**: `refactoring-agent`
- **Documentation**: `documentation-writer`

## ⭐ Django Ecosystem Powerhouse

**Complete Django/DRF/Celery/Redis/Postgres Stack Coverage:**

- **`django-specialist`** - Core Django + DRF + ORM + Pillow patterns
- **`django-admin-specialist`** - Advanced admin customization + image management  
- **`django-unfold-admin-specialist`** - Modern Tailwind admin interfaces
- **`celery-specialist`** - Async task processing + image workflows
- **`redis-specialist`** - Caching + session management + rate limiting
- **`file-storage-specialist`** - Cloud storage + Pillow integration + CDN
- **`monitoring-specialist`** - Sentry + Prometheus + structured logging

**Key Features:**
- 🖼️ **Full Pillow Integration** - Image processing, optimization, WebP conversion
- 🎨 **Modern Admin UX** - Django Unfold with Tailwind components
- ⚡ **Enterprise DRF Patterns** - ViewSets, serializers, permissions, filtering
- 🚀 **Production Ready** - Async processing, caching, monitoring, security

## 🚀 Best Practices

1. **Always delegate to specialists** - Each agent has deep expertise in their domain
2. **Provide context** - Give agents relevant background about your project
3. **Use orchestrator-agent** for multi-step tasks requiring coordination
4. **Start with debugging** - Use `debugger-detective` to investigate issues before fixing
5. **Follow the chain** - Let specialists recommend other agents for follow-up work
6. **Django stack coordination** - Use multiple Django specialists together for complex projects

## 🔧 Capabilities

Each agent includes:
- **Specialized Knowledge**: Deep expertise in their technical domain
- **Production-Ready Code**: Complete, working implementations
- **Best Practices**: Industry-standard patterns and approaches
- **Testing**: Comprehensive test coverage strategies
- **Documentation**: Clear explanations and examples
- **Modern Standards**: Up-to-date with latest technologies and practices

## 🎨 Agent Design System

The repository includes comprehensive templates and guidelines for creating new agents:

### Available Templates
- **story-driven-template.md** - Narrative approach for user-facing agents
- **minimalist-spec.md** - Just input/output contracts for simple utilities
- **capability-matrix-template.md** - Data-driven with performance metrics

### Standards & Guidelines
- **yaml-frontmatter-standard.md** - Consistent metadata structure
- **agent-validation-checklist.md** - Quality gates for all agents
- **agent-design-style-guide.md** - Voice, tone, and formatting guidelines

### Creating New Agents
1. Choose an appropriate template based on complexity and purpose
2. Follow the YAML frontmatter standard for metadata
3. Validate against the quality checklist
4. Place in the appropriate category directory

---

## 🌟 Latest Additions

### **Stockfish Specialist** (New!)
Chess engine integration and analysis systems:
- Complete UCI protocol communication with python-chess
- NNUE neural network evaluation (10-50x faster than traditional)
- REST API servers (FastAPI/Django) for position analysis
- Docker/Kubernetes deployment with horizontal scaling
- Syzygy tablebase configuration for perfect endgame play
- Analysis result caching with Redis for performance
- Batch processing and real-time WebSocket streaming
- Production optimization (threads, hash tables, search depth)

### **Coolify Specialist** (New!)
Self-hosted PaaS management through coolify-mcp-server:
- Application deployment and lifecycle management
- Database provisioning (PostgreSQL, MySQL, MongoDB, Redis, etc.)
- Server resource monitoring and validation
- Environment variable and secret management
- Zero-downtime deployment strategies
- Domain and SSL/TLS configuration
- Project and team organization

### **ChatGPT Specialist**
Complete OpenAI integration expert covering:
- Async/sync Python SDK patterns
- Django + Celery background processing
- Real-time streaming with Django Channels
- Function calling and tool use frameworks
- Embeddings and vector search
- Cost optimization and monitoring
- Production security best practices

### **Stagehand Integration** (Enhanced UX Specialist!)
The `ux-specialist` now includes Stagehand for AI-powered browser automation:
- Natural language test actions ("click on the login button")
- Intelligent bug reproduction and diagnosis
- AI-powered visual regression detection
- Context-aware form filling and validation
- Hybrid testing combining AI with traditional automation

---

**Remember**: These agents are designed to work together. Complex projects often require multiple specialists coordinated by the orchestrator-agent for optimal results. The orchestrator has full visibility into all 38 agents and can effectively coordinate multi-domain tasks.