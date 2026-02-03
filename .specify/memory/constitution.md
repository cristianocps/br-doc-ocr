<!--
Sync Impact Report
==================
Version change: N/A → 1.0.0 (initial ratification)

Added Principles:
- I. Docker-First Environment
- II. Test-First (NON-NEGOTIABLE)
- III. Python 3.12 Standard
- IV. ORM-Based Data Persistence

Added Sections:
- Technology Stack
- Development Workflow
- Governance

Templates Status:
- ✅ plan-template.md: Compatible (Constitution Check section exists)
- ✅ spec-template.md: Compatible (functional requirements and testing sections align)
- ✅ tasks-template.md: Compatible (test-first workflow and foundational phases align)

Follow-up TODOs: None
==================
-->

# BR Doc OCR Constitution

## Core Principles

### I. Docker-First Environment

All development, testing, and CI/CD environments MUST be containerized using Docker.

- Development environment setup MUST be reproducible via Docker Compose or equivalent
- All required tools, databases, and services MUST be defined in container configurations
- Local machine dependencies MUST be limited to Docker and Docker Compose only
- Environment parity MUST be maintained across development, testing, and production

**Rationale**: Eliminates "works on my machine" issues, ensures consistent environments, and simplifies onboarding for new contributors.

### II. Test-First (NON-NEGOTIABLE)

All functional and non-functional requirements MUST have corresponding tests that are developed and passing before code is considered complete.

- TDD workflow is mandatory: write tests → verify tests fail → implement → verify tests pass
- Functional tests MUST cover all user scenarios and acceptance criteria
- Non-functional tests MUST validate performance, security, and reliability requirements
- No feature branch may be merged without all tests passing in CI
- Test coverage reports MUST be generated and reviewed

**Rationale**: Tests are the executable specification. Code without tests is incomplete and introduces risk.

### III. Python 3.12 Standard

Python 3.12 MUST be used as the primary development language version.

- All code MUST be compatible with Python 3.12 features and syntax
- Type hints MUST be used throughout the codebase (PEP 484, PEP 604)
- Modern Python features (match statements, improved error messages, etc.) SHOULD be leveraged
- Dependencies MUST support Python 3.12
- The Docker development container MUST use Python 3.12 as the base image

**Rationale**: Python 3.12 provides improved performance, better error messages, and modern language features essential for maintainable code.

### IV. ORM-Based Data Persistence

All data persistence operations MUST use an Object-Relational Mapping (ORM) framework.

- Direct SQL queries are PROHIBITED except for migrations or performance-critical operations (must be justified)
- Database schema changes MUST be managed through ORM migrations
- Models MUST be defined using ORM patterns with proper relationships and constraints
- Raw database access requires explicit approval and documentation in the Complexity Tracking section
- ORM choice SHOULD be SQLAlchemy 2.0+ for its async support and type safety

**Rationale**: ORM provides abstraction, type safety, and maintainability while reducing SQL injection risks and database coupling.

## Technology Stack

This section defines the mandatory technology choices derived from Core Principles.

| Component | Required Technology | Version/Constraint |
|-----------|--------------------|--------------------|
| Language | Python | 3.12.x |
| Containerization | Docker + Docker Compose | Latest stable |
| ORM | SQLAlchemy | 2.0+ (async preferred) |
| Testing Framework | pytest | Latest stable |
| Type Checking | mypy or pyright | Latest stable |
| Linting | ruff | Latest stable |

Additional technology choices (web framework, database engine, etc.) are feature-specific and documented in respective implementation plans.

## Development Workflow

### Environment Setup

1. Clone repository
2. Run `docker compose up` (or equivalent) to start all services
3. Development environment MUST be fully functional after this single command

### Quality Gates

All code changes MUST pass the following gates before merge:

1. **Unit Tests**: All unit tests pass (`pytest tests/unit/`)
2. **Integration Tests**: All integration tests pass (`pytest tests/integration/`)
3. **Contract Tests**: All contract tests pass (`pytest tests/contract/`)
4. **Type Checking**: No type errors (`mypy` or `pyright`)
5. **Linting**: No linting errors (`ruff check`)
6. **Coverage**: Test coverage meets minimum threshold (configured per project)

### Pre-Commit Checks

- Linting and formatting MUST run as pre-commit hooks
- Type checking SHOULD run as a pre-commit hook
- Fast unit tests MAY run as pre-commit hooks

## Governance

This Constitution is the supreme authority for all development practices in this project.

### Amendment Procedure

1. Propose changes via pull request to this file
2. Changes require review and explicit approval
3. All amendments MUST include rationale and impact assessment
4. Version MUST be updated following semantic versioning:
   - **MAJOR**: Backward-incompatible changes to principles (removal, redefinition)
   - **MINOR**: New principles or sections added, material expansions
   - **PATCH**: Clarifications, wording improvements, non-semantic changes

### Compliance Requirements

- All pull requests MUST verify compliance with Core Principles
- Constitution Check in `plan-template.md` MUST reference current principles
- Violations require explicit justification in Complexity Tracking sections
- Periodic compliance reviews SHOULD be conducted

### Precedence

In case of conflict:
1. This Constitution takes precedence over all other documentation
2. Feature specifications MUST align with Constitutional principles
3. Implementation decisions MUST NOT contradict Constitutional requirements

**Version**: 1.0.0 | **Ratified**: 2026-02-02 | **Last Amended**: 2026-02-02
