<div align="center">

# Agentic AI

### The Definitive Guide to Designing Autonomous LLM Systems

<br>

[![License](https://img.shields.io/badge/License-CC_BY--SA_4.0-39ff14?style=for-the-badge)](https://creativecommons.org/licenses/by-sa/4.0/)

<br>

*A comprehensive, production-focused guide to multi-agent architectures, orchestration patterns, RAG systems, prompt optimization, and the infrastructure powering autonomous AI.*

**Author: Ignat Penshin**

<br>

---

</div>

## What's Inside

This book covers everything you need to design, build, and operate production-grade agent systems — from architecture patterns to deployment infrastructure.

### Part I: Architecture & Patterns
| Chapter | Topic | Key Technologies |
|---------|-------|-----------------|
| [1. Multi-Agent Architecture](src/ch01-multi-agent-architecture.md) | Orchestration patterns, routing, consensus, scaling | LangGraph, Semantic Router |
| [2. Agent Frameworks](src/ch02-agent-frameworks.md) | Deep comparison of 4 major frameworks | LangGraph, CrewAI, AutoGen, Semantic Kernel |

### Part II: Intelligence & Knowledge
| Chapter | Topic | Key Technologies |
|---------|-------|-----------------|
| [3. Prompt Optimization](src/ch03-prompt-optimization.md) | Programmatic prompt engineering & structured output | DSPy, Guidance, LMQL, Instructor |
| [4. RAG & Vector Databases](src/ch04-rag-and-vector-databases.md) | Production RAG pipelines, hybrid search, evaluation | Qdrant, pgvector, RAGAS |

### Part III: Capabilities & Integration
| Chapter | Topic | Key Technologies |
|---------|-------|-----------------|
| [5. Tool & Function Calling](src/ch05-tool-calling.md) | Function calling, ReAct, tool registries, security | OpenAI API, Instructor |
| [6. State & Orchestration](src/ch06-state-and-orchestration.md) | Durable execution, message buses, event sourcing | Temporal, Redis Streams, Kafka |

### Part IV: Engineering Foundations
| Chapter | Topic | Key Technologies |
|---------|-------|-----------------|
| [7. Python Mastery](src/ch07-python-mastery.md) | Metaclasses, descriptors, async patterns | Python 3.11+, asyncio, Pydantic v2 |
| [8. Advanced SQL](src/ch08-sql-advanced.md) | Window functions, CTEs, query optimization | PostgreSQL, pgvector |

### Part V: Putting It All Together
| Chapter | Topic | Key Technologies |
|---------|-------|-----------------|
| [9. System Design](src/ch09-system-design.md) | Complete system designs with all cross-cutting concerns | Full stack |

---

## How to Use

Every chapter is a standalone Markdown file in the [`src/`](src/) directory — fully readable directly on GitHub.

**Open README → click a chapter → deep-dive into the topic.**

---

## Project Structure

```
awesome-agent-design/
├── src/
│   ├── introduction.md    # Book introduction
│   ├── ch01-*.md          # Chapter 1: Multi-Agent Architecture
│   ├── ch02-*.md          # Chapter 2: Agent Frameworks
│   ├── ch03-*.md          # Chapter 3: Prompt Optimization
│   ├── ch04-*.md          # Chapter 4: RAG & Vector Databases
│   ├── ch05-*.md          # Chapter 5: Tool Calling
│   ├── ch06-*.md          # Chapter 6: State & Orchestration
│   ├── ch07-*.md          # Chapter 7: Python Mastery
│   ├── ch08-*.md          # Chapter 8: Advanced SQL
│   ├── ch09-*.md          # Chapter 9: System Design
│   ├── references.md      # All references & resources
│   └── about.md           # About the author
└── README.md
```

---

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

- **Found an error?** Open an issue
- **Want to add content?** Submit a PR
- **Have a suggestion?** Start a discussion

---

## License

This work is licensed under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).

You are free to share and adapt this material with appropriate attribution.

---

<div align="center">

**Built with passion for the AI engineering community**

</div>
