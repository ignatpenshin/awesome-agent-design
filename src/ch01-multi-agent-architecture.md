# Chapter 1: Multi-Agent Architecture

> *"The whole is greater than the sum of its parts."* — Aristotle

---

A single LLM call can answer a question. But to **solve a problem** — one that requires research, planning, execution, verification, and adaptation — you need something more: a system of agents working together. Multi-agent architecture is the foundation upon which all production agentic AI systems are built.

This chapter covers the core patterns for orchestrating multiple agents, managing their shared state, routing tasks between them, achieving consensus, scaling to dozens of specialized agents, handling failures gracefully, and choosing the right memory model. By the end, you will have a comprehensive toolkit for designing multi-agent systems of any complexity.

---

## 1.1 Orchestration Patterns

Orchestration is the art of coordinating multiple agents to accomplish a goal that no single agent could achieve alone. There are four foundational orchestration patterns, each suited to different problem structures. Understanding when to apply each pattern — and how to combine them — is the key skill in multi-agent system design.

### Sequential Pattern

The simplest orchestration pattern is the **sequential pipeline**: agents execute one after another, each consuming the output of the previous agent as its input. This mirrors how humans approach complex tasks — research first, then draft, then review.

```
[Agent A] → [Agent B] → [Agent C] → Result
```

The sequential pattern provides clear data flow, easy debugging, and predictable resource usage. Each agent has a well-defined responsibility and a well-defined contract with its neighbors.

**Implementation with LangGraph:**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict


class State(TypedDict):
    input: str
    research: str
    draft: str
    final: str


def researcher(state: State) -> dict:
    """Agent 1: researches the topic and gathers relevant information."""
    return {"research": f"Research on: {state['input']}"}


def writer(state: State) -> dict:
    """Agent 2: writes a draft based on the research findings."""
    return {"draft": f"Draft based on: {state['research']}"}


def editor(state: State) -> dict:
    """Agent 3: edits and polishes the draft into a final version."""
    return {"final": f"Edited: {state['draft']}"}


# Build the sequential graph
graph = StateGraph(State)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("editor", editor)

# Define the sequential flow
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "editor")
graph.add_edge("editor", END)
graph.set_entry_point("researcher")

app = graph.compile()
```

**Key characteristics of the sequential pattern:**

| Property | Value |
|---|---|
| Latency | Sum of all agent latencies (highest) |
| Throughput | Limited by the slowest agent |
| Error propagation | Errors cascade forward |
| Debugging | Straightforward — inspect each stage |
| State management | Linear — each agent adds to shared state |

**When to use:** Document processing pipelines, staged content generation, any workflow where each step genuinely depends on the output of the previous step. Examples include research-then-write flows, extract-transform-load (ETL) pipelines, and multi-stage reasoning chains.

**When to avoid:** When sub-tasks are independent of each other (use parallel instead), or when the number of stages is dynamic (use hierarchical instead).

---

### Parallel Pattern

When multiple aspects of a task are **independent** of one another, running agents in parallel dramatically reduces latency. Instead of waiting for each agent to finish before starting the next, all agents execute simultaneously, and an aggregator combines their results.

```
         ┌──→ [Agent A] ──┐
         │                 │
Input ──→├──→ [Agent B] ──→├──→ Aggregator → Result
         │                 │
         └──→ [Agent C] ──┘
```

**Implementation with asyncio:**

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def analyze_parallel(document: str) -> dict:
    """Run three independent analysis agents in parallel."""
    tasks = [
        analyze_sentiment(document),
        extract_entities(document),
        summarize(document),
    ]
    sentiment, entities, summary = await asyncio.gather(*tasks)
    return {
        "sentiment": sentiment,
        "entities": entities,
        "summary": summary,
    }


async def analyze_sentiment(doc: str) -> str:
    """Analyze the emotional tone of the document."""
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Analyze sentiment: {doc}"}],
    )
    return resp.choices[0].message.content


async def extract_entities(doc: str) -> str:
    """Extract named entities (people, organizations, locations)."""
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Extract entities: {doc}"}],
    )
    return resp.choices[0].message.content


async def summarize(doc: str) -> str:
    """Generate a concise summary of the document."""
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Summarize: {doc}"}],
    )
    return resp.choices[0].message.content
```

The critical insight is the **independence requirement**: parallel agents must not depend on each other's outputs. If Agent B needs Agent A's result, they cannot run in parallel — you need a sequential step first, then parallelism.

**Key characteristics of the parallel pattern:**

| Property | Value |
|---|---|
| Latency | Max of all agent latencies (lowest) |
| Throughput | Sum of all agent throughputs |
| Error handling | Can use `return_exceptions=True` for partial results |
| Resource usage | Highest — all agents run simultaneously |
| Aggregation | Requires explicit merge logic |

**When to use:** Independent analysis tasks (sentiment + entities + summary), multi-method verification, processing acceleration for large datasets, and any scenario where sub-tasks share inputs but not outputs.

---

### Hierarchical Pattern

Real-world problems rarely decompose into a flat list of sequential or parallel steps. The **hierarchical pattern** introduces a Supervisor agent that dynamically decides which subordinate agents to invoke, in what order, and how to combine their results. This is the most flexible orchestration pattern and the one most commonly used in production systems.

```
                  [Supervisor]
                 /      |      \
          [Agent A] [Agent B] [Agent C]
             |         |         |
          [Sub A1]  [Sub B1]  [Sub C1]
```

The Supervisor acts as a manager: it receives the task, decomposes it, delegates sub-tasks to specialist agents, collects their outputs, and decides whether the overall task is complete or requires additional work.

**Implementation with LangGraph:**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import operator
from typing import Annotated


class SupervisorState(TypedDict):
    input: str
    messages: Annotated[list, operator.add]
    next_agent: str
    final_answer: str


def supervisor(state: SupervisorState) -> dict:
    """
    Supervisor agent: analyzes the task and decides which
    specialist agent should handle it next.

    In production, this would use an LLM call with structured output
    to make routing decisions.
    """
    input_text = state["input"].lower()

    # Route based on task analysis
    if "code" in input_text or "implement" in input_text:
        return {
            "next_agent": "coder",
            "messages": [{"role": "supervisor", "content": "Routing to coder agent"}],
        }
    elif "research" in input_text or "find" in input_text:
        return {
            "next_agent": "researcher",
            "messages": [
                {"role": "supervisor", "content": "Routing to researcher agent"}
            ],
        }
    elif "review" in input_text or "check" in input_text:
        return {
            "next_agent": "reviewer",
            "messages": [
                {"role": "supervisor", "content": "Routing to reviewer agent"}
            ],
        }
    else:
        return {
            "next_agent": "FINISH",
            "messages": [
                {
                    "role": "supervisor",
                    "content": "Task does not require specialist routing",
                }
            ],
            "final_answer": "No specialist needed for this task.",
        }


def coder(state: SupervisorState) -> dict:
    """Specialist agent for code generation and implementation tasks."""
    return {
        "messages": [{"role": "coder", "content": f"Code for: {state['input']}"}],
        "final_answer": f"Implementation complete for: {state['input']}",
    }


def researcher(state: SupervisorState) -> dict:
    """Specialist agent for research and information gathering."""
    return {
        "messages": [
            {"role": "researcher", "content": f"Research on: {state['input']}"}
        ],
        "final_answer": f"Research complete for: {state['input']}",
    }


def reviewer(state: SupervisorState) -> dict:
    """Specialist agent for code review and quality checking."""
    return {
        "messages": [
            {"role": "reviewer", "content": f"Review of: {state['input']}"}
        ],
        "final_answer": f"Review complete for: {state['input']}",
    }


def route_to_agent(
    state: SupervisorState,
) -> Literal["coder", "researcher", "reviewer", "FINISH"]:
    """Routing function that reads the supervisor's decision."""
    return state["next_agent"]


# Build the hierarchical graph
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("coder", coder)
graph.add_node("researcher", researcher)
graph.add_node("reviewer", reviewer)

# Supervisor routes to the appropriate agent
graph.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "coder": "coder",
        "researcher": "researcher",
        "reviewer": "reviewer",
        "FINISH": END,
    },
)

# All agents return to END after completing their task
graph.add_edge("coder", END)
graph.add_edge("researcher", END)
graph.add_edge("reviewer", END)

graph.set_entry_point("supervisor")
app = graph.compile()

# Execute
result = app.invoke({"input": "Implement a binary search algorithm", "messages": []})
print(result["final_answer"])
```

The hierarchical pattern naturally extends to **multiple levels**: a top-level supervisor delegates to team leads, each of whom manages a team of specialist agents. This mirrors organizational structures and scales well to complex systems.

**When to use:** Complex tasks requiring dynamic decomposition, systems with specialist agents, any scenario where the workflow cannot be determined in advance.

---

### Consensus-Based Pattern

For high-stakes decisions where accuracy is paramount, the **consensus pattern** runs multiple agents independently on the same task and then reconciles their answers. This is analogous to getting multiple expert opinions before making a critical decision.

```
         ┌──→ [Agent 1] ──┐
         │                 │
Input ──→├──→ [Agent 2] ──→├──→ Consensus → Result
         │                 │
         └──→ [Agent 3] ──┘
```

**Implementation:**

```python
import asyncio
from openai import AsyncOpenAI
from collections import Counter

client = AsyncOpenAI()


async def consensus_answer(question: str, num_agents: int = 3) -> dict:
    """
    Get answers from multiple independent agents and find consensus.

    Each agent uses a different temperature or system prompt to encourage
    diverse reasoning paths. The final answer is determined by majority vote.
    """
    tasks = []
    for i in range(num_agents):
        tasks.append(get_agent_answer(question, agent_id=i))

    answers = await asyncio.gather(*tasks)

    # Find consensus through majority voting
    vote_counts = Counter(answers)
    consensus = vote_counts.most_common(1)[0]

    return {
        "question": question,
        "individual_answers": answers,
        "consensus_answer": consensus[0],
        "agreement_ratio": consensus[1] / len(answers),
        "total_agents": num_agents,
    }


async def get_agent_answer(question: str, agent_id: int) -> str:
    """
    Get an answer from a single agent.

    Different agents use different temperatures to encourage
    diverse reasoning approaches.
    """
    temperatures = [0.0, 0.3, 0.7]
    temp = temperatures[agent_id % len(temperatures)]

    resp = await client.chat.completions.create(
        model="gpt-4o",
        temperature=temp,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer concisely. Think step by step. "
                    "Provide only the final answer."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content


# Usage
result = asyncio.run(consensus_answer("What is the capital of Australia?"))
print(f"Consensus: {result['consensus_answer']}")
print(f"Agreement: {result['agreement_ratio']:.0%}")
```

The consensus pattern is particularly valuable for **reducing hallucinations**: if multiple independent agents agree on an answer, it is far more likely to be correct than any single agent's response.

**When to use:** Critical decisions, fact verification, hallucination reduction, any scenario where the cost of an incorrect answer exceeds the cost of running multiple agents.

---

### Combining Patterns

In practice, production systems combine multiple patterns. A common architecture is:

```
[Supervisor] → routes to →
    [Research Team] (parallel: 3 researchers) →
    [Writing Pipeline] (sequential: drafter → editor → reviewer) →
    [Quality Gate] (consensus: 3 validators)
```

The key is to match the pattern to the nature of each sub-task: parallel for independent work, sequential for dependent stages, hierarchical for dynamic routing, and consensus for critical decisions.

---

## 1.2 Conversation State Management

In a multi-agent system, state is everything. Agents need to know what has happened, what is happening, and what should happen next. Without disciplined state management, multi-agent systems devolve into chaos — agents repeat work, contradict each other, or lose track of the conversation entirely.

### State Machine Approach

The most rigorous approach to state management is a **finite state machine** (FSM). Each conversation exists in exactly one phase at any time, and transitions between phases are explicitly defined and enforced.

```python
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field


class ConversationPhase(Enum):
    """Defines all possible phases of a multi-agent conversation."""

    GREETING = "greeting"
    INFORMATION_GATHERING = "information_gathering"
    PROCESSING = "processing"
    CLARIFICATION = "clarification"
    RESOLUTION = "resolution"
    FEEDBACK = "feedback"
    CLOSED = "closed"


@dataclass
class ConversationState:
    """
    Manages conversation state with explicit phase transitions.

    The transition table defines which phase changes are legal.
    Any attempt to transition outside the allowed set raises an error,
    preventing the conversation from entering an inconsistent state.
    """

    phase: ConversationPhase = ConversationPhase.GREETING
    context: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Transition table: current_phase -> set of allowed next phases
    TRANSITIONS: dict = field(
        default_factory=lambda: {
            ConversationPhase.GREETING: {
                ConversationPhase.INFORMATION_GATHERING,
                ConversationPhase.CLOSED,
            },
            ConversationPhase.INFORMATION_GATHERING: {
                ConversationPhase.PROCESSING,
                ConversationPhase.CLARIFICATION,
                ConversationPhase.CLOSED,
            },
            ConversationPhase.PROCESSING: {
                ConversationPhase.RESOLUTION,
                ConversationPhase.CLARIFICATION,
                ConversationPhase.CLOSED,
            },
            ConversationPhase.CLARIFICATION: {
                ConversationPhase.INFORMATION_GATHERING,
                ConversationPhase.PROCESSING,
                ConversationPhase.CLOSED,
            },
            ConversationPhase.RESOLUTION: {
                ConversationPhase.FEEDBACK,
                ConversationPhase.CLOSED,
            },
            ConversationPhase.FEEDBACK: {
                ConversationPhase.CLOSED,
                ConversationPhase.INFORMATION_GATHERING,
            },
            ConversationPhase.CLOSED: set(),  # Terminal state
        }
    )

    def transition(self, new_phase: ConversationPhase) -> bool:
        """
        Attempt to transition to a new conversation phase.

        Returns True if the transition is valid, raises ValueError otherwise.
        """
        allowed = self.TRANSITIONS.get(self.phase, set())
        if new_phase not in allowed:
            raise ValueError(
                f"Invalid transition: {self.phase.value} -> {new_phase.value}. "
                f"Allowed transitions: {[p.value for p in allowed]}"
            )

        self.history.append(
            {
                "from": self.phase.value,
                "to": new_phase.value,
                "context_snapshot": dict(self.context),
            }
        )
        self.phase = new_phase
        return True

    def is_terminal(self) -> bool:
        """Check if the conversation has reached a terminal state."""
        return self.phase == ConversationPhase.CLOSED
```

The state machine approach provides **strong guarantees**: you can formally reason about which states are reachable, ensure that every conversation eventually terminates, and prevent agents from bypassing required steps (like gathering information before processing).

### Conversation Graph (LangGraph State)

LangGraph provides a more flexible approach to state management through its `StateGraph` abstraction. Instead of explicit transition tables, state evolves through graph nodes, and the framework handles persistence and checkpointing.

```python
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class AgentState(TypedDict):
    """
    Shared state for the multi-agent conversation.

    The Annotated[list, operator.add] type tells LangGraph to
    *append* new messages rather than replace the list. This ensures
    that the full conversation history is preserved across all agent
    interactions.
    """

    messages: Annotated[list, operator.add]
    current_agent: str
    context: dict
    iteration_count: int


def agent_node(state: AgentState) -> dict:
    """A generic agent node that processes messages and updates state."""
    current = state["current_agent"]
    messages = state["messages"]

    # Agent processes the latest message
    latest = messages[-1] if messages else {"content": "No input"}
    response = f"Agent '{current}' processed: {latest.get('content', '')}"

    return {
        "messages": [{"role": current, "content": response}],
        "iteration_count": state["iteration_count"] + 1,
    }


# Build graph with checkpointing for persistence
graph = StateGraph(AgentState)
graph.add_node("processor", agent_node)
graph.add_edge("processor", END)
graph.set_entry_point("processor")

# Enable checkpointing — this persists state across invocations,
# enabling conversation resumption, time-travel debugging,
# and human-in-the-loop workflows
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Invoke with a thread ID for persistent conversations
config = {"configurable": {"thread_id": "conversation-001"}}
result = app.invoke(
    {
        "messages": [{"role": "user", "content": "Analyze this data"}],
        "current_agent": "processor",
        "context": {},
        "iteration_count": 0,
    },
    config=config,
)
```

The `Annotated[list, operator.add]` pattern is fundamental to LangGraph: it defines **reducer functions** that control how state updates are merged. Without this annotation, each agent's output would overwrite the message list. With it, messages accumulate, preserving the full conversation history.

**Checkpointing** enables several critical capabilities:

- **Conversation resumption** — pick up where you left off after a crash or timeout
- **Time-travel debugging** — inspect the state at any point in the conversation
- **Human-in-the-loop** — pause execution, let a human review, then resume
- **Branching** — explore different execution paths from the same checkpoint

---

## 1.3 Routing Mechanisms

Routing is the decision of **which agent should handle a given request**. In a system with many specialist agents, efficient routing is critical — sending a billing question to a technical support agent wastes time and produces poor results. There are two fundamental approaches: LLM-based routing (flexible, expensive) and semantic routing (fast, cheap).

### Router Agent

The most flexible routing approach uses an LLM to analyze the incoming request and select the appropriate agent. By leveraging structured output (via Pydantic models), the router produces machine-readable routing decisions.

```python
from pydantic import BaseModel, Field
from openai import OpenAI
import json


class RouteDecision(BaseModel):
    """Structured routing decision produced by the router agent."""

    agent: str = Field(
        description="Name of the target agent to handle this request"
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was selected"
    )


class RouterAgent:
    """
    LLM-based router that analyzes requests and selects the best agent.

    The router maintains a registry of available agents with their
    descriptions, and uses an LLM to match incoming requests to
    the most appropriate agent.
    """

    def __init__(self, agents: dict[str, str]):
        """
        Args:
            agents: Mapping of agent_name -> agent_description
        """
        self.client = OpenAI()
        self.agents = agents

    def route(self, query: str) -> RouteDecision:
        """Analyze the query and return a routing decision."""
        agent_descriptions = "\n".join(
            f"- {name}: {desc}" for name, desc in self.agents.items()
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a routing agent. Analyze the user's query "
                        "and decide which agent should handle it.\n\n"
                        f"Available agents:\n{agent_descriptions}\n\n"
                        "Respond in JSON with fields: agent, confidence, reasoning"
                    ),
                },
                {"role": "user", "content": query},
            ],
        )

        result = json.loads(response.choices[0].message.content)
        return RouteDecision(**result)


# Usage
router = RouterAgent(
    agents={
        "billing": "Handles payment, invoice, refund, and pricing questions",
        "technical": "Handles bugs, errors, crashes, and technical issues",
        "sales": "Handles product inquiries, demos, and purchasing decisions",
        "general": "Handles general questions and FAQs",
    }
)

decision = router.route("I was charged twice for my subscription")
print(f"Route to: {decision.agent} (confidence: {decision.confidence})")
print(f"Reason: {decision.reasoning}")
```

**Advantages:** Handles ambiguous queries well, can reason about edge cases, no training data required.

**Disadvantages:** Adds latency (one LLM call per routing decision), costs money, may be overkill for simple routing.

### Semantic Router (No LLM Call)

For high-throughput systems where routing latency matters, **semantic routing** avoids LLM calls entirely. Instead, it pre-computes embeddings for representative examples of each route and uses cosine similarity to match incoming queries to the closest route.

```python
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRouter:
    """
    Fast embedding-based router that avoids LLM calls entirely.

    Pre-computes embeddings for example queries in each route category,
    then uses cosine similarity to match new queries to the closest route.
    Typical latency: <10ms vs 500ms+ for LLM-based routing.
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.routes: dict[str, np.ndarray] = {}

    def add_route(self, name: str, examples: list[str]) -> None:
        """
        Register a route with example queries.

        The route embedding is the mean of all example embeddings,
        creating a centroid in embedding space for this category.
        """
        embeddings = self.model.encode(examples)
        # Store the centroid of all example embeddings
        self.routes[name] = np.mean(embeddings, axis=0)

    def route(self, query: str) -> tuple[str, float]:
        """
        Route a query to the best-matching category.

        Returns (route_name, similarity_score).
        """
        query_embedding = self.model.encode([query])[0]

        best_route = None
        best_score = -1

        for name, route_embedding in self.routes.items():
            # Cosine similarity between query and route centroid
            similarity = np.dot(query_embedding, route_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(route_embedding)
            )
            if similarity > best_score:
                best_score = similarity
                best_route = name

        return best_route, float(best_score)


# Usage
router = SemanticRouter()

router.add_route(
    "billing",
    ["payment", "invoice", "refund", "pricing", "charge", "subscription cost"],
)
router.add_route(
    "technical",
    [
        "error",
        "not working",
        "bug",
        "crash",
        "broken feature",
        "server down",
    ],
)
router.add_route(
    "faq",
    [
        "how to",
        "what is",
        "tell me about",
        "information",
        "getting started",
        "documentation",
    ],
)

route, confidence = router.route("My payment didn't go through")
print(f"Route: {route}, Confidence: {confidence:.3f}")
# Output: Route: billing, Confidence: 0.687
```

**Performance comparison:**

| Property | LLM Router | Semantic Router |
|---|---|---|
| Latency | 500-2000ms | 5-10ms |
| Cost per route | $0.001-0.01 | ~$0 (compute only) |
| Accuracy on ambiguous queries | High | Medium |
| Setup effort | Low (prompt only) | Medium (need examples) |
| Handles novel categories | Yes (zero-shot) | No (must pre-register) |

**Best practice:** Use semantic routing as a **first-pass filter** and fall back to LLM-based routing for low-confidence matches. This gives you the speed of semantic routing for 80% of queries and the accuracy of LLM routing for the ambiguous 20%.

---

## 1.4 Consensus and Verification Mechanisms

When the stakes are high — medical advice, financial decisions, legal analysis — a single agent's output is not enough. This section covers three mechanisms for improving reliability through redundancy and verification.

### Majority Voting

The simplest consensus mechanism: run the same query through multiple agents and take the most common answer. This works best for questions with discrete, well-defined answers.

```python
from collections import Counter


async def majority_vote(
    question: str,
    num_voters: int = 5,
    threshold: float = 0.6,
) -> dict:
    """
    Run a question through multiple agents and take a majority vote.

    Args:
        question: The question to answer
        num_voters: Number of independent agents to query
        threshold: Minimum agreement ratio to accept the result

    Returns:
        Dict with the consensus answer, vote distribution, and confidence
    """
    # Get independent answers from multiple agents
    tasks = [get_agent_answer(question, agent_id=i) for i in range(num_voters)]
    answers = await asyncio.gather(*tasks)

    # Count votes
    vote_counts = Counter(answers)
    winner, winner_votes = vote_counts.most_common(1)[0]
    agreement_ratio = winner_votes / num_voters

    return {
        "answer": winner,
        "confidence": agreement_ratio,
        "meets_threshold": agreement_ratio >= threshold,
        "vote_distribution": dict(vote_counts),
        "total_voters": num_voters,
    }
```

**Threshold tuning:** The `threshold` parameter controls the tradeoff between confidence and coverage. A threshold of 0.6 (3/5 agreement) catches most correct answers; a threshold of 0.8 (4/5) is more conservative but may reject correct answers when agents phrase things differently. For critical applications, normalize answers before comparison (lowercase, strip whitespace, extract key terms).

### Debate Protocol

A more sophisticated approach: agents **debate** the question across multiple rounds, refining their positions and responding to each other's arguments. This often produces higher-quality reasoning than any single agent's initial response.

```python
async def agent_debate(
    question: str,
    num_agents: int = 3,
    num_rounds: int = 3,
) -> dict:
    """
    Multi-round debate protocol between agents.

    In each round, every agent sees the arguments from the previous round
    and can update their position. This simulates the kind of deliberation
    that leads to well-reasoned conclusions.
    """
    client = AsyncOpenAI()

    # Initialize: each agent provides their initial position
    positions = []
    for i in range(num_agents):
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are Agent {i + 1} in a panel of experts. "
                        "Provide your well-reasoned answer to the question. "
                        "Think step by step."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        positions.append(
            {"agent": i + 1, "position": resp.choices[0].message.content}
        )

    debate_history = [{"round": 0, "positions": positions}]

    # Debate rounds: agents see and respond to each other's arguments
    for round_num in range(1, num_rounds + 1):
        all_positions = "\n\n".join(
            f"Agent {p['agent']}: {p['position']}" for p in positions
        )

        new_positions = []
        for i in range(num_agents):
            resp = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are Agent {i + 1}. You have seen the arguments "
                            "from all agents. You may update your position based on "
                            "compelling arguments, or defend your original position "
                            "if you believe it is correct. Be specific about what "
                            "convinced you to change (or not)."
                        ),
                    },
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": f"Previous positions:\n{all_positions}",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Round {round_num}: Review all positions and provide "
                            "your updated answer."
                        ),
                    },
                ],
            )
            new_positions.append(
                {"agent": i + 1, "position": resp.choices[0].message.content}
            )

        positions = new_positions
        debate_history.append({"round": round_num, "positions": positions})

    return {
        "question": question,
        "final_positions": positions,
        "debate_history": debate_history,
        "num_rounds": num_rounds,
    }
```

Research has shown that debate protocols can improve accuracy by 10-20% on complex reasoning tasks compared to single-agent approaches, particularly on questions that require weighing multiple factors or considering edge cases.

### Verification Chain

A **verification chain** is a sequential pipeline specifically designed for quality assurance: one agent generates the answer, a second agent fact-checks it, and a third agent assesses the overall quality.

```python
async def verification_chain(question: str) -> dict:
    """
    Three-stage verification pipeline:
    1. Generator — produces the initial answer
    2. Fact-checker — verifies claims and identifies errors
    3. Quality assessor — evaluates overall quality and coherence
    """
    client = AsyncOpenAI()

    # Stage 1: Generate the answer
    gen_resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Provide a detailed, well-sourced answer. "
                    "Clearly state any assumptions you make."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    generated_answer = gen_resp.choices[0].message.content

    # Stage 2: Fact-check the answer
    check_resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous fact-checker. Analyze the following answer "
                    "for factual errors, unsupported claims, logical fallacies, "
                    "and missing nuances. List each issue found."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nAnswer to verify:\n{generated_answer}"
                ),
            },
        ],
    )
    fact_check = check_resp.choices[0].message.content

    # Stage 3: Assess quality and produce final verdict
    quality_resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a quality assessor. Based on the original answer and "
                    "the fact-checker's findings, provide:\n"
                    "1. An overall quality score (1-10)\n"
                    "2. A corrected version of the answer (if needed)\n"
                    "3. A confidence level (low/medium/high)"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Original answer:\n{generated_answer}\n\n"
                    f"Fact-check results:\n{fact_check}"
                ),
            },
        ],
    )
    quality_assessment = quality_resp.choices[0].message.content

    return {
        "question": question,
        "generated_answer": generated_answer,
        "fact_check": fact_check,
        "quality_assessment": quality_assessment,
    }
```

The verification chain is especially effective for **reducing hallucinations** in production systems. The fact-checker agent, operating independently from the generator, is more likely to catch fabricated claims because it approaches the content with a critical mindset rather than a generative one.

---

## 1.5 Scaling Multi-Agent Systems

A system with 3-5 agents can be managed ad hoc. A system with 20+ agents requires deliberate architectural decisions. This section covers the principles and infrastructure needed to scale multi-agent systems.

### Architectural Principles

1. **Domain-based grouping** — Cluster agents into teams based on their domain expertise (billing team, technical team, onboarding team). This reduces the routing complexity and creates natural boundaries for state isolation.

2. **Supervisor hierarchy (2-3 levels)** — Avoid flat architectures where a single supervisor manages all agents. Instead, use a hierarchy: a meta-supervisor coordinates team leads, each team lead manages 3-5 specialist agents. This gives O(sqrt(N)) routing complexity instead of O(N).

3. **Message bus (Redis Streams/Kafka)** — For systems that require asynchronous communication, use a message bus rather than direct agent-to-agent calls. This decouples agents, enables replay, and provides natural observability.

4. **Agent registry for dynamic discovery** — Agents should be able to discover each other's capabilities at runtime, enabling dynamic team formation and graceful degradation when agents are unavailable.

### Agent Registry

The following `AgentRegistry` implements dynamic agent discovery and team management:

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentInfo:
    """Metadata describing a registered agent's capabilities."""

    name: str
    capabilities: list[str]
    description: str
    endpoint: str
    max_concurrent: int = 5
    current_load: int = 0
    status: str = "active"


class AgentRegistry:
    """
    Central registry for dynamic agent discovery.

    Agents register themselves with their capabilities, and the system
    can query the registry to find agents by capability, form teams,
    and monitor load.
    """

    def __init__(self):
        self._agents: dict[str, AgentInfo] = {}
        self._capability_index: dict[str, list[str]] = {}

    def register(self, agent: AgentInfo) -> None:
        """Register a new agent and index its capabilities."""
        self._agents[agent.name] = agent
        for cap in agent.capabilities:
            if cap not in self._capability_index:
                self._capability_index[cap] = []
            self._capability_index[cap].append(agent.name)

    def find_by_capability(self, capability: str) -> list[AgentInfo]:
        """
        Find all agents that have the specified capability.

        Returns only active agents that have available capacity,
        sorted by current load (least loaded first).
        """
        agent_names = self._capability_index.get(capability, [])
        agents = [
            self._agents[name]
            for name in agent_names
            if self._agents[name].status == "active"
            and self._agents[name].current_load < self._agents[name].max_concurrent
        ]
        # Sort by current load — prefer least-loaded agents
        return sorted(agents, key=lambda a: a.current_load)

    def get_team(self, capabilities: list[str]) -> list[AgentInfo]:
        """
        Assemble a team of agents covering all requested capabilities.

        For each capability, selects the least-loaded available agent.
        Returns one agent per capability (no duplicates if an agent
        covers multiple requested capabilities).
        """
        team = {}
        for cap in capabilities:
            available = self.find_by_capability(cap)
            if available:
                # Pick the least-loaded agent not already on the team
                for agent in available:
                    if agent.name not in team:
                        team[agent.name] = agent
                        break
        return list(team.values())

    def deregister(self, agent_name: str) -> None:
        """Remove an agent from the registry."""
        if agent_name in self._agents:
            agent = self._agents[agent_name]
            for cap in agent.capabilities:
                if cap in self._capability_index:
                    self._capability_index[cap] = [
                        n
                        for n in self._capability_index[cap]
                        if n != agent_name
                    ]
            del self._agents[agent_name]

    def update_load(self, agent_name: str, load: int) -> None:
        """Update the current load counter for an agent."""
        if agent_name in self._agents:
            self._agents[agent_name].current_load = load


# Usage
registry = AgentRegistry()

registry.register(
    AgentInfo(
        name="billing-agent-1",
        capabilities=["billing", "refunds", "invoicing"],
        description="Handles billing inquiries and payment processing",
        endpoint="http://billing-1:8080",
    )
)
registry.register(
    AgentInfo(
        name="tech-agent-1",
        capabilities=["debugging", "troubleshooting", "escalation"],
        description="Handles technical support and debugging",
        endpoint="http://tech-1:8080",
    )
)
registry.register(
    AgentInfo(
        name="analytics-agent-1",
        capabilities=["data_analysis", "reporting", "visualization"],
        description="Handles data analysis and report generation",
        endpoint="http://analytics-1:8080",
    )
)

# Find agents capable of billing
billing_agents = registry.find_by_capability("billing")
print(f"Available billing agents: {[a.name for a in billing_agents]}")

# Assemble a team for a complex task
team = registry.get_team(["billing", "data_analysis"])
print(f"Assembled team: {[a.name for a in team]}")
```

### Loop Detection and Prevention

One of the most insidious failure modes in multi-agent systems is the **infinite loop**: Agent A delegates to Agent B, which delegates to Agent C, which delegates back to Agent A, ad infinitum. This burns tokens, wastes time, and can rack up significant costs.

The following `LoopDetector` implements three layers of protection:

```python
import time
from collections import defaultdict


class LoopDetector:
    """
    Three-layer protection against infinite loops in multi-agent systems.

    Layer 1: Absolute iteration limit — hard cap on total iterations.
    Layer 2: Pattern detection — identifies cyclic patterns in call history.
    Layer 3: Timeout — wall-clock time limit for the entire task.
    """

    def __init__(
        self,
        max_iterations: int = 25,
        pattern_window: int = 10,
        timeout_seconds: float = 300.0,
    ):
        self.max_iterations = max_iterations
        self.pattern_window = pattern_window
        self.timeout_seconds = timeout_seconds
        self.call_history: list[str] = []
        self.start_time: float = time.time()
        self.iteration_count: int = 0

    def check(self, agent_name: str) -> dict:
        """
        Record an agent call and check all three loop detection layers.

        Returns a dict with:
            - is_loop: bool — True if a loop was detected
            - reason: str — explanation of which check failed (if any)
        """
        self.call_history.append(agent_name)
        self.iteration_count += 1

        # Layer 1: Absolute iteration limit
        if self.iteration_count > self.max_iterations:
            return {
                "is_loop": True,
                "reason": (
                    f"Maximum iterations exceeded: {self.iteration_count} "
                    f"> {self.max_iterations}"
                ),
            }

        # Layer 2: Pattern detection in recent call history
        if self._detect_cycle():
            return {
                "is_loop": True,
                "reason": (
                    f"Cyclic pattern detected in recent call history: "
                    f"{self.call_history[-self.pattern_window:]}"
                ),
            }

        # Layer 3: Timeout
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            return {
                "is_loop": True,
                "reason": (
                    f"Timeout exceeded: {elapsed:.1f}s > {self.timeout_seconds}s"
                ),
            }

        return {"is_loop": False, "reason": ""}

    def _detect_cycle(self) -> bool:
        """
        Detect repeating patterns in the call history.

        Checks for cycles of length 2, 3, ..., up to half the window size.
        For example, [A, B, A, B] is a cycle of length 2.
        """
        history = self.call_history[-self.pattern_window :]
        if len(history) < 4:
            return False

        # Check for repeating patterns of various lengths
        for cycle_len in range(2, len(history) // 2 + 1):
            pattern = history[-cycle_len:]
            prev_pattern = history[-2 * cycle_len : -cycle_len]
            if pattern == prev_pattern:
                return True

        return False

    def reset(self) -> None:
        """Reset all counters for a new task."""
        self.call_history.clear()
        self.start_time = time.time()
        self.iteration_count = 0


# Usage within a multi-agent loop
detector = LoopDetector(max_iterations=25, timeout_seconds=120.0)

agent_sequence = ["router", "researcher", "writer", "reviewer", "router",
                   "researcher", "writer", "reviewer", "router"]

for agent in agent_sequence:
    result = detector.check(agent)
    if result["is_loop"]:
        print(f"LOOP DETECTED: {result['reason']}")
        break
    print(f"  Executing: {agent} (iteration {detector.iteration_count})")
```

In production, always configure LangGraph's `recursion_limit` parameter as an additional safety net, and set up monitoring alerts for any agent execution that exceeds expected iteration counts.

---

## 1.6 Error Handling and Resilience

Multi-agent systems have many failure points: LLM API timeouts, rate limits, malformed outputs, agent crashes, and cascading failures. The **Circuit Breaker** pattern, borrowed from distributed systems engineering, prevents a failing agent from bringing down the entire system.

### Circuit Breaker Pattern

The circuit breaker operates in three states:

```
CLOSED (normal) ──[failures exceed threshold]──→ OPEN (blocking)
                                                      │
                                                [timeout expires]
                                                      │
                                                      ↓
                                              HALF_OPEN (testing)
                                                  /        \
                                          [success]     [failure]
                                              ↓              ↓
                                           CLOSED          OPEN
```

```python
import time
from enum import Enum
from typing import Any, Callable
from dataclasses import dataclass, field


class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation — requests flow through
    OPEN = "open"            # Circuit tripped — requests are blocked
    HALF_OPEN = "half_open"  # Testing — one request allowed through


@dataclass
class AgentCircuitBreaker:
    """
    Circuit breaker for agent calls.

    Prevents cascading failures by temporarily blocking calls to
    agents that are failing. After a cooldown period, allows a
    single test request through. If it succeeds, the circuit closes
    and normal operation resumes. If it fails, the circuit opens again.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 1

    # Internal state
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    half_open_calls: int = field(default=0)

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function through the circuit breaker.

        In CLOSED state: all calls pass through.
        In OPEN state: calls are blocked until recovery_timeout expires.
        In HALF_OPEN state: limited calls pass through for testing.
        """
        # Check if we should transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. "
                    f"Retry in {self.recovery_timeout - elapsed:.1f}s"
                )

        # In HALF_OPEN state, limit the number of test calls
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    "Circuit breaker is HALF_OPEN — max test calls reached"
                )
            self.half_open_calls += 1

        # Execute the call
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            # Test call succeeded — close the circuit
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
        self.success_count += 1

    def _on_failure(self) -> None:
        """Handle a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Test call failed — reopen the circuit
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            # Too many failures — open the circuit
            self.state = CircuitState.OPEN

    def get_status(self) -> dict:
        """Return the current circuit breaker status for monitoring."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when a call is blocked by an open circuit breaker."""

    pass


# Usage
breaker = AgentCircuitBreaker(failure_threshold=3, recovery_timeout=30.0)

async def call_agent(query: str) -> str:
    """Call an agent through the circuit breaker."""
    return await breaker.call(some_agent_function, query)
```

**Production considerations:**

- Assign a separate circuit breaker to each external dependency (each LLM provider, each tool API)
- Log all state transitions for observability
- Expose circuit breaker status via health check endpoints
- Consider using a fallback agent when the primary circuit is open

---

## 1.7 Memory Models

How agents share (or isolate) information is one of the most important architectural decisions in a multi-agent system. The wrong memory model leads to data races, security leaks, or agents working with stale information.

### Comparison of Memory Models

| Model | Description | When to Use |
|---|---|---|
| **Shared memory** | All agents read/write a shared state store | Tightly coupled agents that need shared context |
| **Isolated memory** | Each agent has its own private memory | Independent agents, security-sensitive domains |
| **Hierarchical** | Shared within team, isolated between teams | Large-scale systems with 20+ agents |
| **Message passing** | Agents communicate exclusively via messages | Loosely coupled, event-driven architectures |

### Shared Memory Implementation

```python
import asyncio
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryEntry:
    """A single entry in shared memory with metadata."""

    value: Any
    author: str
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: list[str] = field(default_factory=list)


class SharedMemory:
    """
    Thread-safe shared memory for multi-agent systems.

    Provides namespaced key-value storage with access tracking,
    tagging, and search capabilities. Uses asyncio locks to prevent
    data races when multiple agents read/write concurrently.
    """

    def __init__(self):
        self._store: dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()

    async def write(
        self,
        key: str,
        value: Any,
        author: str,
        tags: Optional[list[str]] = None,
    ) -> None:
        """Write a value to shared memory."""
        async with self._lock:
            self._store[key] = MemoryEntry(
                value=value,
                author=author,
                tags=tags or [],
            )

    async def read(self, key: str) -> Optional[Any]:
        """Read a value from shared memory. Returns None if not found."""
        async with self._lock:
            entry = self._store.get(key)
            if entry:
                entry.access_count += 1
                return entry.value
            return None

    async def search_by_tag(self, tag: str) -> dict[str, Any]:
        """Find all memory entries with the specified tag."""
        async with self._lock:
            results = {}
            for key, entry in self._store.items():
                if tag in entry.tags:
                    entry.access_count += 1
                    results[key] = entry.value
            return results

    async def search_by_author(self, author: str) -> dict[str, Any]:
        """Find all memory entries written by a specific agent."""
        async with self._lock:
            return {
                key: entry.value
                for key, entry in self._store.items()
                if entry.author == author
            }

    async def get_stats(self) -> dict:
        """Return memory usage statistics for monitoring."""
        async with self._lock:
            return {
                "total_entries": len(self._store),
                "authors": list(
                    set(entry.author for entry in self._store.values())
                ),
                "most_accessed": sorted(
                    self._store.items(),
                    key=lambda x: x[1].access_count,
                    reverse=True,
                )[:5],
            }


# Usage
async def main():
    memory = SharedMemory()

    # Researcher agent stores findings
    await memory.write(
        key="market_analysis",
        value={"trend": "growing", "confidence": 0.85},
        author="researcher",
        tags=["analysis", "market"],
    )

    # Writer agent reads the research
    research = await memory.read("market_analysis")
    print(f"Research data: {research}")

    # Find all analysis-related entries
    analyses = await memory.search_by_tag("analysis")
    print(f"Analysis entries: {analyses}")

asyncio.run(main())
```

### Choosing the Right Memory Model

The decision tree for selecting a memory model:

1. **Do agents need to share state in real-time?** If no, use **message passing** — it is the most decoupled and scalable option.

2. **Is there sensitive data that some agents should not access?** If yes, use **isolated memory** with explicit sharing gates, or **hierarchical memory** where teams have shared state but cross-team access is controlled.

3. **Are there more than 10-15 agents?** If yes, **hierarchical memory** prevents the coordination overhead that plagues flat shared memory at scale.

4. **Is the system tightly coupled with shared context?** If yes, **shared memory** is the simplest and most efficient option, as long as you implement proper locking.

---

## Key Insights

> **Designing 20+ agent systems.** Use hierarchical architecture: group agents into domain teams of 3-5 agents each, with a team supervisor for each group, and one meta-supervisor coordinating between teams. Use an `AgentRegistry` for dynamic discovery and async communication via a message bus (Redis Streams or Kafka). This architecture gives O(sqrt(N)) routing complexity instead of O(N) for a flat supervisor. The team structure also provides natural fault isolation — a failing agent affects only its team, not the entire system.

> **Preventing infinite loops.** Implement three levels of protection: (1) `max_iterations` as an absolute hard limit that can never be exceeded, (2) pattern detection that identifies cyclic patterns like A-B-C-A-B-C in the call history, and (3) a wall-clock timeout for the entire task. Additionally, always configure LangGraph's `recursion_limit`, and set up monitoring and alerts for any execution that exceeds normal iteration counts. In production, prefer failing fast over running indefinitely — it is always cheaper to retry a task than to let it run forever.

> **Task decomposition patterns.** There are five primary patterns for decomposing complex tasks across agents: (1) **Functional** — by work type (research, writing, review), (2) **Data** — by data partition (each agent handles a shard), (3) **Skill-based** — by competence (code agent, math agent, language agent), (4) **Pipeline** — sequential stages where each stage transforms the output, and (5) **MapReduce** — parallel processing of sub-tasks followed by aggregation of results. The best decomposition strategy depends on the nature of the task: use functional for workflows, data for throughput, skill-based for diverse tasks, pipeline for transformations, and MapReduce for embarrassingly parallel problems.

> **Choosing between orchestration patterns.** The four patterns are not mutually exclusive — production systems almost always combine them. A common architecture uses hierarchical routing at the top level, sequential pipelines within each team, parallel execution for independent sub-tasks, and consensus for critical decision points. The key principle: match the pattern to the **dependency structure** of each sub-task, not to the system as a whole.

---

## References

- **"A Survey on Large Language Model based Autonomous Agents"** — Xi et al. (2023). Comprehensive survey of LLM-based agent architectures. [https://arxiv.org/abs/2308.11432](https://arxiv.org/abs/2308.11432)

- **"AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"** — Wu et al. (2023). Microsoft's framework for multi-agent conversations. [https://arxiv.org/abs/2308.08155](https://arxiv.org/abs/2308.08155)

- **"CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society"** — Li et al. (2023). Role-playing framework for agent communication. [https://arxiv.org/abs/2303.17760](https://arxiv.org/abs/2303.17760)

- **LangGraph Documentation** — Official documentation for LangChain's graph-based agent framework. [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)

- **"Multi-Agent Architectures"** — LangChain blog post on multi-agent workflow patterns. [https://blog.langchain.dev/langgraph-multi-agent-workflows/](https://blog.langchain.dev/langgraph-multi-agent-workflows/)
