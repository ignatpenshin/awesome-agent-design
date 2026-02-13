# Chapter 9: System Design for Agent Systems

> *"Architecture is the decisions you wish you could get right early."* --- Ralph Johnson

---

Designing a production multi-agent system is a multidisciplinary challenge. It requires reasoning about concurrency, fault tolerance, data pipelines, cost economics, security, and human oversight --- all within the constraints of LLM latency and token budgets. This chapter presents a structured framework for agent system design, then applies it to three complete case studies: an intelligent customer support platform, a document analysis pipeline, and an automated application processing system.

---

## System Design Framework

Every agent system design conversation should follow a structured framework. This ensures you address all critical dimensions systematically, even under time pressure.

### The 6-Step Framework

| Step | Focus | Time | Key Questions |
|------|-------|------|---------------|
| **1. Requirements** | Scope & constraints | 2 min | Users? Scale? Latency SLA? Accuracy target? Compliance? |
| **2. Agent Design** | Individual agents | 3 min | Which agents? What model per agent? Tools needed? Input/output schema? |
| **3. Orchestration** | Agent coordination | 5 min | LangGraph/CrewAI? Sequential/parallel/hierarchical? State schema? Error handling? |
| **4. Data & RAG** | Knowledge & memory | 3 min | What data sources? Chunking strategy? Vector DB? Hybrid search? Memory architecture? |
| **5. Infrastructure** | Deployment & scaling | 3 min | Redis/Kafka/Temporal? Caching? Auto-scaling? Queue strategy? |
| **6. Monitoring & Safety** | Observability & guardrails | 2 min | PII protection? Prompt injection defense? Cost budgets? HITL triggers? Evaluation metrics? |

**Principles:**

1. **Start with requirements.** Every architectural decision flows from constraints.
2. **Design agents before orchestration.** You cannot coordinate agents whose responsibilities are undefined.
3. **State is the hardest problem.** Define the state schema early --- it determines what agents can share and what must be persisted.
4. **Plan for failure.** LLMs are non-deterministic. Every agent call can fail, hallucinate, or exceed its budget.
5. **Human-in-the-loop is a feature, not a fallback.** Design escalation paths from the start.

---

## Design 1: Intelligent Customer Support Platform

### Requirements

**Functional:**
- Multi-channel support: chat, email, voice (transcribed)
- Automatic classification and routing of inquiries
- FAQ answering from knowledge base (RAG)
- Transaction execution via service APIs (order status, account updates, refunds)
- Complex complaint handling with reasoning
- Seamless human escalation when confidence is low

**Non-functional:**
- 10,000+ concurrent conversations
- < 3 second response time (p95)
- 70%+ auto-resolution rate
- PII protection and full audit trail
- Multi-language support
- 99.9% uptime

### Architecture

```
                         ┌──────────────────┐
                         │   Load Balancer   │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │   API Gateway     │
                         │  (Rate Limiting)  │
                         └────────┬─────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │       Router Agent          │
                    │  (Classification + Routing) │
                    └─────┬──────┬──────┬────────┘
                          │      │      │
              ┌───────────▼┐  ┌──▼──────▼──┐  ┌─────────────┐
              │  FAQ Agent  │  │ Transaction │  │  Complaint   │
              │   (RAG)     │  │   Agent     │  │   Agent      │
              └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
                     │                │                 │
              ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
              │  Knowledge  │  │  Service    │  │  Reasoning  │
              │  Base (VDB) │  │  APIs       │  │  Model (o1) │
              └─────────────┘  └─────────────┘  └─────────────┘
                                      │
                               ┌──────▼──────┐
                               │  Human      │
                               │  Escalation │
                               └─────────────┘
```

Supporting infrastructure:
- **Redis**: Session state, semantic cache, rate limiting
- **Kafka**: Event streaming, audit trail, analytics pipeline
- **PostgreSQL**: Conversation history, user profiles, knowledge base metadata
- **Vector DB (Qdrant/Pinecone)**: FAQ embeddings, documentation search

### Agent Design

| Agent | Model | Tools | Purpose |
|-------|-------|-------|---------|
| **Router** | GPT-4o-mini | classification prompt | Classify intent, route to specialist |
| **FAQ Agent** | GPT-4o-mini + RAG | vector_search, knowledge_base | Answer questions from documentation |
| **Transaction Agent** | GPT-4o | order_status, account_update, refund_process | Execute service operations via API |
| **Complaint Agent** | o1-mini | search, escalate, create_ticket | Reason through complex complaints |
| **Escalation Agent** | --- | route_to_human, create_summary | Prepare context for human agents |

### Orchestration

The support platform uses a LangGraph state machine to coordinate agents:

```python
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    conversation_id: str
    customer_id: str
    intent: str | None                 # classified intent
    confidence: float                  # router confidence
    agent_type: str | None             # current handling agent
    tool_results: list[dict]           # results from tool calls
    escalation_reason: str | None      # why escalated
    resolution_status: str             # open, resolved, escalated
    turn_count: int                    # number of agent turns
    total_tokens: int                  # token budget tracking
    pii_detected: bool                 # PII flag


async def router_node(state: SupportState) -> dict:
    """Classify intent and determine routing."""
    last_message = state["messages"][-1].content

    # PII check before LLM call
    cleaned_message, pii_found = pii_protector.scan(last_message)

    classification = await classify_intent(cleaned_message)

    return {
        "intent": classification["intent"],
        "confidence": classification["confidence"],
        "pii_detected": pii_found,
    }


def route_decision(state: SupportState) -> str:
    """Determine next node based on classification."""
    if state["confidence"] < 0.6:
        return "escalation"
    if state["turn_count"] > 10:
        return "escalation"

    routing = {
        "faq": "faq_agent",
        "transaction": "transaction_agent",
        "complaint": "complaint_agent",
        "general": "faq_agent",
    }
    return routing.get(state["intent"], "escalation")


async def faq_agent_node(state: SupportState) -> dict:
    """Answer questions using RAG over knowledge base."""
    query = state["messages"][-1].content
    context = await vector_search(query, top_k=5)
    response = await generate_faq_response(query, context)

    return {
        "messages": [AIMessage(content=response["answer"])],
        "resolution_status": "resolved" if response["confidence"] > 0.8 else "open",
        "total_tokens": state["total_tokens"] + response["tokens_used"],
    }


async def transaction_agent_node(state: SupportState) -> dict:
    """Execute transactions via service APIs."""
    query = state["messages"][-1].content
    plan = await plan_transaction(query, state["customer_id"])

    if plan["requires_confirmation"]:
        return {
            "messages": [AIMessage(content=plan["confirmation_prompt"])],
            "tool_results": [plan],
        }

    result = await execute_transaction(plan)
    return {
        "messages": [AIMessage(content=result["response"])],
        "tool_results": state["tool_results"] + [result],
        "resolution_status": "resolved",
        "total_tokens": state["total_tokens"] + result["tokens_used"],
    }


async def complaint_agent_node(state: SupportState) -> dict:
    """Handle complex complaints with reasoning model."""
    response = await reason_about_complaint(
        messages=state["messages"],
        customer_id=state["customer_id"],
    )

    if response["needs_escalation"]:
        return {
            "escalation_reason": response["reason"],
            "resolution_status": "escalated",
        }

    return {
        "messages": [AIMessage(content=response["answer"])],
        "resolution_status": "resolved",
        "total_tokens": state["total_tokens"] + response["tokens_used"],
    }


async def escalation_node(state: SupportState) -> dict:
    """Prepare context and escalate to human agent."""
    summary = await generate_escalation_summary(state["messages"])
    await route_to_human_queue(
        conversation_id=state["conversation_id"],
        summary=summary,
        reason=state.get("escalation_reason", "low_confidence"),
    )
    return {
        "messages": [AIMessage(
            content="I'm connecting you with a specialist who can help further. "
                    "They'll have full context of our conversation."
        )],
        "resolution_status": "escalated",
    }


def should_continue(state: SupportState) -> str:
    """Check if conversation should continue or end."""
    if state["resolution_status"] in ("resolved", "escalated"):
        return "end"
    if state["total_tokens"] > 50_000:
        return "escalation"
    return "router"


# --- Graph Assembly ---
graph = StateGraph(SupportState)

graph.add_node("router", router_node)
graph.add_node("faq_agent", faq_agent_node)
graph.add_node("transaction_agent", transaction_agent_node)
graph.add_node("complaint_agent", complaint_agent_node)
graph.add_node("escalation", escalation_node)

graph.set_entry_point("router")

graph.add_conditional_edges("router", route_decision, {
    "faq_agent": "faq_agent",
    "transaction_agent": "transaction_agent",
    "complaint_agent": "complaint_agent",
    "escalation": "escalation",
})

graph.add_conditional_edges("faq_agent", should_continue, {
    "end": END,
    "router": "router",
    "escalation": "escalation",
})
graph.add_conditional_edges("transaction_agent", should_continue, {
    "end": END,
    "router": "router",
    "escalation": "escalation",
})
graph.add_conditional_edges("complaint_agent", should_continue, {
    "end": END,
    "router": "router",
    "escalation": "escalation",
})
graph.add_edge("escalation", END)

support_app = graph.compile()
```

### State Management & Scaling

**Session state** is stored in Redis with a TTL matching the conversation timeout:

```python
import redis.asyncio as redis
import json

class SessionManager:
    def __init__(self, redis_url: str, ttl_seconds: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl_seconds

    async def save_state(self, conversation_id: str, state: dict):
        key = f"support:session:{conversation_id}"
        await self.redis.setex(key, self.ttl, json.dumps(state, default=str))

    async def load_state(self, conversation_id: str) -> dict | None:
        key = f"support:session:{conversation_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def extend_ttl(self, conversation_id: str):
        key = f"support:session:{conversation_id}"
        await self.redis.expire(key, self.ttl)
```

**Scaling strategy:**
- Horizontal scaling of stateless agent workers behind the load balancer.
- Redis Cluster for session state (no sticky sessions needed).
- Kafka partitioned by `conversation_id` for ordered event processing.
- Auto-scaling based on queue depth and response latency metrics.

---

## Design 2: Document Analysis Pipeline

Large documents --- contracts, research papers, compliance reports --- exceed LLM context windows and require a structured pipeline approach. The Map-Reduce pattern processes documents in parallel chunks and synthesizes a unified analysis.

### Architecture

```
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │  Document     │     │   Chunking   │     │    Map       │
   │  Ingestion    │────▶│   Strategy   │────▶│  (Parallel   │
   │  (PDF/DOCX)  │     │  (Semantic)  │     │   Analysis)  │
   └──────────────┘     └──────────────┘     └──────┬───────┘
                                                     │
                                              ┌──────▼───────┐
                                              │    Reduce     │
                                              │  (Synthesis)  │
                                              └──────┬───────┘
                                                     │
                                              ┌──────▼───────┐
                                              │   Output      │
                                              │  (Structured) │
                                              └──────────────┘
```

### Implementation

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    start_page: int
    end_page: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkAnalysis:
    chunk_id: int
    summary: str
    key_findings: list[str]
    entities: list[dict]
    risk_flags: list[str]
    confidence: float


class LongDocumentProcessor:
    """Map-Reduce processor for documents exceeding context window."""

    def __init__(
        self,
        map_model: str = "gpt-4o-mini",
        reduce_model: str = "gpt-4o",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        max_concurrent: int = 10,
    ):
        self.map_model = map_model
        self.reduce_model = reduce_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_concurrent = max_concurrent

    def chunk_document(self, text: str, pages: list[dict]) -> list[DocumentChunk]:
        """Split document into overlapping semantic chunks."""
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for para in paragraphs:
            if current_length + len(para) > self.chunk_size and current_chunk:
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text='\n\n'.join(current_chunk),
                    start_page=0,  # Simplified; map from char offset to page
                    end_page=0,
                    metadata={"paragraph_count": len(current_chunk)},
                ))
                chunk_id += 1
                # Keep overlap
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)

            current_chunk.append(para)
            current_length += len(para)

        if current_chunk:
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text='\n\n'.join(current_chunk),
                start_page=0,
                end_page=0,
            ))

        return chunks

    async def map_phase(self, chunks: list[DocumentChunk]) -> list[ChunkAnalysis]:
        """Analyze each chunk in parallel (Map step)."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def analyze_chunk(chunk: DocumentChunk) -> ChunkAnalysis:
            async with semaphore:
                prompt = f"""Analyze the following document section.
Extract: (1) summary, (2) key findings, (3) named entities, (4) risk flags.

SECTION (chunk {chunk.chunk_id}):
{chunk.text}

Respond in JSON format."""

                response = await call_llm(
                    model=self.map_model,
                    prompt=prompt,
                    response_format="json",
                )

                return ChunkAnalysis(
                    chunk_id=chunk.chunk_id,
                    summary=response["summary"],
                    key_findings=response["key_findings"],
                    entities=response["entities"],
                    risk_flags=response.get("risk_flags", []),
                    confidence=response.get("confidence", 0.8),
                )

        tasks = [analyze_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if isinstance(r, ChunkAnalysis)]

    async def reduce_phase(self, analyses: list[ChunkAnalysis]) -> dict:
        """Synthesize chunk analyses into a unified report (Reduce step)."""
        combined_summaries = "\n\n".join(
            f"[Section {a.chunk_id}] {a.summary}" for a in analyses
        )
        all_findings = []
        all_entities = []
        all_risks = []

        for a in analyses:
            all_findings.extend(a.key_findings)
            all_entities.extend(a.entities)
            all_risks.extend(a.risk_flags)

        prompt = f"""You are synthesizing a multi-section document analysis.

SECTION SUMMARIES:
{combined_summaries}

ALL KEY FINDINGS ({len(all_findings)} total):
{chr(10).join(f'- {f}' for f in all_findings)}

RISK FLAGS ({len(all_risks)} total):
{chr(10).join(f'- {r}' for r in all_risks)}

Produce a unified analysis with:
1. Executive summary (3-5 sentences)
2. Top findings (deduplicated, ranked by importance)
3. Risk assessment (critical/high/medium/low)
4. Recommendations
5. Entities (deduplicated)

Respond in JSON format."""

        response = await call_llm(
            model=self.reduce_model,
            prompt=prompt,
            response_format="json",
        )

        return response

    async def process(self, document_text: str) -> dict:
        """Full pipeline: chunk → map → reduce."""
        chunks = self.chunk_document(document_text, [])
        print(f"Document split into {len(chunks)} chunks")

        analyses = await self.map_phase(chunks)
        print(f"Analyzed {len(analyses)}/{len(chunks)} chunks successfully")

        report = await self.reduce_phase(analyses)
        report["metadata"] = {
            "total_chunks": len(chunks),
            "successful_analyses": len(analyses),
            "models_used": {
                "map": self.map_model,
                "reduce": self.reduce_model,
            },
        }

        return report


# --- Usage ---
async def analyze_document(file_path: str) -> dict:
    processor = LongDocumentProcessor(
        map_model="gpt-4o-mini",       # Fast, cheap for individual chunks
        reduce_model="gpt-4o",         # Powerful for synthesis
        chunk_size=4000,
        max_concurrent=10,
    )

    text = load_document(file_path)     # PDF/DOCX extraction
    return await processor.process(text)
```

---

## Design 3: Automated Application Processing

This design handles multi-step application workflows --- loan applications, insurance claims, permit requests --- where each stage requires different validation, some decisions require human approval, and the entire process must be auditable and durable.

### Requirements

- Process 10,000+ applications per day
- Multi-stage pipeline: intake, data extraction, validation, risk assessment, decision, notification
- Human-in-the-loop for high-risk or edge-case decisions
- Durable execution: survive crashes, restarts, and deployments
- Full audit trail for compliance
- SLA: 95% of routine applications processed within 1 hour

### Architecture with Temporal

Temporal provides durable execution --- workflows survive process crashes and can wait for human input for hours or days without consuming resources.

```python
from datetime import timedelta
from temporalio import workflow, activity
from temporalio.common import RetryPolicy
from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Application:
    application_id: str
    applicant_name: str
    application_type: str
    submitted_data: dict
    documents: list[str]


@dataclass
class AssessmentResult:
    risk_level: RiskLevel
    risk_score: float
    factors: list[str]
    recommended_action: str
    requires_human_review: bool


@dataclass
class HumanDecision:
    reviewer_id: str
    approved: bool
    comments: str
    conditions: list[str]


# --- Activities (individual steps) ---

@activity.defn
async def extract_application_data(application: Application) -> dict:
    """Use LLM to extract structured data from application documents."""
    extracted = {}
    for doc_path in application.documents:
        doc_text = await load_document(doc_path)
        fields = await call_llm(
            model="gpt-4o",
            prompt=f"Extract all relevant fields from this document:\n{doc_text}",
            response_format="json",
        )
        extracted.update(fields)

    return {
        "applicant_info": extracted.get("applicant", {}),
        "financial_data": extracted.get("financial", {}),
        "supporting_details": extracted.get("details", {}),
    }


@activity.defn
async def validate_application(data: dict) -> dict:
    """Validate extracted data against business rules."""
    errors = []
    warnings = []

    required_fields = ["applicant_info", "financial_data"]
    for field in required_fields:
        if not data.get(field):
            errors.append(f"Missing required section: {field}")

    # LLM-based validation for complex rules
    validation = await call_llm(
        model="gpt-4o-mini",
        prompt=f"Validate this application data for completeness and consistency:\n{data}",
        response_format="json",
    )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings + validation.get("warnings", []),
    }


@activity.defn
async def assess_risk(data: dict) -> AssessmentResult:
    """AI-powered risk assessment."""
    assessment = await call_llm(
        model="gpt-4o",
        prompt=f"""Perform a risk assessment for this application.
Data: {data}

Evaluate:
1. Completeness of documentation
2. Consistency of provided information
3. Risk factors based on application type
4. Any anomalies or red flags

Respond with: risk_level (low/medium/high/critical), risk_score (0-1),
factors (list), recommended_action, requires_human_review (bool)""",
        response_format="json",
    )

    return AssessmentResult(
        risk_level=RiskLevel(assessment["risk_level"]),
        risk_score=assessment["risk_score"],
        factors=assessment["factors"],
        recommended_action=assessment["recommended_action"],
        requires_human_review=assessment.get("requires_human_review", False),
    )


@activity.defn
async def notify_applicant(
    application_id: str,
    decision: str,
    details: str,
) -> bool:
    """Send notification to applicant."""
    await send_notification(
        template="application_decision",
        recipient_id=application_id,
        params={"decision": decision, "details": details},
    )
    return True


@activity.defn
async def request_human_review(
    application_id: str,
    assessment: AssessmentResult,
    data: dict,
) -> str:
    """Create a human review task and return the task ID."""
    task_id = await create_review_task(
        application_id=application_id,
        priority="high" if assessment.risk_level == RiskLevel.CRITICAL else "normal",
        context={
            "risk_assessment": {
                "level": assessment.risk_level.value,
                "score": assessment.risk_score,
                "factors": assessment.factors,
            },
            "application_data": data,
        },
    )
    return task_id


# --- Workflow (durable orchestration) ---

@workflow.defn
class ApplicationProcessingWorkflow:
    """Durable workflow for end-to-end application processing."""

    def __init__(self):
        self.human_decision: HumanDecision | None = None
        self.status = "started"

    @workflow.signal
    async def receive_human_decision(self, decision: HumanDecision):
        """Signal handler: human reviewer submits their decision."""
        self.human_decision = decision

    @workflow.query
    def get_status(self) -> str:
        return self.status

    @workflow.run
    async def run(self, application: Application) -> dict:
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(minutes=5),
            maximum_attempts=3,
        )

        # Step 1: Extract data from documents
        self.status = "extracting_data"
        extracted_data = await workflow.execute_activity(
            extract_application_data,
            application,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=retry_policy,
        )

        # Step 2: Validate
        self.status = "validating"
        validation = await workflow.execute_activity(
            validate_application,
            extracted_data,
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=retry_policy,
        )

        if not validation["is_valid"]:
            self.status = "rejected_invalid"
            await workflow.execute_activity(
                notify_applicant,
                args=[application.application_id, "rejected",
                      f"Validation failed: {validation['errors']}"],
                start_to_close_timeout=timedelta(minutes=1),
            )
            return {"status": "rejected", "reason": "validation_failed",
                    "errors": validation["errors"]}

        # Step 3: Risk assessment
        self.status = "assessing_risk"
        assessment = await workflow.execute_activity(
            assess_risk,
            extracted_data,
            start_to_close_timeout=timedelta(minutes=3),
            retry_policy=retry_policy,
        )

        # Step 4: Decision — automatic or human
        if assessment.requires_human_review or assessment.risk_level in (
            RiskLevel.HIGH, RiskLevel.CRITICAL
        ):
            # Human-in-the-loop
            self.status = "pending_human_review"
            review_task_id = await workflow.execute_activity(
                request_human_review,
                args=[application.application_id, assessment, extracted_data],
                start_to_close_timeout=timedelta(minutes=2),
            )

            # Wait for human decision — can wait for hours/days
            # Temporal handles this durably without consuming resources
            await workflow.wait_condition(
                lambda: self.human_decision is not None,
                timeout=timedelta(hours=48),
            )

            if self.human_decision is None:
                # Timeout — escalate
                self.status = "escalated_timeout"
                return {"status": "escalated", "reason": "review_timeout"}

            decision = "approved" if self.human_decision.approved else "rejected"
            decision_details = self.human_decision.comments

        else:
            # Automatic approval for low-risk applications
            decision = "approved"
            decision_details = (
                f"Auto-approved. Risk: {assessment.risk_level.value} "
                f"(score: {assessment.risk_score:.2f})"
            )

        # Step 5: Notify
        self.status = f"notifying_{decision}"
        await workflow.execute_activity(
            notify_applicant,
            args=[application.application_id, decision, decision_details],
            start_to_close_timeout=timedelta(minutes=1),
        )

        self.status = f"completed_{decision}"
        return {
            "status": decision,
            "risk_assessment": {
                "level": assessment.risk_level.value,
                "score": assessment.risk_score,
            },
            "human_reviewed": self.human_decision is not None,
            "details": decision_details,
        }
```

**Key Temporal features used:**
- **Activities** with retry policies handle transient failures (LLM timeouts, API errors).
- **Signals** allow external events (human decisions) to be injected into a running workflow.
- **wait_condition** pauses the workflow durably --- no threads or connections are held.
- **Queries** allow external systems to check workflow status without affecting execution.

---

## Cross-Cutting Concerns

### Observability

Every agent call, tool invocation, and routing decision should be traced. Langfuse provides LLM-specific observability.

```python
from langfuse import Langfuse
from langfuse.decorators import observe
import time

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com",
)


@observe(name="support-pipeline")
async def handle_support_request(conversation_id: str, message: str):
    """Traced support request handler."""

    # Trace the router
    with langfuse.trace(
        name="router",
        metadata={"conversation_id": conversation_id},
    ) as trace:
        start = time.perf_counter()
        classification = await classify_intent(message)
        latency = time.perf_counter() - start

        trace.score(name="confidence", value=classification["confidence"])
        trace.update(
            output=classification,
            metadata={
                "latency_ms": latency * 1000,
                "intent": classification["intent"],
            },
        )

    # Trace the agent execution
    with langfuse.trace(
        name=f"{classification['intent']}-agent",
        parent_observation_id=trace.id,
    ) as agent_trace:
        response = await dispatch_to_agent(classification, message)

        agent_trace.generation(
            name="llm-call",
            model=response["model"],
            input=message,
            output=response["answer"],
            usage={
                "input_tokens": response["tokens"]["input"],
                "output_tokens": response["tokens"]["output"],
            },
        )

    return response
```

### Cost Optimization

LLM costs can escalate rapidly. A cost manager routes requests to appropriate models and enforces budgets.

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio


@dataclass
class ModelCost:
    input_per_1k: float    # USD per 1K input tokens
    output_per_1k: float   # USD per 1K output tokens
    latency_p50_ms: int    # Typical latency


class CostManager:
    """Track and optimize LLM costs across the agent system."""

    MODEL_COSTS: dict[str, ModelCost] = {
        "gpt-4o":        ModelCost(0.0025, 0.010, 800),
        "gpt-4o-mini":   ModelCost(0.00015, 0.0006, 300),
        "o1":            ModelCost(0.015, 0.060, 3000),
        "o1-mini":       ModelCost(0.003, 0.012, 1500),
        "claude-3.5-sonnet": ModelCost(0.003, 0.015, 600),
        "claude-3-haiku":   ModelCost(0.00025, 0.00125, 200),
    }

    def __init__(self, daily_budget_usd: float = 100.0):
        self.daily_budget = daily_budget_usd
        self.daily_spend: float = 0.0
        self.last_reset = datetime.now(timezone.utc).date()
        self._lock = asyncio.Lock()
        self.usage_log: list[dict] = []

    def select_model(
        self,
        task_complexity: str,      # "simple", "moderate", "complex", "reasoning"
        latency_requirement: str,  # "fast", "normal", "relaxed"
    ) -> str:
        """Select the most cost-effective model for the task."""
        model_map = {
            ("simple", "fast"):      "gpt-4o-mini",
            ("simple", "normal"):    "gpt-4o-mini",
            ("simple", "relaxed"):   "claude-3-haiku",
            ("moderate", "fast"):    "gpt-4o-mini",
            ("moderate", "normal"):  "gpt-4o",
            ("moderate", "relaxed"): "claude-3.5-sonnet",
            ("complex", "fast"):     "gpt-4o",
            ("complex", "normal"):   "claude-3.5-sonnet",
            ("complex", "relaxed"):  "claude-3.5-sonnet",
            ("reasoning", "fast"):   "o1-mini",
            ("reasoning", "normal"): "o1-mini",
            ("reasoning", "relaxed"):"o1",
        }
        return model_map.get(
            (task_complexity, latency_requirement),
            "gpt-4o-mini",
        )

    async def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        conversation_id: str = "",
    ) -> dict:
        """Track cost of an LLM call. Raises if budget exceeded."""
        async with self._lock:
            today = datetime.now(timezone.utc).date()
            if today != self.last_reset:
                self.daily_spend = 0.0
                self.last_reset = today

            cost_info = self.MODEL_COSTS.get(model)
            if not cost_info:
                raise ValueError(f"Unknown model: {model}")

            cost = (
                (input_tokens / 1000) * cost_info.input_per_1k +
                (output_tokens / 1000) * cost_info.output_per_1k
            )

            if self.daily_spend + cost > self.daily_budget:
                raise RuntimeError(
                    f"Daily budget exceeded: ${self.daily_spend:.2f} + "
                    f"${cost:.4f} > ${self.daily_budget:.2f}"
                )

            self.daily_spend += cost

            record = {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "daily_total": self.daily_spend,
                "conversation_id": conversation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.usage_log.append(record)

            return record


# --- Usage ---
cost_mgr = CostManager(daily_budget_usd=50.0)

model = cost_mgr.select_model(
    task_complexity="simple",
    latency_requirement="fast",
)
print(f"Selected model: {model}")  # gpt-4o-mini

record = await cost_mgr.track(
    model=model,
    input_tokens=500,
    output_tokens=200,
    conversation_id="conv-123",
)
print(f"Cost: ${record['cost_usd']:.6f}, Daily total: ${record['daily_total']:.4f}")
```

### Security --- PII Protection

PII must be masked before sending data to LLMs. This is a compliance requirement for any production agent system.

```python
import re
from dataclasses import dataclass


@dataclass
class PIIScanResult:
    cleaned_text: str
    pii_found: bool
    detections: list[dict]


class PIIProtector:
    """Detect and mask PII before LLM calls."""

    PATTERNS = {
        "email": {
            "regex": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "mask": "[EMAIL_REDACTED]",
        },
        "phone": {
            # International format: +1-xxx-xxx-xxxx, +44 xxxx xxxxxx, etc.
            "regex": r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            "mask": "[PHONE_REDACTED]",
        },
        "card_number": {
            # Standard 16-digit card numbers with optional separators
            "regex": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "mask": "[CARD_REDACTED]",
        },
        "national_id": {
            # Common formats: SSN (xxx-xx-xxxx), or generic 9-11 digit IDs
            "regex": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            "mask": "[ID_REDACTED]",
        },
        "ip_address": {
            "regex": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "mask": "[IP_REDACTED]",
        },
        "date_of_birth": {
            # Common date formats: MM/DD/YYYY, DD-MM-YYYY, YYYY-MM-DD
            "regex": r'\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b',
            "mask": "[DOB_REDACTED]",
        },
    }

    def __init__(self, additional_patterns: dict | None = None):
        self.patterns = dict(self.PATTERNS)
        if additional_patterns:
            self.patterns.update(additional_patterns)

        # Compile all regexes
        self._compiled = {
            name: re.compile(info["regex"])
            for name, info in self.patterns.items()
        }

    def scan(self, text: str) -> tuple[str, bool]:
        """Scan text for PII and return (cleaned_text, pii_found)."""
        result = self.scan_detailed(text)
        return result.cleaned_text, result.pii_found

    def scan_detailed(self, text: str) -> PIIScanResult:
        """Scan text with detailed detection report."""
        cleaned = text
        detections = []

        for name, pattern in self._compiled.items():
            matches = pattern.finditer(cleaned)
            for match in matches:
                detections.append({
                    "type": name,
                    "value_hash": hash(match.group()),  # Hash, not the actual value
                    "position": match.span(),
                })

            mask = self.patterns[name]["mask"]
            cleaned = pattern.sub(mask, cleaned)

        return PIIScanResult(
            cleaned_text=cleaned,
            pii_found=len(detections) > 0,
            detections=detections,
        )


# --- Usage ---
protector = PIIProtector()

text = (
    "Customer John Smith (john.smith@example.com) called from +1-555-123-4567. "
    "Card ending 4532-1234-5678-9012. SSN: 123-45-6789. "
    "DOB: 03/15/1990. IP: 192.168.1.100."
)

cleaned, pii_found = protector.scan(text)
print(f"PII found: {pii_found}")
print(f"Cleaned: {cleaned}")
# Customer John Smith ([EMAIL_REDACTED]) called from [PHONE_REDACTED].
# Card ending [CARD_REDACTED]. SSN: [ID_REDACTED].
# DOB: [DOB_REDACTED]. IP: [IP_REDACTED].
```

### Testing Agent Systems

Agent testing requires a multi-level strategy because LLMs are non-deterministic and agent behavior emerges from the interaction of multiple components.

**Level 1: Unit Tests** --- Test each agent in isolation with deterministic inputs.

```python
import pytest
from unittest.mock import AsyncMock, patch


class TestRouterAgent:
    """Unit tests for the router agent."""

    @pytest.mark.asyncio
    async def test_faq_classification(self):
        """Router should classify FAQ-type questions correctly."""
        test_cases = [
            {
                "input": "What are your business hours?",
                "expected_intent": "faq",
                "min_confidence": 0.8,
            },
            {
                "input": "How do I reset my password?",
                "expected_intent": "faq",
                "min_confidence": 0.7,
            },
            {
                "input": "I want to check my order status for #12345",
                "expected_intent": "transaction",
                "min_confidence": 0.7,
            },
            {
                "input": "Your service is terrible and I want a full refund",
                "expected_intent": "complaint",
                "min_confidence": 0.6,
            },
        ]

        for case in test_cases:
            result = await classify_intent(case["input"])
            assert result["intent"] == case["expected_intent"], (
                f"Expected {case['expected_intent']} for: {case['input']}, "
                f"got {result['intent']}"
            )
            assert result["confidence"] >= case["min_confidence"], (
                f"Confidence {result['confidence']} below threshold "
                f"{case['min_confidence']} for: {case['input']}"
            )

    @pytest.mark.asyncio
    async def test_pii_detection_before_routing(self):
        """Router should detect PII in messages."""
        message = "My email is user@example.com and card is 4111-1111-1111-1111"
        protector = PIIProtector()
        cleaned, pii_found = protector.scan(message)

        assert pii_found is True
        assert "user@example.com" not in cleaned
        assert "4111" not in cleaned
```

**Level 2: Integration Tests** --- Test agent pairs and context passing.

```python
class TestSupportPipeline:
    """Integration tests for the full support pipeline."""

    @pytest.mark.asyncio
    async def test_faq_resolution_flow(self):
        """Test full flow: router → FAQ agent → resolution."""
        initial_state = SupportState(
            messages=[HumanMessage(content="What is the return policy?")],
            conversation_id="test-conv-001",
            customer_id="cust-123",
            intent=None,
            confidence=0.0,
            agent_type=None,
            tool_results=[],
            escalation_reason=None,
            resolution_status="open",
            turn_count=0,
            total_tokens=0,
            pii_detected=False,
        )

        result = await support_app.ainvoke(initial_state)

        assert result["resolution_status"] == "resolved"
        assert result["intent"] == "faq"
        assert len(result["messages"]) > 1  # At least one AI response
        assert result["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_low_confidence_escalation(self):
        """Ambiguous messages should escalate to human."""
        state = SupportState(
            messages=[HumanMessage(content="hmm")],
            conversation_id="test-conv-002",
            customer_id="cust-456",
            intent=None,
            confidence=0.0,
            agent_type=None,
            tool_results=[],
            escalation_reason=None,
            resolution_status="open",
            turn_count=0,
            total_tokens=0,
            pii_detected=False,
        )

        result = await support_app.ainvoke(state)

        assert result["resolution_status"] == "escalated"
```

**Level 3: Evaluation Dataset** --- Measure agent quality at scale.

```python
import json
from dataclasses import dataclass


@dataclass
class EvalCase:
    input_message: str
    expected_intent: str
    expected_resolution: str
    quality_criteria: list[str]


class AgentEvaluator:
    """Evaluate agent system against a curated dataset."""

    def __init__(self, eval_dataset_path: str):
        with open(eval_dataset_path) as f:
            raw = json.load(f)
        self.cases = [EvalCase(**c) for c in raw]

    async def run_evaluation(self) -> dict:
        results = {
            "total": len(self.cases),
            "intent_correct": 0,
            "resolution_correct": 0,
            "quality_scores": [],
            "failures": [],
        }

        for case in self.cases:
            try:
                output = await support_app.ainvoke({
                    "messages": [HumanMessage(content=case.input_message)],
                    "conversation_id": f"eval-{hash(case.input_message)}",
                    "customer_id": "eval-user",
                    "intent": None,
                    "confidence": 0.0,
                    "agent_type": None,
                    "tool_results": [],
                    "escalation_reason": None,
                    "resolution_status": "open",
                    "turn_count": 0,
                    "total_tokens": 0,
                    "pii_detected": False,
                })

                if output["intent"] == case.expected_intent:
                    results["intent_correct"] += 1
                if output["resolution_status"] == case.expected_resolution:
                    results["resolution_correct"] += 1

                # LLM-as-judge for quality
                quality = await evaluate_response_quality(
                    input_message=case.input_message,
                    response=output["messages"][-1].content,
                    criteria=case.quality_criteria,
                )
                results["quality_scores"].append(quality["score"])

            except Exception as e:
                results["failures"].append({
                    "input": case.input_message,
                    "error": str(e),
                })

        results["intent_accuracy"] = results["intent_correct"] / results["total"]
        results["resolution_accuracy"] = results["resolution_correct"] / results["total"]
        results["avg_quality"] = (
            sum(results["quality_scores"]) / len(results["quality_scores"])
            if results["quality_scores"] else 0
        )

        return results
```

---

## Key Insights

> **Designing multi-agent systems:** Follow the framework: Requirements -> Agents -> Orchestration (LangGraph) -> Data (RAG) -> Infrastructure (Redis + Kafka + Temporal) -> Safety (PII, prompt injection, cost budgets, HITL). Every decision at each layer is constrained by the layers above it.

> **Cost control:** (1) Model routing --- cheap models for simple tasks, expensive models for complex reasoning. (2) Semantic caching --- identical or similar queries return cached responses. (3) Token budgets per conversation with hard cutoffs. (4) Batching --- group multiple small requests into a single LLM call. (5) Monitoring cost per conversation and per agent to identify waste.

> **Testing strategy:** Testing pyramid: (1) Unit --- each agent on an evaluation dataset with deterministic mocks. (2) Integration --- agent pairs, context passing, state transitions. (3) End-to-end --- full scenarios from user input to final output. (4) Regression --- a fixed dataset re-evaluated on every prompt or model change. (5) A/B testing --- measure real-world impact on resolution rate, CSAT, and cost.

> **Security checklist:** (1) PII masking before LLM calls. (2) Prompt injection protection (input validation, output filtering). (3) Least privilege per agent --- each agent accesses only the tools and data it needs. (4) Audit logging via Kafka --- every decision is traceable. (5) Rate limiting per user and per agent. (6) Human-in-the-loop for critical operations (financial transactions, account deletions, high-risk decisions).

---

## References

- LangSmith Documentation: [https://docs.smith.langchain.com/](https://docs.smith.langchain.com/)
- Langfuse Documentation: [https://langfuse.com/docs](https://langfuse.com/docs)
- Temporal Documentation: [https://docs.temporal.io/](https://docs.temporal.io/)
- "Building LLM Apps" course: [https://www.deeplearning.ai/](https://www.deeplearning.ai/)
- ML System Design: [https://github.com/chiphuyen/machine-learning-systems-design](https://github.com/chiphuyen/machine-learning-systems-design)
- OWASP LLM Top 10: [https://owasp.org/www-project-top-10-for-large-language-model-applications/](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
