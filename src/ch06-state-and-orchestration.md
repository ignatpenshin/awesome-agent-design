# Chapter 6: State Management & Orchestration Infrastructure

> *"State is the root of all complexity in distributed systems — and agents are distributed systems."*

---

Agent systems are inherently stateful. A multi-step workflow that calls an LLM, retrieves documents, waits for human approval, and writes to a database must track progress, handle failures, and resume after crashes. This is not a problem that can be solved with retries alone. It requires durable execution infrastructure.

This chapter covers three technologies that form the orchestration backbone of production agent systems: **Temporal** for durable workflows with built-in fault tolerance, **Redis Streams** for real-time inter-agent communication, and **Apache Kafka** for event sourcing and audit trails. Together, they provide the state management layer that sits beneath every reliable agent deployment.

---

## Temporal — Durable Execution

Temporal is a workflow orchestration platform that guarantees workflow completion even through infrastructure failures, process crashes, and network partitions. It achieves this through **event sourcing**: every step of a workflow is recorded as an event in Temporal's persistence layer. If a worker crashes mid-workflow, a new worker replays the event history to reconstruct the exact state and continues from where execution left off.

### Core Concepts

**Workflow** is a deterministic function that defines the overall orchestration logic. It specifies which activities to call, in what order, with what retry policies, and how to handle signals from the outside world. Workflow code must be deterministic because Temporal replays it from the event history to rebuild state.

**Activity** is a function that performs a single unit of work — calling an LLM, querying a database, sending an email. Activities are the non-deterministic parts of the system. They can fail, be retried, and time out independently.

**Worker** is a process that polls a task queue and executes workflows and activities. Workers are stateless; any worker can pick up any task from the queue. This makes scaling trivial: add more worker processes to increase throughput.

**Task Queue** is the channel that connects workflow/activity invocations to workers. Different task queues can route work to different worker pools, enabling resource isolation and priority scheduling.

### RAG Agent Workflow

The following example implements a complete RAG (Retrieval-Augmented Generation) agent as a Temporal workflow. It demonstrates activities with retry policies, workflow orchestration with timeouts, and the worker setup required to run it all.

```python
import asyncio
from datetime import timedelta
from dataclasses import dataclass

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.common import RetryPolicy


# ── Data classes ──

@dataclass
class RAGRequest:
    query: str
    user_id: str
    session_id: str


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    confidence: float


@dataclass
class LLMResponse:
    text: str
    confidence: float


@dataclass
class SearchResult:
    content: str
    source: str
    score: float


# ── Activities ──
# Each activity is a single unit of work that can fail and be retried.

@activity.defn
async def call_llm(
    query: str, context: str, model: str = "gpt-4o"
) -> LLMResponse:
    """Call the LLM with the query and retrieved context.

    This is an activity because LLM calls are non-deterministic,
    can fail due to rate limits, and have variable latency.
    """
    activity.logger.info(
        "Calling LLM with query: %s", query[:80]
    )

    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question based on the provided "
                    "context. If the context doesn't contain "
                    "relevant information, say so.\n\n"
                    f"Context:\n{context}"
                )
            },
            {"role": "user", "content": query}
        ],
        temperature=0.1
    )

    return LLMResponse(
        text=response.choices[0].message.content,
        confidence=0.85  # In production, compute from logprobs
    )


@activity.defn
async def search_knowledge_base(
    query: str, top_k: int = 5
) -> list[SearchResult]:
    """Search the vector database for relevant documents.

    This activity encapsulates the retrieval step of RAG.
    """
    activity.logger.info(
        "Searching knowledge base for: %s", query[:80]
    )

    # In production, this calls your vector database
    # (Qdrant, Pinecone, Weaviate, etc.)
    import numpy as np

    # Simulated retrieval
    results = [
        SearchResult(
            content=f"Document about {query}",
            source=f"doc_{i}.md",
            score=float(np.random.uniform(0.7, 0.95))
        )
        for i in range(top_k)
    ]

    return sorted(results, key=lambda r: r.score, reverse=True)


@activity.defn
async def save_to_database(
    user_id: str,
    session_id: str,
    query: str,
    answer: str,
    sources: list[str]
) -> bool:
    """Persist the interaction to the database for audit and analytics.

    This activity handles the write path. If it fails, Temporal
    retries it without re-running the LLM call.
    """
    activity.logger.info(
        "Saving interaction for user %s, session %s",
        user_id, session_id
    )

    import asyncpg
    conn = await asyncpg.connect(
        "postgresql://localhost:5432/agents"
    )

    try:
        await conn.execute(
            """
            INSERT INTO interactions
                (user_id, session_id, query, answer, sources,
                 created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            """,
            user_id, session_id, query, answer, sources
        )
        return True
    finally:
        await conn.close()


# ── Workflow ──

@workflow.defn
class RAGAgentWorkflow:
    """Durable RAG agent workflow.

    Orchestrates retrieval, LLM generation, and persistence
    with full fault tolerance. If a worker crashes after the
    LLM call but before saving to the database, a new worker
    replays the history, skips the LLM call (result is in the
    event log), and retries only the database write.
    """

    @workflow.run
    async def run(self, request: RAGRequest) -> RAGResponse:
        # Define retry policies for different activity types
        llm_retry = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            backoff_coefficient=2.0,
            maximum_attempts=3,
            maximum_interval=timedelta(seconds=30),
            non_retryable_error_types=["ValueError"]
        )

        db_retry = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_attempts=5,
            maximum_interval=timedelta(seconds=60)
        )

        # Step 1: Search the knowledge base
        search_results = await workflow.execute_activity(
            search_knowledge_base,
            args=[request.query],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=llm_retry
        )

        # Step 2: Build context from search results
        context = "\n\n".join(
            f"[Source: {r.source}] {r.content}"
            for r in search_results
        )
        sources = [r.source for r in search_results]

        # Step 3: Call LLM with retrieved context
        llm_response = await workflow.execute_activity(
            call_llm,
            args=[request.query, context],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=llm_retry
        )

        # Step 4: Save interaction to database
        await workflow.execute_activity(
            save_to_database,
            args=[
                request.user_id,
                request.session_id,
                request.query,
                llm_response.text,
                sources
            ],
            start_to_close_timeout=timedelta(seconds=15),
            retry_policy=db_retry
        )

        return RAGResponse(
            answer=llm_response.text,
            sources=sources,
            confidence=llm_response.confidence
        )


# ── Worker setup ──

async def run_worker():
    """Start a Temporal worker that processes RAG workflows."""
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="rag-agent-queue",
        workflows=[RAGAgentWorkflow],
        activities=[
            call_llm,
            search_knowledge_base,
            save_to_database
        ]
    )

    print("Worker started, listening on 'rag-agent-queue'")
    await worker.run()


# ── Execute a workflow ──

async def main():
    client = await Client.connect("localhost:7233")

    # Start the workflow — returns immediately
    result = await client.execute_workflow(
        RAGAgentWorkflow.run,
        RAGRequest(
            query="How do I configure retry policies in Temporal?",
            user_id="user_123",
            session_id="sess_abc"
        ),
        id="rag-workflow-user123-001",
        task_queue="rag-agent-queue"
    )

    print(f"Answer: {result.answer}")
    print(f"Sources: {result.sources}")
    print(f"Confidence: {result.confidence}")


if __name__ == "__main__":
    asyncio.run(main())
```

The key insight is the separation between the workflow (deterministic orchestration logic) and activities (non-deterministic side effects). When a worker replays a workflow, it does not re-execute completed activities. Instead, it reads their results from the event history. This means an LLM call that took 5 seconds and cost money is never duplicated after a crash.

### Signals and Queries

Real-world agent workflows often need to pause for human approval, receive external events, or expose their current status to monitoring systems. Temporal provides **signals** for injecting data into a running workflow and **queries** for reading workflow state without affecting execution.

```python
import asyncio
from datetime import timedelta
from dataclasses import dataclass, field
from enum import Enum

from temporalio import workflow
from temporalio.common import RetryPolicy


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ActionRequest:
    action: str
    description: str
    risk_level: str  # "low", "medium", "high"
    requested_by: str


@dataclass
class WorkflowState:
    status: str
    current_step: str
    approval_status: str
    result: str | None = None


@workflow.defn
class HumanInTheLoopWorkflow:
    """Workflow that pauses for human approval on high-risk actions.

    Demonstrates:
    - Signals: external systems can approve/reject the action
    - Queries: monitoring can read the current state at any time
    - Wait conditions: workflow blocks until a signal arrives or
      a timeout expires
    """

    def __init__(self):
        self._approval_status = ApprovalStatus.PENDING
        self._current_step = "initializing"
        self._result: str | None = None

    # ── Signal: receive approval/rejection from outside ──
    @workflow.signal
    async def approve(self, approved: bool, reviewer: str):
        """Signal handler for human approval decisions."""
        if approved:
            self._approval_status = ApprovalStatus.APPROVED
            workflow.logger.info(
                "Action approved by %s", reviewer
            )
        else:
            self._approval_status = ApprovalStatus.REJECTED
            workflow.logger.info(
                "Action rejected by %s", reviewer
            )

    # ── Query: read current state without side effects ──
    @workflow.query
    def get_status(self) -> WorkflowState:
        """Query handler to inspect the workflow's current state."""
        return WorkflowState(
            status="running",
            current_step=self._current_step,
            approval_status=self._approval_status.value,
            result=self._result
        )

    @workflow.run
    async def run(self, request: ActionRequest) -> dict:
        self._current_step = "analyzing_request"

        # Step 1: Analyze the request
        analysis = await workflow.execute_activity(
            analyze_action,
            args=[request.action, request.description],
            start_to_close_timeout=timedelta(seconds=30)
        )

        # Step 2: Check if human approval is needed
        if request.risk_level in ("medium", "high"):
            self._current_step = "awaiting_human_approval"

            # Notify the review team
            await workflow.execute_activity(
                send_approval_request,
                args=[request.requested_by, request.description],
                start_to_close_timeout=timedelta(seconds=10)
            )

            # Block until signal arrives or timeout
            try:
                await workflow.wait_condition(
                    lambda: self._approval_status
                    != ApprovalStatus.PENDING,
                    timeout=timedelta(hours=24)
                )
            except asyncio.TimeoutError:
                self._approval_status = ApprovalStatus.REJECTED
                return {
                    "status": "timeout",
                    "message": "Approval timed out after 24 hours"
                }

            if self._approval_status == ApprovalStatus.REJECTED:
                return {
                    "status": "rejected",
                    "message": "Action was rejected by reviewer"
                }

        # Step 3: Execute the approved action
        self._current_step = "executing_action"

        result = await workflow.execute_activity(
            execute_action,
            args=[request.action, request.description],
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )

        self._current_step = "completed"
        self._result = result

        return {"status": "completed", "result": result}


# ── Client-side: sending signals and queries ──

async def approve_workflow(
    workflow_id: str, approved: bool, reviewer: str
):
    """Send an approval signal to a running workflow."""
    from temporalio.client import Client

    client = await Client.connect("localhost:7233")
    handle = client.get_workflow_handle(workflow_id)

    await handle.signal(
        HumanInTheLoopWorkflow.approve,
        args=[approved, reviewer]
    )


async def check_workflow_status(workflow_id: str):
    """Query a running workflow for its current state."""
    from temporalio.client import Client

    client = await Client.connect("localhost:7233")
    handle = client.get_workflow_handle(workflow_id)

    state = await handle.query(
        HumanInTheLoopWorkflow.get_status
    )

    print(f"Step: {state.current_step}")
    print(f"Approval: {state.approval_status}")
```

The `wait_condition` call is what makes this pattern powerful. The workflow is not polling. It is not consuming resources while waiting. Temporal persists the workflow state and only resumes execution when a matching signal arrives (or the timeout expires). A workflow can wait for days with zero resource consumption.

### Saga Pattern

Long-running agent workflows that span multiple services need a compensation strategy for partial failures. If step 3 of a 5-step process fails, steps 1 and 2 may have already produced side effects that need to be reversed. The Saga pattern implements this by maintaining a stack of compensation actions that are executed in reverse order on failure.

```python
from datetime import timedelta
from dataclasses import dataclass

from temporalio import activity, workflow
from temporalio.common import RetryPolicy


@dataclass
class OrderRequest:
    order_id: str
    user_id: str
    amount: float
    product_id: str


# ── Activities and their compensations ──

@activity.defn
async def reserve_funds(
    user_id: str, amount: float
) -> str:
    """Reserve funds in the user's account."""
    # Call payment service to place a hold
    activity.logger.info(
        "Reserving $%.2f for user %s", amount, user_id
    )
    return "reservation_12345"


@activity.defn
async def release_funds(
    user_id: str, reservation_id: str
):
    """Compensation: release previously reserved funds."""
    activity.logger.info(
        "Releasing reservation %s for user %s",
        reservation_id, user_id
    )


@activity.defn
async def verify_identity(user_id: str) -> dict:
    """Verify user identity through KYC service."""
    activity.logger.info(
        "Verifying identity for user %s", user_id
    )
    return {"verified": True, "verification_id": "kyc_67890"}


@activity.defn
async def revoke_verification(verification_id: str):
    """Compensation: revoke identity verification."""
    activity.logger.info(
        "Revoking verification %s", verification_id
    )


@activity.defn
async def approve_order(
    order_id: str, user_id: str, amount: float
) -> dict:
    """Final approval and order creation."""
    activity.logger.info(
        "Approving order %s for user %s, amount $%.2f",
        order_id, user_id, amount
    )
    return {"approved": True, "confirmation": "CONF-11111"}


@activity.defn
async def cancel_order(order_id: str):
    """Compensation: cancel an approved order."""
    activity.logger.info("Cancelling order %s", order_id)


@activity.defn
async def send_rejection_notification(
    user_id: str, order_id: str, reason: str
):
    """Notify the user that their order was rejected."""
    activity.logger.info(
        "Sending rejection to user %s for order %s: %s",
        user_id, order_id, reason
    )


# ── Saga Workflow ──

@workflow.defn
class OrderProcessingSaga:
    """Saga pattern: execute steps with compensations on failure.

    Each step pushes its compensation onto a stack. If any step
    fails, all compensations are executed in reverse order to
    restore consistent state.
    """

    @workflow.run
    async def run(self, request: OrderRequest) -> dict:
        compensations: list[tuple[callable, list]] = []

        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_attempts=3
        )

        try:
            # Step 1: Reserve funds
            reservation_id = await workflow.execute_activity(
                reserve_funds,
                args=[request.user_id, request.amount],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=retry_policy
            )
            compensations.append(
                (release_funds,
                 [request.user_id, reservation_id])
            )

            # Step 2: Verify identity
            verification = await workflow.execute_activity(
                verify_identity,
                args=[request.user_id],
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=retry_policy
            )

            if not verification["verified"]:
                raise RuntimeError(
                    "Identity verification failed"
                )

            compensations.append(
                (revoke_verification,
                 [verification["verification_id"]])
            )

            # Step 3: Approve the order
            approval = await workflow.execute_activity(
                approve_order,
                args=[
                    request.order_id,
                    request.user_id,
                    request.amount
                ],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=retry_policy
            )
            compensations.append(
                (cancel_order, [request.order_id])
            )

            return {
                "status": "approved",
                "confirmation": approval["confirmation"],
                "reservation_id": reservation_id,
            }

        except Exception as e:
            # Execute compensations in reverse order
            workflow.logger.warning(
                "Saga failed at step, executing %d compensations: "
                "%s", len(compensations), str(e)
            )

            for comp_activity, comp_args in reversed(compensations):
                try:
                    await workflow.execute_activity(
                        comp_activity,
                        args=comp_args,
                        start_to_close_timeout=timedelta(
                            seconds=30
                        ),
                        retry_policy=RetryPolicy(
                            maximum_attempts=5
                        )
                    )
                except Exception as comp_error:
                    # Log but continue — compensations must be
                    # best-effort
                    workflow.logger.error(
                        "Compensation failed: %s", str(comp_error)
                    )

            # Notify the user
            await workflow.execute_activity(
                send_rejection_notification,
                args=[
                    request.user_id,
                    request.order_id,
                    str(e)
                ],
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=retry_policy
            )

            return {"status": "rejected", "reason": str(e)}
```

The compensation stack is the central mechanism. Each successful step pushes an undo action. On failure, the stack is unwound in reverse order, ensuring that side effects are cleaned up in the correct sequence. This is essential for agent systems that interact with external services — you cannot leave reserved funds locked or half-created records in a database.

### Temporal vs Celery vs Airflow

| Feature | Temporal | Celery | Airflow |
|---|---|---|---|
| **Primary use case** | Durable workflows | Task queues | Data pipelines (DAGs) |
| **Durability** | Full event sourcing | Redis/RabbitMQ persistence | Database-backed state |
| **Failure recovery** | Automatic replay from history | Manual retry, DLQ | Task-level retry, manual re-run |
| **Long-running workflows** | Native (days/months) | Not designed for this | DAG schedules, not continuous |
| **Human-in-the-loop** | Signals and wait conditions | Not supported natively | External trigger sensors |
| **Dynamic workflows** | Full programming model | Chain/chord patterns | Limited dynamic DAGs |
| **Latency** | Sub-second dispatch | Sub-second dispatch | Minute-level scheduler |
| **Best for agents** | Complex multi-step orchestration | Simple fire-and-forget tasks | Scheduled batch processing |

Temporal is the strongest fit for agent orchestration because agent workflows are inherently dynamic (the next step depends on LLM output), long-running (they may wait for human approval), and failure-prone (external API calls fail). Celery works well for simple background tasks like sending notifications. Airflow excels at scheduled batch processing but is not designed for the interactive, dynamic execution patterns that agents require.

---

## Redis Streams

Redis Streams is a log-based data structure built into Redis that supports consumer groups, message acknowledgment, and persistent message storage. For agent systems, it provides the real-time communication layer that connects agents to each other and to external event sources.

### Fundamentals

```python
import redis.asyncio as redis
import json
import asyncio

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, decode_responses=True)


# ── Producing messages ──

async def publish_event():
    """Add an event to a stream."""
    message_id = await r.xadd(
        "agent-events",               # Stream name
        {
            "type": "task_completed",
            "agent_id": "agent-001",
            "payload": json.dumps({
                "task_id": "task_123",
                "result": "success",
                "duration_ms": 450
            })
        },
        maxlen=10000  # Cap stream length for memory management
    )
    print(f"Published event: {message_id}")
    # Output: Published event: 1700000000000-0


# ── Reading messages (simple) ──

async def read_events():
    """Read events from a stream starting from a given ID."""
    events = await r.xread(
        {"agent-events": "0-0"},  # Read from beginning
        count=10,                  # Max messages per read
        block=5000                 # Block for 5 seconds if empty
    )

    for stream_name, messages in events:
        for message_id, data in messages:
            print(f"[{message_id}] {data}")


# ── Consumer groups for distributed processing ──

async def setup_consumer_group():
    """Create a consumer group for load-balanced consumption."""
    try:
        await r.xgroup_create(
            "agent-events",       # Stream name
            "processing-group",   # Consumer group name
            id="0",               # Start from beginning
            mkstream=True         # Create stream if it doesn't exist
        )
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
        # Group already exists — safe to ignore


async def consume_as_group(consumer_name: str):
    """Consume messages as part of a consumer group.

    Each message is delivered to exactly one consumer in the group.
    Messages must be acknowledged after processing.
    """
    while True:
        messages = await r.xreadgroup(
            groupname="processing-group",
            consumername=consumer_name,
            streams={"agent-events": ">"},  # ">" = new messages only
            count=5,
            block=2000
        )

        for stream_name, stream_messages in messages:
            for message_id, data in stream_messages:
                # Process the message
                print(
                    f"[{consumer_name}] Processing {message_id}: "
                    f"{data}"
                )

                # Acknowledge after successful processing
                await r.xack(
                    "agent-events",
                    "processing-group",
                    message_id
                )
```

Consumer groups are the critical feature for agent systems. They provide at-least-once delivery and load balancing across multiple consumers. If an agent crashes before acknowledging a message, Redis re-delivers it to another consumer in the group. This ensures that no event is lost, even during rolling deployments or agent failures.

### Inter-Agent Communication

The following `AgentEventBus` wraps Redis Streams into a clean publish/subscribe interface for agent-to-agent communication:

```python
import redis.asyncio as redis
import json
import asyncio
from datetime import datetime
from typing import Any, Callable, Awaitable
from dataclasses import dataclass, asdict


@dataclass
class AgentEvent:
    event_type: str
    source_agent: str
    payload: dict[str, Any]
    timestamp: str | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class AgentEventBus:
    """Redis Streams-based event bus for inter-agent communication.

    Supports:
    - Publishing events to named channels (streams)
    - Subscribing to channels with consumer groups
    - Automatic deserialization of events
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_stream_length: int = 10000
    ):
        self.redis_url = redis_url
        self.max_stream_length = max_stream_length
        self._redis: redis.Redis | None = None

    async def connect(self):
        """Initialize the Redis connection."""
        self._redis = redis.from_url(
            self.redis_url, decode_responses=True
        )

    async def disconnect(self):
        """Close the Redis connection."""
        if self._redis:
            await self._redis.aclose()

    async def publish(
        self, channel: str, event: AgentEvent
    ) -> str:
        """Publish an event to a channel.

        Args:
            channel: The stream/channel name.
            event: The event to publish.

        Returns:
            The Redis message ID.
        """
        data = {
            "event_type": event.event_type,
            "source_agent": event.source_agent,
            "payload": json.dumps(event.payload),
            "timestamp": event.timestamp,
        }

        message_id = await self._redis.xadd(
            channel, data, maxlen=self.max_stream_length
        )
        return message_id

    async def subscribe(
        self,
        channel: str,
        group: str,
        consumer: str,
        handler: Callable[[AgentEvent], Awaitable[None]],
        batch_size: int = 10,
        block_ms: int = 1000
    ):
        """Subscribe to a channel and process events.

        Creates the consumer group if it doesn't exist, then
        continuously reads and processes new messages.

        Args:
            channel: The stream/channel to subscribe to.
            group: Consumer group name.
            consumer: This consumer's unique name.
            handler: Async function to process each event.
            batch_size: Number of messages to read per batch.
            block_ms: Milliseconds to block when no messages.
        """
        # Ensure consumer group exists
        try:
            await self._redis.xgroup_create(
                channel, group, id="0", mkstream=True
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        while True:
            try:
                messages = await self._redis.xreadgroup(
                    groupname=group,
                    consumername=consumer,
                    streams={channel: ">"},
                    count=batch_size,
                    block=block_ms
                )

                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        event = AgentEvent(
                            event_type=data["event_type"],
                            source_agent=data["source_agent"],
                            payload=json.loads(data["payload"]),
                            timestamp=data.get("timestamp")
                        )

                        try:
                            await handler(event)
                            await self._redis.xack(
                                channel, group, message_id
                            )
                        except Exception as e:
                            # Don't ack — message will be
                            # redelivered to another consumer
                            print(
                                f"Handler error: {e}, message "
                                f"{message_id} will be redelivered"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Subscription error: {e}")
                await asyncio.sleep(1)


# ── Usage ──

async def main():
    bus = AgentEventBus()
    await bus.connect()

    # Agent A publishes a task result
    await bus.publish(
        "agent-tasks",
        AgentEvent(
            event_type="research_complete",
            source_agent="research-agent",
            payload={
                "query": "latest AI papers",
                "results": ["paper1.pdf", "paper2.pdf"],
                "summary": "Found 2 relevant papers on RAG."
            }
        )
    )

    # Agent B subscribes and processes events
    async def handle_research_result(event: AgentEvent):
        print(f"Received from {event.source_agent}: "
              f"{event.event_type}")
        print(f"Results: {event.payload['results']}")

    # Run subscriber (in production, this runs in a separate process)
    await bus.subscribe(
        channel="agent-tasks",
        group="writer-agents",
        consumer="writer-agent-01",
        handler=handle_research_result
    )

    await bus.disconnect()
```

The event bus pattern decouples agents from each other. The research agent does not need to know which agent will consume its results. It publishes to a channel, and any interested agent subscribes. This makes it straightforward to add new agents, replace existing ones, or scale consumers independently.

---

## Apache Kafka

Kafka is a distributed event streaming platform designed for high throughput, durability, and replayability. Where Redis Streams excels at real-time inter-agent messaging, Kafka excels at event sourcing, audit trails, and any scenario where you need to replay the complete history of events.

### Fundamentals for Agent Systems

```python
import json
import asyncio
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer


# ── Producer ──

async def produce_agent_events():
    """Publish agent events to Kafka."""
    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )
    await producer.start()

    try:
        # Send an event with a key for partitioning
        await producer.send_and_wait(
            topic="agent-actions",
            key="agent-001",              # Partition key
            value={
                "event_type": "tool_call",
                "agent_id": "agent-001",
                "tool": "search_database",
                "arguments": {"query": "revenue report"},
                "timestamp": "2025-01-15T10:30:00Z"
            }
        )

        await producer.send_and_wait(
            topic="agent-actions",
            key="agent-001",
            value={
                "event_type": "tool_result",
                "agent_id": "agent-001",
                "tool": "search_database",
                "result": {"rows": 42, "status": "success"},
                "duration_ms": 150,
                "timestamp": "2025-01-15T10:30:01Z"
            }
        )

        print("Events published to Kafka")

    finally:
        await producer.stop()


# ── Consumer ──

async def consume_agent_events():
    """Consume and process agent events from Kafka."""
    consumer = AIOKafkaConsumer(
        "agent-actions",
        bootstrap_servers="localhost:9092",
        group_id="monitoring-group",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",  # Start from beginning
        enable_auto_commit=False        # Manual commit for safety
    )
    await consumer.start()

    try:
        async for message in consumer:
            event = message.value
            print(
                f"[partition={message.partition} "
                f"offset={message.offset}] "
                f"{event['event_type']}: "
                f"agent={event['agent_id']}"
            )

            # Process the event
            await process_event(event)

            # Commit offset after successful processing
            await consumer.commit()

    finally:
        await consumer.stop()


async def process_event(event: dict):
    """Process a single agent event."""
    if event["event_type"] == "tool_call":
        print(f"  Tool: {event['tool']}, Args: {event['arguments']}")
    elif event["event_type"] == "tool_result":
        print(
            f"  Result: {event['result']}, "
            f"Duration: {event['duration_ms']}ms"
        )
```

### Event Sourcing for Agents

Event sourcing stores every state change as an immutable event. Instead of persisting the current state of an agent conversation, you store the sequence of events that produced it. This enables complete replay, time-travel debugging, and audit trails.

```python
import json
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer


@dataclass
class AgentStateEvent:
    """A single event in an agent's state history."""
    event_id: str
    session_id: str
    event_type: str       # "message", "tool_call", "tool_result",
                          # "state_change", "error"
    agent_id: str
    data: dict[str, Any]
    timestamp: str | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class AgentEventStore:
    """Kafka-backed event store for agent state.

    Provides:
    - Append-only event storage with guaranteed ordering
    - Full replay to reconstruct agent state at any point
    - Session-based partitioning for parallel processing
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "agent-events"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self._producer: AIOKafkaProducer | None = None

    async def connect(self):
        """Initialize the Kafka producer."""
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )
        await self._producer.start()

    async def disconnect(self):
        """Close the Kafka producer."""
        if self._producer:
            await self._producer.stop()

    async def append(self, event: AgentStateEvent):
        """Append an event to the store.

        Events are partitioned by session_id so all events
        for a session land on the same partition, preserving order.
        """
        await self._producer.send_and_wait(
            topic=self.topic,
            key=event.session_id,
            value=asdict(event)
        )

    async def replay(
        self, session_id: str
    ) -> list[AgentStateEvent]:
        """Replay all events for a session to reconstruct state.

        Creates a temporary consumer, reads all messages for the
        session, and returns them in order.
        """
        consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda v: json.loads(
                v.decode("utf-8")
            ),
            key_deserializer=lambda k: k.decode("utf-8")
            if k else None,
            auto_offset_reset="earliest",
            group_id=None,  # No group — read all messages
            consumer_timeout_ms=5000
        )
        await consumer.start()

        events = []
        try:
            async for message in consumer:
                if message.key == session_id:
                    events.append(
                        AgentStateEvent(**message.value)
                    )
        except asyncio.TimeoutError:
            pass
        finally:
            await consumer.stop()

        return sorted(events, key=lambda e: e.timestamp)


# ── Usage ──

async def main():
    store = AgentEventStore()
    await store.connect()

    session_id = "session_abc_123"

    # Record an agent conversation as events
    await store.append(AgentStateEvent(
        event_id="evt_001",
        session_id=session_id,
        event_type="message",
        agent_id="assistant",
        data={
            "role": "user",
            "content": "What's the revenue for Q4?"
        }
    ))

    await store.append(AgentStateEvent(
        event_id="evt_002",
        session_id=session_id,
        event_type="tool_call",
        agent_id="assistant",
        data={
            "tool": "query_database",
            "arguments": {
                "sql": "SELECT revenue FROM financials "
                       "WHERE quarter = 'Q4'"
            }
        }
    ))

    await store.append(AgentStateEvent(
        event_id="evt_003",
        session_id=session_id,
        event_type="tool_result",
        agent_id="assistant",
        data={
            "tool": "query_database",
            "result": {"revenue": 2_400_000, "currency": "USD"}
        }
    ))

    await store.append(AgentStateEvent(
        event_id="evt_004",
        session_id=session_id,
        event_type="message",
        agent_id="assistant",
        data={
            "role": "assistant",
            "content": "Q4 revenue was $2.4M USD."
        }
    ))

    # Replay events to reconstruct the full conversation
    events = await store.replay(session_id)

    print(f"Replayed {len(events)} events for session "
          f"{session_id}:")
    for event in events:
        print(f"  [{event.event_type}] {event.data}")

    await store.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```

Event sourcing provides capabilities that are impossible with traditional state persistence. You can replay a session to debug unexpected behavior. You can rebuild agent state after a crash without losing any context. You can feed events into an analytics pipeline to understand tool usage patterns across all sessions. And you can implement "time travel" — rolling an agent back to any previous point in its conversation.

---

## Kafka vs Redis Streams Comparison

| Feature | Redis Streams | Apache Kafka |
|---|---|---|
| **Latency** | Sub-millisecond | Single-digit milliseconds |
| **Throughput** | ~100K msg/s per node | >500K msg/s per partition |
| **Durability** | Optional (RDB/AOF) | Built-in replication |
| **Retention** | Memory-bounded (maxlen) | Time/size-based, configurable |
| **Consumer groups** | Yes, built-in | Yes, built-in |
| **Message replay** | Limited (within retention) | Full replay from any offset |
| **Ordering** | Per-stream | Per-partition |
| **Operational overhead** | Low (single Redis instance) | High (ZooKeeper/KRaft + brokers) |
| **Best for agents** | Real-time inter-agent messaging, session state | Event sourcing, audit logs, analytics |

The typical production architecture uses both: Redis Streams for low-latency inter-agent communication (agent A tells agent B to start a task), and Kafka for durable event sourcing (every tool call, every LLM response, every state change is recorded for replay and audit).

---

## Event-Driven Architecture Patterns

### Outbox Pattern

The outbox pattern solves a fundamental problem: how do you atomically update a database and publish an event? If you write to the database and then publish to Kafka, a crash between the two operations means the event is lost. If you publish first and then write, a crash means the database is inconsistent.

The solution is to write the event to an "outbox" table in the same database transaction as the state change. A separate process reads the outbox and publishes to Kafka. Because the state change and the outbox write are in the same transaction, they are atomic.

```python
import json
import asyncio
from datetime import datetime

import asyncpg
from aiokafka import AIOKafkaProducer


async def agent_action_with_outbox(
    pool: asyncpg.Pool,
    action_type: str,
    agent_id: str,
    session_id: str,
    payload: dict
):
    """Execute an agent action and write an event to the outbox
    in a single database transaction.

    This guarantees that the action record and the event are
    either both committed or both rolled back.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. Write the action record
            await conn.execute(
                """
                INSERT INTO agent_actions
                    (agent_id, session_id, action_type, payload,
                     created_at)
                VALUES ($1, $2, $3, $4, NOW())
                """,
                agent_id, session_id, action_type,
                json.dumps(payload)
            )

            # 2. Write to the outbox (same transaction!)
            await conn.execute(
                """
                INSERT INTO event_outbox
                    (aggregate_id, event_type, payload,
                     created_at, published)
                VALUES ($1, $2, $3, NOW(), FALSE)
                """,
                session_id,
                f"agent.{action_type}",
                json.dumps({
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "action_type": action_type,
                    "payload": payload,
                    "timestamp": datetime.utcnow().isoformat()
                })
            )


async def outbox_publisher(
    pool: asyncpg.Pool,
    producer: AIOKafkaProducer,
    topic: str = "agent-events",
    poll_interval: float = 0.5
):
    """Background process that reads the outbox and publishes
    events to Kafka.

    Runs in a loop, polling the outbox table for unpublished
    events. After successful publish, marks them as published.
    """
    while True:
        async with pool.acquire() as conn:
            # Read unpublished events in order
            rows = await conn.fetch(
                """
                SELECT id, aggregate_id, event_type, payload
                FROM event_outbox
                WHERE published = FALSE
                ORDER BY created_at
                LIMIT 100
                """
            )

            for row in rows:
                try:
                    # Publish to Kafka
                    await producer.send_and_wait(
                        topic=topic,
                        key=row["aggregate_id"].encode("utf-8"),
                        value=row["payload"].encode("utf-8")
                    )

                    # Mark as published
                    await conn.execute(
                        """
                        UPDATE event_outbox
                        SET published = TRUE
                        WHERE id = $1
                        """,
                        row["id"]
                    )

                except Exception as e:
                    print(
                        f"Failed to publish event {row['id']}: {e}"
                    )
                    # Will be retried on next poll iteration
                    break

        await asyncio.sleep(poll_interval)


# ── SQL schema for the outbox ──
OUTBOX_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_outbox (
    id              BIGSERIAL PRIMARY KEY,
    aggregate_id    TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published       BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX idx_outbox_unpublished
    ON event_outbox (created_at)
    WHERE published = FALSE;

CREATE TABLE IF NOT EXISTS agent_actions (
    id              BIGSERIAL PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    action_type     TEXT NOT NULL,
    payload         JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""
```

The outbox pattern is particularly important for agent systems because agents produce events that downstream systems depend on — analytics, billing, compliance logging. If an agent records a tool call in its database but fails to publish the corresponding event, the audit trail has a gap. The outbox eliminates this class of inconsistency.

---

## Key Insights

> **Temporal Durability:** Event sourcing records every workflow step. On failure, new worker replays events to restore exact state. Workflow code must be deterministic.

> **Kafka vs Redis Streams for Agents:** Redis Streams: real-time inter-agent communication, sub-ms latency. Kafka: audit logging, event sourcing for replay, high throughput >50K msg/s, long-term event storage. Typical setup: Redis Streams for inter-agent comms + Kafka for audit trail.

---

## References

- Temporal Python SDK: [https://docs.temporal.io/develop/python](https://docs.temporal.io/develop/python)
- Redis Streams: [https://redis.io/docs/data-types/streams/](https://redis.io/docs/data-types/streams/)
- Kafka Python: [https://aiokafka.readthedocs.io/](https://aiokafka.readthedocs.io/)
- Event Sourcing: [https://microservices.io/patterns/data/event-sourcing.html](https://microservices.io/patterns/data/event-sourcing.html)
