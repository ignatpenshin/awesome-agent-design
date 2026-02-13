# Chapter 2: Agent Frameworks

> *"A framework is not just a library — it is an opinion about how software should be built."*

---

Building multi-agent systems from scratch is possible but rarely practical. The orchestration patterns described in Chapter 1 — sequential pipelines, hierarchical delegation, consensus mechanisms — all demand careful management of state, routing, error handling, and agent lifecycle. Agent frameworks encode these patterns into reusable abstractions, letting you focus on the logic of your system rather than the plumbing.

This chapter provides a deep technical examination of four major frameworks: **LangGraph**, **CrewAI**, **AutoGen**, and **Semantic Kernel**. Each embodies a fundamentally different philosophy about how agents should be composed and coordinated. LangGraph thinks in graphs. CrewAI thinks in roles. AutoGen thinks in conversations. Semantic Kernel thinks in plugins. Understanding these paradigms — and knowing when to apply each — is essential for any architect building production agent systems.

We begin with LangGraph, the most flexible and production-oriented of the four, and devote the majority of this chapter to its internals.

---

## LangGraph

LangGraph is a low-level orchestration framework built on top of LangChain that models agent workflows as **directed graphs**. Every workflow is a graph of nodes (computation steps) connected by edges (transitions). State flows through the graph, being read and written by each node, with the framework managing persistence, checkpointing, and control flow.

This graph-based model gives you complete control over execution flow. Unlike higher-level frameworks that impose a fixed orchestration pattern, LangGraph lets you define arbitrary topologies — linear pipelines, fan-out/fan-in parallelism, cycles for iterative refinement, conditional branching based on runtime state — all expressed as nodes and edges.

### Core Concepts

LangGraph is built on three primitives:

1. **State** — A shared data structure (typically a `TypedDict`) that flows through the graph. Every node reads from and writes to this state.
2. **Nodes** — Python functions that receive the current state, perform computation, and return state updates.
3. **Edges** — Connections between nodes that define execution order. Edges can be unconditional (always follow) or conditional (route based on state).

The `StateGraph` class ties these together:

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(StateSchema)
graph.add_node("node_name", node_function)
graph.add_edge("node_a", "node_b")
graph.add_conditional_edges("node_a", routing_function, {"route1": "node_b", "route2": "node_c"})
```

### State Schema and Reducers

State schemas are defined using Python's `TypedDict`. Each field in the schema represents a piece of data that nodes can read and write. Here is where LangGraph introduces a critical concept: **reducers**.

By default, when a node returns a value for a state field, it **overwrites** the previous value. But for fields like message lists or accumulated results, you want new values to be **appended** rather than replaced. This is accomplished with `Annotated` types and reducer functions:

```python
import operator
from typing import Annotated, TypedDict

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_agent: str
    final_report: str
```

The declaration `Annotated[list, operator.add]` tells LangGraph that when a node returns a value for `messages`, it should be **concatenated** with the existing list using Python's `operator.add` (i.e., list concatenation), not replace it. This is essential for message accumulation patterns where multiple nodes contribute to a growing conversation history.

In contrast, `current_agent` and `final_report` are plain types with no reducer, so returning a value for these fields **overwrites** the previous value entirely.

This distinction matters enormously in practice. If you forget the `Annotated[list, operator.add]` wrapper on a message list, each node will overwrite the entire conversation history with its own messages — a subtle bug that can be difficult to diagnose.

### Full Example: Multi-Agent Supervisor System

Let us build a complete multi-agent system with a **supervisor** that routes tasks between a **researcher** and a **writer**. This is one of the most common production patterns: a coordinating agent that delegates to specialists.

#### Step 1: Define the State

```python
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage

class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    current_agent: str
    research_data: str
    final_report: str
    iteration_count: int
```

The `messages` field uses the append reducer so that every agent's messages accumulate into a shared conversation history. The remaining fields are overwritten on each update.

#### Step 2: Define the Agent Functions

Each agent is a node function that receives the full state and returns a partial state update:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def supervisor(state: SupervisorState) -> dict:
    """Supervisor decides which agent to call next or whether to finish."""
    system_prompt = """You are a supervisor coordinating a research team.
    Based on the current state of the task, decide the next step:
    - "researcher": if more information is needed
    - "writer": if enough research is gathered and a report should be written
    - "FINISH": if the final report is complete

    Respond with ONLY one of: researcher, writer, FINISH"""

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)
    next_agent = response.content.strip()

    return {
        "current_agent": next_agent,
        "messages": [AIMessage(content=f"Supervisor decision: route to {next_agent}")]
    }


def researcher(state: SupervisorState) -> dict:
    """Research agent gathers and analyzes information."""
    system_prompt = """You are a research analyst. Analyze the given topic
    and provide structured, factual information with key findings.
    Be thorough but concise."""

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    return {
        "research_data": response.content,
        "messages": [AIMessage(content=f"Researcher: {response.content}")],
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def writer(state: SupervisorState) -> dict:
    """Writer agent composes the final report from research data."""
    system_prompt = f"""You are a technical writer. Using the research data below,
    compose a well-structured report.

    Research data:
    {state.get('research_data', 'No research data available.')}"""

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    return {
        "final_report": response.content,
        "messages": [AIMessage(content=f"Writer: {response.content}")]
    }
```

Notice that each function returns a dictionary with only the fields it wants to update. For the `messages` field, the returned list will be **appended** to the existing messages (due to the `operator.add` reducer). For all other fields, the returned values **replace** the current values.

#### Step 3: Define the Routing Function

The routing function inspects state and returns the name of the next node:

```python
def route_supervisor(state: SupervisorState) -> str:
    """Route based on the supervisor's decision."""
    next_agent = state.get("current_agent", "")

    if next_agent == "FINISH":
        return "end"
    elif next_agent == "researcher":
        return "researcher"
    elif next_agent == "writer":
        return "writer"
    else:
        return "end"  # Safety fallback
```

#### Step 4: Assemble and Compile the Graph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Build the graph
workflow = StateGraph(SupervisorState)

# Add nodes
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)

# Define edges
workflow.add_edge(START, "supervisor")

# Conditional routing from supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "writer": "writer",
        "end": END
    }
)

# After researcher or writer, always return to supervisor
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", "supervisor")

# Compile with checkpointing
checkpointer = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=checkpointer)
```

The graph topology is: `START -> supervisor -> (researcher | writer | END)`, with researcher and writer always looping back to the supervisor. The supervisor acts as a central hub, deciding after each step whether to continue or terminate.

#### Step 5: Execute the Graph

```python
# Run the agent system
config = {"configurable": {"thread_id": "research-task-001"}}

initial_input = {
    "messages": [HumanMessage(content="Analyze the current state of quantum computing and its commercial applications")],
    "iteration_count": 0
}

# Stream execution to observe each step
for event in app.stream(initial_input, config=config):
    for node_name, output in event.items():
        print(f"\n{'='*50}")
        print(f"Node: {node_name}")
        if "messages" in output:
            for msg in output["messages"]:
                print(f"  {msg.content[:200]}...")
```

The `thread_id` in the config is critical — it ties this execution to a specific conversation thread, enabling the checkpointer to save and restore state. If the process crashes mid-execution, you can resume from the last completed node simply by invoking `app.stream()` or `app.invoke()` with the same `thread_id`.

### Checkpointing and Persistence

Checkpointing is one of LangGraph's most important production features. After every node execution, the checkpointer **serializes the complete graph state** and stores it in a persistent backend. This unlocks three capabilities:

**1. Recovery after failures.** If a node fails (network timeout, API rate limit, out-of-memory), you can resume execution from the last successful checkpoint rather than restarting the entire workflow. For long-running agent pipelines that make dozens of LLM calls, this can save significant time and cost.

**2. Human-in-the-loop workflows.** By pausing execution at a checkpoint, you can present intermediate results to a human for review and approval before continuing. The graph waits indefinitely at the interrupt point — minutes, hours, or days — and resumes exactly where it left off.

**3. Time travel.** Because every intermediate state is persisted, you can "rewind" to any previous step and replay execution from that point with modified inputs. This is invaluable for debugging complex multi-agent interactions.

LangGraph supports two checkpointing backends:

```python
# SQLite — suitable for development and testing
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# PostgreSQL — suitable for production deployments
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost:5432/agents"
)

# Compile the graph with the chosen checkpointer
app = workflow.compile(checkpointer=checkpointer)
```

Use SQLite during development for zero-configuration persistence. Switch to PostgreSQL for production systems where you need concurrent access, durability guarantees, and integration with existing infrastructure.

### Human-in-the-Loop with `interrupt_before`

The `interrupt_before` parameter tells LangGraph to pause execution **before** a specified node runs, giving a human the opportunity to review the current state and decide whether to proceed:

```python
# Compile with an interrupt point before the writer node
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["writer"]
)

config = {"configurable": {"thread_id": "review-task-001"}}

# First run — executes until it reaches the writer node, then pauses
for event in app.stream(
    {"messages": [HumanMessage(content="Write a market analysis report")]},
    config=config
):
    print(event)

# At this point, execution is paused before the writer node.
# A human can inspect the current state:
current_state = app.get_state(config)
print("Research data so far:", current_state.values.get("research_data"))

# Human reviews and decides to continue.
# Passing None resumes from the interrupt point:
for event in app.stream(None, config=config):
    print(event)
```

The key detail is that calling `app.stream(None, config=config)` with the same `thread_id` resumes execution from exactly where it was interrupted. The `None` input signals "continue with the existing state" rather than starting a new execution.

You can also use `interrupt_after` to pause **after** a node completes, which is useful when you want to review a node's output before the next node begins.

### Subgraphs: Modular Composition

As agent systems grow in complexity, a single flat graph becomes difficult to maintain. LangGraph supports **subgraphs** — self-contained graphs that are embedded as nodes within a parent graph:

```python
from langgraph.graph import StateGraph, START, END

# Define a research subgraph
class ResearchState(TypedDict):
    query: str
    sources: Annotated[list, operator.add]
    summary: str

def search_web(state: ResearchState) -> dict:
    """Search the web for information."""
    # ... web search implementation ...
    return {"sources": [{"url": "...", "content": "..."}]}

def analyze_sources(state: ResearchState) -> dict:
    """Analyze and summarize collected sources."""
    # ... LLM analysis implementation ...
    return {"summary": "Synthesized findings..."}

research_graph = StateGraph(ResearchState)
research_graph.add_node("search", search_web)
research_graph.add_node("analyze", analyze_sources)
research_graph.add_edge(START, "search")
research_graph.add_edge("search", "analyze")
research_graph.add_edge("analyze", END)
research_subgraph = research_graph.compile()


# Embed the subgraph in a parent graph
class ParentState(TypedDict):
    messages: Annotated[list, operator.add]
    research_summary: str
    final_output: str

def prepare_research(state: ParentState) -> dict:
    """Extract the research query from the conversation."""
    return {"query": state["messages"][-1].content}

def format_output(state: ParentState) -> dict:
    """Format the final output from research results."""
    return {"final_output": f"Report based on: {state.get('research_summary', '')}"}

parent_graph = StateGraph(ParentState)
parent_graph.add_node("prepare", prepare_research)
parent_graph.add_node("research", research_subgraph)  # Subgraph as a node
parent_graph.add_node("format", format_output)
parent_graph.add_edge(START, "prepare")
parent_graph.add_edge("prepare", "research")
parent_graph.add_edge("research", "format")
parent_graph.add_edge("format", END)

app = parent_graph.compile()
```

Subgraphs enable **modular composition**: you can develop, test, and version each subgraph independently, then combine them into larger systems. A research subgraph, a writing subgraph, and a review subgraph can each be maintained by different teams and composed into various parent workflows.

---

## CrewAI

CrewAI takes a fundamentally different approach from LangGraph. Instead of thinking in graphs and state transitions, CrewAI models agent systems as **crews** — teams of agents with defined roles, goals, and backstories who collaborate on tasks.

This role-based paradigm maps naturally to how human teams work: a researcher gathers information, a writer drafts content, an editor polishes it. Each agent has a clear identity and purpose, and the framework handles the coordination between them.

### Core Concepts

CrewAI is built around four abstractions:

1. **Agent** — An autonomous entity with a `role`, `goal`, and `backstory`. The backstory provides persona context that shapes the agent's behavior. Agents can be given tools and the ability to delegate work to other agents.
2. **Task** — A unit of work assigned to a specific agent. Tasks have a `description`, an `expected_output` specification, and optional `context` dependencies on other tasks.
3. **Tool** — External capabilities that agents can invoke (web search, file I/O, API calls, etc.).
4. **Crew** — The orchestration layer that assembles agents and tasks, defines the execution process (sequential or hierarchical), and manages the overall workflow.

### Full Example: Content Production Crew

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# --- Define Tools ---
search_tool = SerperDevTool()
web_tool = WebsiteSearchTool()

# --- Define Agents ---
researcher = Agent(
    role="Senior Research Analyst",
    goal="Conduct thorough research and provide comprehensive, factual analysis",
    backstory="""You are an experienced research analyst with 15 years
    of experience in technology market analysis. You are meticulous about
    facts and always verify information from multiple sources. You have
    a talent for identifying key trends and their implications.""",
    tools=[search_tool, web_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False  # This agent works independently
)

writer = Agent(
    role="Technical Content Writer",
    goal="Transform research findings into clear, engaging, well-structured articles",
    backstory="""You are a skilled technical writer who excels at making
    complex topics accessible. You have written for major technology
    publications and have a knack for finding the narrative thread
    in technical material. You always structure content with clear
    headings and logical flow.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

editor = Agent(
    role="Senior Editor",
    goal="Ensure content is polished, accurate, and publication-ready",
    backstory="""You are a seasoned editor with expertise in technical
    content. You have a sharp eye for logical inconsistencies, unclear
    phrasing, and factual errors. You improve content while preserving
    the author's voice. You verify that all claims are supported
    by the research.""",
    llm=llm,
    verbose=True,
    allow_delegation=True  # Can request rewrites from the writer
)

# --- Define Tasks ---
research_task = Task(
    description="""Research the current state of large language model agents
    in enterprise applications. Focus on:
    1. Key adoption trends and market size
    2. Common architectural patterns
    3. Major challenges and limitations
    4. Notable case studies and implementations
    Provide detailed findings with specific data points.""",
    expected_output="A comprehensive research brief with data, trends, and analysis",
    agent=researcher
)

writing_task = Task(
    description="""Using the research findings, write a 1500-word article titled
    'The Rise of LLM Agents in the Enterprise'. The article should be
    informative, well-structured, and accessible to a technical audience.
    Include an introduction, 3-4 main sections, and a conclusion.""",
    expected_output="A polished, well-structured article of approximately 1500 words",
    agent=writer,
    context=[research_task]  # Writer receives researcher's output
)

editing_task = Task(
    description="""Review and edit the article for:
    1. Factual accuracy against the original research
    2. Clarity and readability
    3. Logical flow and structure
    4. Grammar and style consistency
    Provide the final, publication-ready version.""",
    expected_output="The final edited article ready for publication",
    agent=editor,
    context=[research_task, writing_task]  # Editor sees both research and draft
)

# --- Assemble and Run the Crew ---
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,  # Tasks execute in order
    verbose=True
)

result = crew.kickoff()
print(result)
```

The `context` parameter on tasks is how CrewAI handles data flow between agents. When `writing_task` lists `research_task` in its context, the writer automatically receives the researcher's output as part of its input. This is simpler than LangGraph's explicit state management but less flexible — you can only pass entire task outputs, not individual state fields.

### Hierarchical Process

For more complex coordination, CrewAI supports a **hierarchical process** where a manager agent automatically delegates tasks to team members:

```python
from crewai import Crew, Process
from langchain_openai import ChatOpenAI

manager_llm = ChatOpenAI(model="gpt-4o", temperature=0)

hierarchical_crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.hierarchical,
    manager_llm=manager_llm,  # LLM that powers the auto-generated manager
    verbose=True
)

result = hierarchical_crew.kickoff()
```

In hierarchical mode, CrewAI automatically creates a manager agent powered by the specified LLM. This manager decides which agent to assign each task to, reviews intermediate outputs, and can request revisions — similar to the supervisor pattern we implemented manually in LangGraph, but with zero configuration.

The tradeoff is control: you cannot customize the manager's decision logic, its prompts, or the routing criteria. For production systems where you need deterministic behavior and fine-grained control over orchestration, LangGraph's explicit graph definition is preferable. For rapid prototyping and scenarios where the default behavior is sufficient, CrewAI's hierarchical mode can save significant development time.

---

## AutoGen

AutoGen, developed by Microsoft Research, models multi-agent systems as **conversations**. Instead of graphs or role-based crews, AutoGen agents communicate by exchanging messages in a shared chat. The framework provides agent types, conversation patterns, and — critically — a built-in code execution sandbox.

### Core Concepts

AutoGen's agent model is built around:

1. **AssistantAgent** — An LLM-powered agent that generates responses, writes code, and reasons about tasks.
2. **UserProxyAgent** — Represents a human user or acts as an automated proxy that can execute code and provide feedback.
3. **GroupChat** — A conversation container that holds multiple agents and manages turn-taking.
4. **GroupChatManager** — Coordinates a `GroupChat`, deciding which agent speaks next based on the configured selection method.

The fundamental interaction model is simple: agents send messages to each other (or to a group), and each agent responds based on its system prompt, the conversation history, and any tools it has available.

### Full Example: Collaborative Code Generation

```python
import autogen

# Configuration for the LLM
config_list = [
    {
        "model": "gpt-4o",
        "api_key": "your-api-key"
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0,
    "timeout": 120
}

# --- Define Agents ---

# Architect agent — designs the solution
architect = autogen.AssistantAgent(
    name="Architect",
    system_message="""You are a software architect. Your role is to:
    1. Analyze requirements and design solutions
    2. Break down complex tasks into implementation steps
    3. Define interfaces and data structures
    4. Review code for architectural consistency
    Do NOT write implementation code — focus on design and review.""",
    llm_config=llm_config
)

# Programmer agent — writes the code
programmer = autogen.AssistantAgent(
    name="Programmer",
    system_message="""You are an expert Python programmer. Your role is to:
    1. Implement solutions based on the architect's design
    2. Write clean, well-documented, production-quality code
    3. Include error handling and type hints
    4. Write code in ```python blocks so it can be executed""",
    llm_config=llm_config
)

# Reviewer agent — reviews code quality
reviewer = autogen.AssistantAgent(
    name="Reviewer",
    system_message="""You are a senior code reviewer. Your role is to:
    1. Review code for bugs, edge cases, and security issues
    2. Check code style and best practices
    3. Suggest improvements and optimizations
    4. Verify the implementation matches the architectural design
    Be specific and constructive in your feedback.""",
    llm_config=llm_config
)

# User proxy — executes code and provides human oversight
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="TERMINATE",  # Only ask human at the end
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": True  # Sandboxed execution via Docker
    }
)

# --- Set Up Group Chat ---
group_chat = autogen.GroupChat(
    agents=[user_proxy, architect, programmer, reviewer],
    messages=[],
    max_round=20,
    speaker_selection_method="auto"  # LLM decides who speaks next
)

manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config
)

# --- Run the Conversation ---
user_proxy.initiate_chat(
    manager,
    message="""Build a Python REST API for a task management system with:
    1. CRUD operations for tasks (title, description, status, priority)
    2. SQLite database backend
    3. Input validation
    4. Error handling
    5. Unit tests"""
)
```

### GroupChat Speaker Selection

The `speaker_selection_method` parameter controls how the `GroupChatManager` decides which agent speaks next in each round:

- **`"auto"`** — The LLM examines the conversation history and selects the most appropriate next speaker. This is the most flexible option but adds an extra LLM call per turn.
- **`"round_robin"`** — Agents speak in a fixed rotation. Simple and deterministic, but does not adapt to conversation dynamics.
- **`"manual"`** — A custom Python function determines the next speaker based on the current state of the conversation.

```python
# Manual speaker selection with a custom function
def custom_speaker_selection(last_speaker, group_chat):
    """Determine the next speaker based on conversation state."""
    messages = group_chat.messages

    if last_speaker.name == "Architect":
        return programmer  # After design, implement
    elif last_speaker.name == "Programmer":
        return reviewer    # After implementation, review
    elif last_speaker.name == "Reviewer":
        # If review found issues, send back to programmer
        last_message = messages[-1]["content"] if messages else ""
        if "APPROVED" in last_message:
            return user_proxy  # Done — return to user
        else:
            return programmer  # Fix the issues
    else:
        return architect  # Start with design

group_chat = autogen.GroupChat(
    agents=[user_proxy, architect, programmer, reviewer],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection
)
```

### Code Execution Sandbox

One of AutoGen's distinguishing features is its built-in code execution environment. When an agent generates a code block (in markdown triple-backtick format), the `UserProxyAgent` can automatically extract and execute it. With `use_docker=True`, execution happens inside a Docker container, providing sandboxed isolation:

```python
user_proxy = autogen.UserProxyAgent(
    name="CodeExecutor",
    code_execution_config={
        "work_dir": "workspace",      # Working directory for code files
        "use_docker": "python:3.11",  # Specific Docker image
        "timeout": 60                  # Execution timeout in seconds
    },
    human_input_mode="NEVER"  # Fully automated
)
```

This makes AutoGen particularly well-suited for **code generation and execution tasks** — data analysis pipelines, automated testing, code refactoring — where agents need to write code, run it, observe the output, and iterate.

---

## Semantic Kernel

Semantic Kernel is Microsoft's SDK for integrating LLMs into applications. Unlike the other frameworks in this chapter, Semantic Kernel was not designed specifically for multi-agent systems. Instead, it provides a **plugin-based architecture** where LLM capabilities are composed from modular functions and orchestrated by planners.

Its primary strength is **enterprise integration**, particularly with the Azure ecosystem. If your organization is already invested in Azure OpenAI, Azure Cognitive Search, and the broader Microsoft stack, Semantic Kernel provides first-class support for these services.

### Core Concepts

Semantic Kernel is built around:

1. **Kernel** — The central orchestration object that manages plugins, memory, and LLM connections.
2. **Plugins** — Collections of functions that extend the kernel's capabilities. Plugins contain two types of functions:
   - **Native Functions** — Standard Python functions decorated with `@kernel_function`.
   - **Prompt Functions** — LLM-powered functions defined by prompt templates.
3. **Planner** — An LLM-powered component that automatically decomposes a goal into a sequence of plugin function calls.
4. **Memory** — A built-in abstraction for semantic memory backed by vector stores.

### Code Example: Plugin-Based Agent

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.core_plugins import TextPlugin
from semantic_kernel.functions import kernel_function

# Initialize the kernel
kernel = sk.Kernel()

# Add an Azure OpenAI service
kernel.add_service(
    AzureChatCompletion(
        deployment_name="gpt-4o",
        endpoint="https://your-resource.openai.azure.com/",
        api_key="your-api-key"
    )
)

# --- Define a Native Plugin ---
class ResearchPlugin:
    """Plugin for research operations."""

    @kernel_function(
        name="search_documents",
        description="Search internal documents for information on a topic"
    )
    def search_documents(self, query: str) -> str:
        """Search the document database and return relevant results."""
        # Implementation: query a vector store, database, or search API
        results = perform_document_search(query)
        return format_results(results)

    @kernel_function(
        name="summarize_findings",
        description="Summarize a collection of research findings into key points"
    )
    def summarize_findings(self, findings: str) -> str:
        """Summarize research findings."""
        # Implementation: process and condense findings
        return create_summary(findings)

class WritingPlugin:
    """Plugin for content generation."""

    @kernel_function(
        name="draft_report",
        description="Draft a structured report from research summary and topic"
    )
    def draft_report(self, summary: str, topic: str) -> str:
        """Generate a report draft based on summarized research."""
        # Implementation: use templates and LLM to draft content
        return generate_report(summary, topic)

# Register plugins with the kernel
kernel.add_plugin(ResearchPlugin(), plugin_name="Research")
kernel.add_plugin(WritingPlugin(), plugin_name="Writing")

# --- Define a Prompt Function ---
# Prompt functions are LLM-powered and defined via templates
report_review_function = kernel.add_function(
    plugin_name="Review",
    function_name="review_report",
    prompt="""Review the following report for accuracy, clarity, and completeness.
    Provide specific feedback and a quality score from 1-10.

    Report:
    {{$input}}

    Review:""",
    description="Review a report for quality and accuracy"
)

# --- Use the Sequential Planner ---
from semantic_kernel.planners import SequentialPlanner

planner = SequentialPlanner(kernel)

# The planner automatically decomposes the goal into plugin function calls
plan = await planner.create_plan(
    goal="Research the impact of LLM agents on software development, "
         "write a comprehensive report, and review it for quality."
)

print("Generated plan:")
for step in plan.steps:
    print(f"  - {step.plugin_name}.{step.name}: {step.description}")

# Execute the plan
result = await plan.invoke(kernel)
print(result)
```

The `SequentialPlanner` analyzes the available plugins and their function descriptions, then constructs a plan — an ordered sequence of function calls — that achieves the stated goal. This is a powerful abstraction for building goal-oriented systems, but it depends heavily on the quality of function descriptions and the LLM's planning ability.

### When to Choose Semantic Kernel

Semantic Kernel is the right choice when:

- Your organization is invested in the **Azure ecosystem** and needs native Azure OpenAI, Azure Cognitive Search, and Azure AI integration.
- You want a **plugin-based architecture** where capabilities are modular and independently deployable.
- You need enterprise features like **role-based access control**, **audit logging**, and compliance out of the box.
- Your use case fits the **planner paradigm** — decomposing goals into sequences of well-defined function calls.

It is less suitable for complex multi-agent orchestration with custom control flow, cycles, or sophisticated state management — use LangGraph for those scenarios.

---

## Framework Comparison

The following table summarizes the key differences between the four frameworks:

| Criterion | LangGraph | CrewAI | AutoGen | Semantic Kernel |
|---|---|---|---|---|
| **Core model** | Graph (nodes/edges) | Role-based crews | Conversation-based | Plugin + Planner |
| **Flexibility** | Very high | Medium | Medium | Medium |
| **State management** | Built-in, persistent | Limited | Via chat history | Via memory |
| **Human-in-the-loop** | `interrupt_before`/`after` | Limited | `human_input_mode` | Via hooks |
| **Production-ready** | Yes (LangSmith) | Partially | Partially | Yes (Azure) |
| **Learning curve** | High | Low | Medium | Medium |
| **Code execution** | Via tools | Via tools | Built-in sandbox | Via plugins |
| **Best use case** | Complex orchestration | Quick prototypes | Code generation | Enterprise/Azure |

### Decision Guide

Choosing the right framework depends on your requirements, timeline, and operational context:

**Choose LangGraph when** you need complex control flow with cycles, conditional branching, and fan-out/fan-in parallelism. Choose it when you require persistent state with checkpointing — for recovery, human-in-the-loop review, or time-travel debugging. Choose it when you are building a production system that demands full observability (via LangSmith) and fine-grained control over every aspect of the orchestration. LangGraph has the steepest learning curve but offers the most flexibility and the strongest production story.

**Choose CrewAI when** you want to prototype multi-agent systems rapidly with minimal boilerplate. The role-based paradigm (role, goal, backstory) is intuitive and maps naturally to team-based workflows. CrewAI is excellent for simple sequential or hierarchical pipelines where default orchestration behavior is sufficient. It is less suitable for systems that require custom control flow, persistent state, or fine-grained routing logic.

**Choose AutoGen when** your primary use case involves code generation, execution, and iterative refinement. AutoGen's built-in code execution sandbox (with Docker isolation) and conversation-based architecture make it ideal for research prototyping, data analysis automation, and any scenario where agents need to write, run, and debug code collaboratively. Its GroupChat abstraction is also well-suited for brainstorming and deliberation patterns.

**Choose Semantic Kernel when** you are building within the Microsoft/Azure ecosystem and need native integration with Azure OpenAI, Azure Cognitive Search, and enterprise infrastructure. The plugin-based architecture is a good fit for organizations that want modular, independently deployable capabilities. Semantic Kernel is the most enterprise-oriented option, with strong support for compliance, access control, and audit requirements.

---

## Key Insights

> **LangGraph vs. CrewAI:** LangGraph is a low-level graph framework that gives you full control over execution flow and state management. CrewAI is a high-level, role-based framework designed for rapid development. Choose LangGraph when you need custom control flow, persistent state, human-in-the-loop, and production-grade observability. Choose CrewAI when you want to prototype quickly with a sequential or hierarchical pipeline and the default orchestration is sufficient. The two frameworks occupy different points on the abstraction spectrum — LangGraph is closer to building your own orchestration engine, while CrewAI provides an opinionated, batteries-included experience.

> **LangGraph Checkpointing:** The checkpointer serializes the complete graph state after every node execution. State is stored in SQLite (development) or PostgreSQL (production). This enables three critical capabilities: (1) **recovery after failures** — resume from the last successful node instead of restarting the entire workflow; (2) **human-in-the-loop** — pause execution via `interrupt_before` or `interrupt_after`, present intermediate results for human review, and resume with `app.stream(None, config)`; (3) **time travel** — rewind to any previous step and replay execution with modified inputs. Checkpointing transforms LangGraph from a simple workflow engine into a durable, resumable orchestration platform.

> **AutoGen GroupChat:** The `GroupChatManager` coordinates multi-agent conversations by selecting which agent speaks next. The `speaker_selection_method` parameter controls this selection: `"auto"` uses an LLM call to choose the most appropriate speaker, `"round_robin"` cycles through agents in fixed order, and `"manual"` delegates to a custom Python function. Every agent in the group sees the full conversation history, enabling context-aware responses. The GroupChat pattern is particularly effective for collaborative problem-solving where multiple perspectives need to converge on a solution.

---

## References

- LangGraph Documentation: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- CrewAI Documentation: [https://docs.crewai.com/](https://docs.crewai.com/)
- AutoGen Documentation: [https://microsoft.github.io/autogen/](https://microsoft.github.io/autogen/)
- Semantic Kernel Documentation: [https://learn.microsoft.com/en-us/semantic-kernel/](https://learn.microsoft.com/en-us/semantic-kernel/)
- LangChain Hub: [https://smith.langchain.com/hub](https://smith.langchain.com/hub)
- AutoGen Research Paper: Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation," 2023.
