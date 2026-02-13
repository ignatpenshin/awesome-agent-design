# Chapter 5: Tool & Function Calling

> *"A tool is only as good as the hand that wields it — and the instructions that describe it."*

---

Tools transform LLMs from passive text generators into active agents that can query databases, call APIs, execute code, and interact with the physical world. Function calling is the protocol that makes this possible: a structured contract between the model and external systems that turns natural language intent into executable actions.

This chapter covers the full lifecycle of tool integration — from the raw protocol mechanics of function calling, through description engineering and dynamic selection at scale, to security hardening and resilient execution in production.

---

## Function Calling Protocol

### How It Works

Function calling is not magic. It is a structured protocol where the model outputs a JSON object specifying which function to call and with what arguments, rather than generating free-form text. The application then executes the function and feeds the result back to the model for interpretation.

Here is the complete five-step flow using the OpenAI API:

```python
import openai
import json

client = openai.OpenAI()

# Step 1: Define tools with JSON Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specific city. "
                           "Returns temperature, humidity, and conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search a product database by query string. "
                           "Returns matching products with name, price, "
                           "and availability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for product lookup"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "food", "all"],
                        "description": "Product category filter"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Step 2: Send the user message with tool definitions
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    tools=tools,
    tool_choice="auto"  # Model decides whether to call a tool
)

message = response.choices[0].message

# Step 3: Check if the model wants to call a tool
if message.tool_calls:
    tool_call = message.tool_calls[0]

    function_name = tool_call.function.name   # "get_weather"
    arguments = json.loads(tool_call.function.arguments)  # {"city": "Tokyo"}

    # Step 4: Execute the function locally
    def get_weather(city: str, unit: str = "celsius") -> dict:
        # In production, this calls a real weather API
        return {
            "city": city,
            "temperature": 22,
            "unit": unit,
            "conditions": "partly cloudy",
            "humidity": 65
        }

    result = get_weather(**arguments)

    # Step 5: Send the result back for the model to interpret
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
            message,  # The assistant's tool_call message
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            }
        ],
        tools=tools
    )

    # Model generates a natural language response:
    # "The current weather in Tokyo is 22°C with partly cloudy
    #  skies and 65% humidity."
    print(final_response.choices[0].message.content)
```

The critical insight is that **the model never executes anything**. It produces structured JSON expressing intent; the application owns execution. This separation of concerns is what makes function calling safe and controllable.

### Parallel Tool Calling

Modern models can request multiple tool calls in a single response. When a user asks a question that requires data from several independent sources, the model emits multiple tool calls simultaneously, and the application can execute them in parallel:

```python
import asyncio
import json
import openai

client = openai.OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Compare the weather in Tokyo, London, and New York"
        }
    ],
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message

# The model returns MULTIPLE tool_calls in a single response
# message.tool_calls = [
#     ToolCall(id="call_1", function=Function(name="get_weather",
#              arguments='{"city": "Tokyo"}')),
#     ToolCall(id="call_2", function=Function(name="get_weather",
#              arguments='{"city": "London"}')),
#     ToolCall(id="call_3", function=Function(name="get_weather",
#              arguments='{"city": "New York"}')),
# ]

async def execute_weather(city: str) -> dict:
    """Simulate async weather API call."""
    await asyncio.sleep(0.1)  # Simulate network latency
    weather_data = {
        "Tokyo": {"temp": 22, "conditions": "partly cloudy"},
        "London": {"temp": 15, "conditions": "rainy"},
        "New York": {"temp": 28, "conditions": "sunny"},
    }
    return weather_data.get(city, {"temp": 0, "conditions": "unknown"})

async def execute_all_tool_calls(tool_calls):
    """Execute all tool calls in parallel."""
    tasks = []
    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        tasks.append(execute_weather(args["city"]))

    results = await asyncio.gather(*tasks)

    # Build tool response messages
    tool_messages = []
    for tc, result in zip(tool_calls, results):
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result)
        })
    return tool_messages

# Execute in parallel and get final response
tool_messages = asyncio.run(
    execute_all_tool_calls(message.tool_calls)
)

final_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user",
         "content": "Compare the weather in Tokyo, London, and New York"},
        message,
        *tool_messages  # All tool results at once
    ],
    tools=tools
)

print(final_response.choices[0].message.content)
```

Parallel tool calling reduces latency from `N * avg_latency` to `max(latencies)`. For three weather API calls at 200ms each, that is 200ms instead of 600ms. In agent loops that make dozens of tool calls, this optimization is significant.

---

## Tool Description Engineering

The quality of your tool descriptions directly determines whether the model selects the right tool and constructs correct arguments. A vague description forces the model to guess; a precise one eliminates ambiguity.

**Bad descriptions:**

```python
# Vague — model doesn't know when to use it
{
    "name": "search",
    "description": "Search for stuff",
    "parameters": {
        "type": "object",
        "properties": {
            "q": {"type": "string"}
        }
    }
}

# Ambiguous — what kind of data? What format?
{
    "name": "get_data",
    "description": "Gets data from the system",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string"}
        }
    }
}
```

**Good descriptions:**

```python
# Clear purpose, parameter semantics, return value
{
    "name": "search_knowledge_base",
    "description": (
        "Search the internal knowledge base for documentation articles. "
        "Use this when the user asks about product features, "
        "troubleshooting steps, or company policies. "
        "Returns a list of relevant articles ranked by relevance score. "
        "Each result includes title, snippet, and URL."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language search query. Be specific — "
                    "'how to reset password' works better than 'password'."
                )
            },
            "category": {
                "type": "string",
                "enum": ["product", "billing", "technical", "policy"],
                "description": (
                    "Filter by article category. Use 'technical' for "
                    "how-to guides and troubleshooting, 'billing' for "
                    "payment and subscription questions."
                )
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1-20)",
                "default": 5,
                "minimum": 1,
                "maximum": 20
            }
        },
        "required": ["query"]
    }
}
```

The principles behind effective tool descriptions:

1. **State the purpose explicitly.** "Search the internal knowledge base for documentation articles" tells the model exactly what this tool accesses.

2. **Specify when to use it.** "Use this when the user asks about product features, troubleshooting steps, or company policies" gives the model selection criteria.

3. **Describe what it returns.** "Returns a list of relevant articles ranked by relevance score" sets expectations for the output format.

4. **Document parameter semantics.** Every parameter should explain not just its type but its meaning and best practices for populating it.

5. **Use enums aggressively.** Constrained values eliminate an entire class of argument construction errors.

In practice, improving tool descriptions often yields a larger accuracy gain than changing the model or adding examples. It is the highest-leverage optimization in the tool calling pipeline.

---

## Dynamic Tool Selection at Scale

When an agent has access to 50 or more tools, passing all of them in every request becomes impractical. Token costs increase, latency grows, and the model's selection accuracy degrades as the tool list expands. The solution is a two-level architecture that first classifies the query into a category, then selects only the relevant tools for the main model.

```python
import numpy as np
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI()


@dataclass
class Tool:
    name: str
    description: str
    category: str
    schema: dict
    handler: callable


class DynamicToolSelector:
    """Two-level tool selection for large tool registries.

    Level 1: A cheap, fast model classifies the query into a category.
    Level 2: Within that category, tools are ranked by semantic similarity
             to the query, and the top-k are selected.
    """

    def __init__(self, tools: list[Tool]):
        self.tools = tools
        self.categories = self._build_categories()
        self.embeddings_cache: dict[str, list[float]] = {}
        self._precompute_tool_embeddings()

    def _build_categories(self) -> dict[str, list[Tool]]:
        """Group tools by category."""
        categories: dict[str, list[Tool]] = {}
        for tool in self.tools:
            categories.setdefault(tool.category, []).append(tool)
        return categories

    def _precompute_tool_embeddings(self):
        """Pre-compute embeddings for all tool descriptions."""
        texts = [
            f"{tool.name}: {tool.description}" for tool in self.tools
        ]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        for tool, data in zip(self.tools, response.data):
            self.embeddings_cache[tool.name] = data.embedding

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for the user query."""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        return response.data[0].embedding

    def _cosine_similarity(
        self, a: list[float], b: list[float]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr, b_arr = np.array(a), np.array(b)
        return float(
            np.dot(a_arr, b_arr)
            / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        )

    def _classify_category(self, query: str) -> str:
        """Level 1: Use a cheap model to classify the query category."""
        category_list = "\n".join(
            f"- {cat}: {len(tools)} tools"
            for cat, tools in self.categories.items()
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap and fast for classification
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user query into exactly one "
                        "category. Reply with the category name only.\n"
                        f"Available categories:\n{category_list}"
                    )
                },
                {"role": "user", "content": query}
            ],
            temperature=0
        )

        predicted = response.choices[0].message.content.strip()
        # Fallback if classification returns unknown category
        if predicted not in self.categories:
            return max(
                self.categories,
                key=lambda c: len(self.categories[c])
            )
        return predicted

    def select_tools(
        self, query: str, top_k: int = 8
    ) -> list[dict]:
        """Select the most relevant tools for a query.

        Returns OpenAI-formatted tool definitions ready for the API.
        """
        # Level 1: Classify into category
        category = self._classify_category(query)
        candidate_tools = self.categories[category]

        # Level 2: Rank by semantic similarity within category
        query_embedding = self._get_query_embedding(query)
        scored_tools = []
        for tool in candidate_tools:
            similarity = self._cosine_similarity(
                query_embedding, self.embeddings_cache[tool.name]
            )
            scored_tools.append((tool, similarity))

        scored_tools.sort(key=lambda x: x[1], reverse=True)
        selected = scored_tools[:top_k]

        # Convert to OpenAI tool format
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema
                }
            }
            for tool, score in selected
        ]


# Usage
selector = DynamicToolSelector(tools=all_registered_tools)

# Instead of passing all 50+ tools:
relevant_tools = selector.select_tools(
    "What's the refund policy for enterprise customers?"
)

# Pass only 5-8 relevant tools to the main model
response = client.chat.completions.create(
    model="gpt-4o",
    messages=conversation,
    tools=relevant_tools  # Focused, relevant subset
)
```

This architecture delivers three benefits simultaneously. First, the main model sees fewer tools, so it selects more accurately. Second, the token cost drops because 8 tool definitions consume far fewer tokens than 50. Third, the classification step uses a cheap model, so the additional latency is minimal.

---

## The ReAct Pattern

ReAct (Reasoning + Acting) is the foundational loop that turns an LLM with tools into an agent. The model alternates between reasoning about what to do next and taking actions through tool calls, continuing until it has enough information to answer the user's question.

```python
import json
from openai import OpenAI

client = OpenAI()

REACT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

For each user request, follow this process:
1. THINK: Analyze what information you need and which tool can provide it.
2. ACT: Call the appropriate tool with the right arguments.
3. OBSERVE: Examine the tool's output.
4. REPEAT: If you need more information, go back to step 1.
5. RESPOND: When you have enough information, provide a final answer.

Always explain your reasoning before making tool calls.
If a tool call fails, try an alternative approach.
Never fabricate data — if you cannot retrieve it, say so."""


def react_loop(
    user_message: str,
    tools: list[dict],
    tool_handlers: dict[str, callable],
    max_iterations: int = 10,
    model: str = "gpt-4o"
) -> str:
    """Execute a ReAct loop until the model produces a final answer.

    Args:
        user_message: The user's query.
        tools: OpenAI-formatted tool definitions.
        tool_handlers: Map of function_name -> callable.
        max_iterations: Safety limit to prevent infinite loops.
        model: Model to use for reasoning.

    Returns:
        The model's final text response.
    """
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        # If no tool calls, the model is done reasoning
        if not message.tool_calls:
            return message.content

        # Execute each tool call
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            handler = tool_handlers.get(function_name)
            if handler is None:
                result = json.dumps({
                    "error": f"Unknown tool: {function_name}"
                })
            else:
                try:
                    result = json.dumps(handler(**arguments))
                except Exception as e:
                    result = json.dumps({
                        "error": f"{type(e).__name__}: {str(e)}"
                    })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    # Safety: if we hit max iterations, ask the model to wrap up
    messages.append({
        "role": "user",
        "content": (
            "You have reached the maximum number of tool calls. "
            "Please provide your best answer with the information "
            "gathered so far."
        )
    })

    final = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return final.choices[0].message.content


# Usage
answer = react_loop(
    user_message="What's the weather in Tokyo and what restaurants "
                 "are nearby with good reviews?",
    tools=tools,
    tool_handlers={
        "get_weather": get_weather,
        "search_restaurants": search_restaurants,
        "get_reviews": get_reviews,
    },
    max_iterations=10
)
```

The `max_iterations` parameter is critical. Without it, a confused model can loop indefinitely, burning tokens and time. In production, values between 5 and 15 cover the vast majority of realistic tasks. If an agent routinely hits the limit, that is a signal that the tools or system prompt need refinement.

---

## Tool Registry

As agent systems grow, managing tool definitions manually becomes unwieldy. A tool registry centralizes registration, automatic schema generation from type hints, categorized lookup, and execution in a single abstraction.

```python
import inspect
import json
from typing import Any, Callable, get_type_hints


class ToolRegistry:
    """Centralized registry for agent tools.

    Provides:
    - Decorator-based registration
    - Automatic JSON Schema generation from type hints
    - Category-based filtering
    - Unified execution interface
    """

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(
        self,
        category: str = "general",
        description: str | None = None
    ) -> Callable:
        """Decorator to register a function as an agent tool.

        Args:
            category: Tool category for filtering.
            description: Override for the tool description.
                         Defaults to the function's docstring.
        """
        def decorator(func: Callable) -> Callable:
            tool_name = func.__name__
            tool_desc = description or func.__doc__ or ""
            schema = self._func_to_schema(func)

            self._tools[tool_name] = {
                "function": func,
                "category": category,
                "definition": {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_desc.strip(),
                        "parameters": schema
                    }
                }
            }
            return func

        return decorator

    def _func_to_schema(self, func: Callable) -> dict:
        """Generate JSON Schema from function signature and type hints.

        Inspects the function's parameters, type annotations, and
        defaults to produce a valid JSON Schema object.
        """
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        properties = {}
        required = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = hints.get(param_name, str)
            json_type = type_map.get(param_type, "string")

            prop: dict[str, Any] = {"type": json_type}

            # Extract parameter description from docstring
            if func.__doc__:
                for line in func.__doc__.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith(f"{param_name}:"):
                        prop["description"] = stripped.split(
                            ":", 1
                        )[1].strip()

            if param.default is not inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                required.append(param_name)

            properties[param_name] = prop

        schema = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        return schema

    def get_tools(
        self, category: str | None = None
    ) -> list[dict]:
        """Get OpenAI-formatted tool definitions.

        Args:
            category: If provided, filter to this category only.

        Returns:
            List of tool definitions ready for the API.
        """
        return [
            entry["definition"]
            for entry in self._tools.values()
            if category is None or entry["category"] == category
        ]

    def execute(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        """Execute a registered tool by name.

        Args:
            tool_name: Name of the registered tool.
            arguments: Keyword arguments for the function.

        Returns:
            The function's return value.

        Raises:
            KeyError: If the tool is not registered.
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered")

        func = self._tools[tool_name]["function"]
        return func(**arguments)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def categories(self) -> list[str]:
        return list({
            entry["category"] for entry in self._tools.values()
        })


# ── Usage ──

registry = ToolRegistry()

@registry.register(category="weather")
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get the current weather for a city.

    city: The name of the city to check weather for.
    unit: Temperature unit — celsius or fahrenheit.
    """
    # Implementation calls a real weather API
    return {"city": city, "temperature": 22, "unit": unit}


@registry.register(category="database")
def search_products(query: str, max_results: int = 10) -> list:
    """Search the product catalog by keyword.

    query: Search terms for finding products.
    max_results: Maximum number of products to return.
    """
    # Implementation queries the product database
    return [{"name": "Widget", "price": 9.99}]


@registry.register(category="database")
def get_user_orders(user_id: str) -> list:
    """Retrieve order history for a specific user.

    user_id: The unique identifier of the user.
    """
    return [{"order_id": "ORD-001", "status": "delivered"}]


# Get all tools
all_tools = registry.get_tools()

# Get only database tools
db_tools = registry.get_tools(category="database")

# Execute a tool
result = registry.execute(
    "get_weather", {"city": "Tokyo", "unit": "celsius"}
)
# {"city": "Tokyo", "temperature": 22, "unit": "celsius"}

# Integration with the ReAct loop
answer = react_loop(
    user_message="Find me some widgets and check Tokyo weather",
    tools=registry.get_tools(),
    tool_handlers={
        name: registry._tools[name]["function"]
        for name in registry.tool_names
    }
)
```

The registry pattern scales naturally. New tools are added by writing a function and applying a decorator. Schema generation is automatic. Category filtering integrates cleanly with the dynamic tool selector described earlier.

---

## Structured Output with Instructor

When the desired output is not a tool call but a structured data object, the Instructor library provides a clean abstraction over function calling. It uses Pydantic models to define the output schema, validates the response, and retries automatically on validation failure.

```python
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI


class ExtractedData(BaseModel):
    """Structured extraction of company information."""

    company_name: str = Field(
        description="Official company name"
    )
    ticker: str | None = Field(
        default=None,
        description="Stock ticker symbol if publicly traded"
    )
    revenue: float | None = Field(
        default=None,
        description="Most recent annual revenue in billions USD"
    )
    num_employees: int | None = Field(
        default=None,
        description="Approximate number of employees"
    )
    key_products: list[str] = Field(
        default_factory=list,
        description="Main products or services"
    )
    summary: str = Field(
        description="One-sentence company summary"
    )


# Patch the client to enable structured output
client = instructor.from_openai(OpenAI())

# The response is a validated Pydantic object, not raw text
data = client.chat.completions.create(
    model="gpt-4o",
    response_model=ExtractedData,
    messages=[
        {
            "role": "user",
            "content": (
                "Extract company information from this text: "
                "Apple Inc. (AAPL) reported $383 billion in revenue "
                "for fiscal year 2023. The company employs approximately "
                "161,000 people and is known for the iPhone, Mac, iPad, "
                "and Apple Watch. Apple designs consumer electronics, "
                "software, and services."
            )
        }
    ],
    max_retries=3  # Retries with validation feedback on failure
)

print(data.model_dump_json(indent=2))
# {
#   "company_name": "Apple Inc.",
#   "ticker": "AAPL",
#   "revenue": 383.0,
#   "num_employees": 161000,
#   "key_products": ["iPhone", "Mac", "iPad", "Apple Watch"],
#   "summary": "Apple designs and sells consumer electronics,
#               software, and services worldwide."
# }
```

Instructor works by converting the Pydantic model into a function calling schema, sending it to the API, and then parsing and validating the response. On validation failure, it sends the error message back to the model and retries. This feedback loop means the model learns from its mistakes within the same request, producing remarkably reliable structured output.

---

## Security

Tool calling introduces a direct interface between LLM output and system execution. Without proper safeguards, a prompt injection attack or a malformed argument could trigger unauthorized database queries, file system access, or external API calls. Security must be enforced at the execution layer, not the prompt layer.

```python
import re
import time
import logging
from typing import Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class SecureToolExecutor:
    """Security-hardened tool execution layer.

    Enforces input validation, permission checks, and rate limiting
    before any tool is executed. All calls are audit-logged.
    """

    # Permission levels for tools
    PERMISSION_LEVELS = {
        "basic": ["get_weather", "search_products", "get_faq"],
        "elevated": ["get_user_data", "search_database", "get_orders"],
        "admin": ["modify_user", "delete_record", "execute_query"],
    }

    # Patterns that indicate injection attempts
    DANGEROUS_PATTERNS = [
        r";\s*DROP\s+TABLE",       # SQL injection
        r";\s*DELETE\s+FROM",      # SQL injection
        r"\bOR\b\s+1\s*=\s*1",    # SQL injection
        r"__import__\s*\(",        # Python code injection
        r"subprocess\.",           # Command injection
        r"\bexec\s*\(",            # Code execution
        r"\beval\s*\(",            # Code evaluation
        r"\.\./",                  # Path traversal
        r"<script",               # XSS
    ]

    def __init__(self, registry, rate_limit_per_minute: int = 30):
        self.registry = registry
        self.rate_limit = rate_limit_per_minute
        self._call_timestamps: dict[str, list[float]] = defaultdict(list)

    def validate_args(self, tool_name: str, args: dict) -> dict:
        """Validate and sanitize tool arguments.

        Checks all string arguments against known dangerous patterns
        and strips potentially harmful content.

        Raises:
            ValueError: If a dangerous pattern is detected.
        """
        sanitized = {}
        for key, value in args.items():
            if isinstance(value, str):
                # Check against injection patterns
                for pattern in self.DANGEROUS_PATTERNS:
                    if re.search(pattern, value, re.IGNORECASE):
                        logger.warning(
                            "Blocked dangerous input for tool "
                            "'%s', arg '%s': matched pattern '%s'",
                            tool_name, key, pattern
                        )
                        raise ValueError(
                            f"Potentially dangerous input detected "
                            f"in argument '{key}'"
                        )

                # Basic sanitization: strip control characters
                sanitized[key] = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
                                        "", value)
            else:
                sanitized[key] = value

        return sanitized

    def check_permission(
        self, tool_name: str, user_role: str
    ) -> bool:
        """Check if the user role has permission to call this tool.

        Follows least-privilege: a role can access its own tools
        and all tools at lower privilege levels.
        """
        role_hierarchy = ["basic", "elevated", "admin"]

        if user_role not in role_hierarchy:
            return False

        user_level = role_hierarchy.index(user_role)

        for level_name, tools in self.PERMISSION_LEVELS.items():
            if tool_name in tools:
                required_level = role_hierarchy.index(level_name)
                return user_level >= required_level

        # Unknown tool: deny by default
        return False

    def check_rate_limit(
        self, tool_name: str, user_id: str
    ) -> bool:
        """Enforce per-user, per-tool rate limiting.

        Uses a sliding window to count calls in the last minute.
        """
        key = f"{user_id}:{tool_name}"
        now = time.time()
        window_start = now - 60

        # Remove timestamps outside the window
        self._call_timestamps[key] = [
            ts for ts in self._call_timestamps[key]
            if ts > window_start
        ]

        if len(self._call_timestamps[key]) >= self.rate_limit:
            logger.warning(
                "Rate limit exceeded for user '%s' on tool '%s'",
                user_id, tool_name
            )
            return False

        self._call_timestamps[key].append(now)
        return True

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_id: str,
        user_role: str = "basic"
    ) -> dict:
        """Execute a tool with full security checks.

        Performs input validation, permission checking, and rate
        limiting before delegating to the actual tool handler.

        Returns:
            A dict with either 'result' or 'error' key.
        """
        # 1. Permission check
        if not self.check_permission(tool_name, user_role):
            logger.warning(
                "Permission denied: user '%s' (role=%s) "
                "attempted to call '%s'",
                user_id, user_role, tool_name
            )
            return {
                "error": f"Permission denied for tool '{tool_name}'"
            }

        # 2. Rate limit check
        if not self.check_rate_limit(tool_name, user_id):
            return {
                "error": "Rate limit exceeded. Try again later."
            }

        # 3. Input validation and sanitization
        try:
            safe_args = self.validate_args(tool_name, arguments)
        except ValueError as e:
            return {"error": str(e)}

        # 4. Execute the tool
        try:
            result = self.registry.execute(tool_name, safe_args)

            # 5. Audit log
            logger.info(
                "Tool executed: user='%s' tool='%s' args=%s",
                user_id, tool_name, list(safe_args.keys())
            )

            return {"result": result}

        except Exception as e:
            logger.error(
                "Tool execution failed: tool='%s' error='%s'",
                tool_name, str(e)
            )
            return {"error": f"Tool execution failed: {str(e)}"}


# Usage
secure_executor = SecureToolExecutor(registry, rate_limit_per_minute=20)

result = secure_executor.execute(
    tool_name="search_products",
    arguments={"query": "wireless headphones"},
    user_id="user_123",
    user_role="basic"
)

# This would be blocked:
result = secure_executor.execute(
    tool_name="delete_record",
    arguments={"record_id": "123"},
    user_id="user_123",
    user_role="basic"  # Needs "admin" role
)
# {"error": "Permission denied for tool 'delete_record'"}
```

The defense-in-depth approach here is intentional. Input validation catches injection attacks. Permission checks enforce least privilege. Rate limiting prevents abuse. And audit logging provides the forensic trail needed for incident response. No single layer is sufficient on its own.

---

## Error Handling and Resilience

In production, tools fail. APIs time out, databases go down, rate limits are hit, and responses arrive malformed. A resilient tool executor must handle all of these cases gracefully, with retries, timeouts, exponential backoff, and fallback strategies.

```python
import asyncio
import time
import logging
from typing import Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class RobustToolExecutor:
    """Production-grade tool executor with retry, timeout,
    exponential backoff, and fallback support.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = 30.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self._fallbacks: dict[str, Callable] = {}

    def register_fallback(
        self, tool_name: str, fallback_fn: Callable
    ):
        """Register a fallback function for a tool.

        The fallback is invoked when the primary tool fails
        after exhausting all retries.
        """
        self._fallbacks[tool_name] = fallback_fn

    def _calculate_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        import random
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)
        # Add jitter: +/- 25%
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter

    async def execute_with_retry(
        self,
        tool_name: str,
        func: Callable,
        arguments: dict[str, Any]
    ) -> dict:
        """Execute a tool with retry, timeout, and fallback.

        Attempts:
        1. Call the function with a timeout.
        2. On failure, retry with exponential backoff.
        3. After exhausting retries, invoke the fallback if available.
        4. If no fallback, return a structured error.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Apply timeout
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(**arguments),
                        timeout=self.timeout
                    )
                else:
                    # Run sync functions in a thread pool with timeout
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda: func(**arguments)
                        ),
                        timeout=self.timeout
                    )

                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1
                }

            except asyncio.TimeoutError:
                last_error = (
                    f"Timeout after {self.timeout}s on "
                    f"attempt {attempt + 1}"
                )
                logger.warning(
                    "Tool '%s' timed out on attempt %d/%d",
                    tool_name, attempt + 1, self.max_retries
                )

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                logger.warning(
                    "Tool '%s' failed on attempt %d/%d: %s",
                    tool_name, attempt + 1, self.max_retries,
                    last_error
                )

            # Exponential backoff before next retry
            if attempt < self.max_retries - 1:
                delay = self._calculate_delay(attempt)
                logger.info(
                    "Retrying tool '%s' in %.1fs...",
                    tool_name, delay
                )
                await asyncio.sleep(delay)

        # All retries exhausted — try fallback
        if tool_name in self._fallbacks:
            logger.info(
                "Invoking fallback for tool '%s'", tool_name
            )
            try:
                fallback_result = self._fallbacks[tool_name](
                    **arguments
                )
                return {
                    "success": True,
                    "result": fallback_result,
                    "attempts": self.max_retries,
                    "used_fallback": True
                }
            except Exception as e:
                logger.error(
                    "Fallback for tool '%s' also failed: %s",
                    tool_name, str(e)
                )

        return {
            "success": False,
            "error": last_error,
            "attempts": self.max_retries,
            "tool": tool_name
        }


# ── Usage ──

executor = RobustToolExecutor(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    timeout=10.0
)

# Register a fallback for the weather tool
def weather_fallback(city: str, **kwargs) -> dict:
    """Return cached or default weather when API is down."""
    return {
        "city": city,
        "temperature": None,
        "conditions": "unavailable — using cached data",
        "cached": True
    }

executor.register_fallback("get_weather", weather_fallback)

# Execute with full resilience
result = await executor.execute_with_retry(
    tool_name="get_weather",
    func=get_weather,
    arguments={"city": "Tokyo"}
)

if result["success"]:
    print(f"Weather data: {result['result']}")
    if result.get("used_fallback"):
        print("Note: used fallback data")
else:
    print(f"Failed after {result['attempts']} attempts: "
          f"{result['error']}")
```

The combination of timeout, retry with exponential backoff, and fallback handles the three most common production failure modes: slow responses, transient errors, and total outages. The jitter in the backoff calculation prevents the thundering herd problem when multiple agents retry simultaneously against the same service.

---

## Key Insights

> **System Design for 50+ Tools:** Two-level architecture: (1) Category router with cheap model classifies query, (2) Select 5-10 tools from category for main model. Plus: semantic search over tool descriptions for ranking, tool registry with auto schema generation.

> **Tool Calling Security:** (1) Input validation/sanitization, (2) Permission model with least privilege, (3) Rate limiting per tool and user, (4) Sandbox for code execution, (5) Audit logging, (6) Prompt injection protection.

---

## References

- OpenAI Function Calling: [https://platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
- Anthropic Tool Use: [https://docs.anthropic.com/en/docs/build-with-claude/tool-use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- Instructor: [https://python.useinstructor.com/](https://python.useinstructor.com/)
- "Toolformer" Paper: [https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)
