# Chapter 3: Prompt Optimization

> *"The best programs are the ones that can rewrite themselves."* — Adapted from the tradition of self-improving systems

---

Manual prompt engineering is fragile, time-consuming, and fundamentally unscalable. A carefully tuned prompt for GPT-4 may fail when migrated to Claude or Llama. A prompt that works for one task distribution may degrade as your data shifts. And the iterative cycle of "tweak the prompt, test, repeat" is antithetical to principled engineering.

This chapter explores the tools and techniques that transform prompt engineering from an art into a science. We begin with **DSPy**, which replaces hand-crafted prompts with programmatic optimization. We then examine **Guidance** and **LMQL**, which provide fine-grained control over the generation process itself. Finally, we cover **structured output generation** — the critical problem of ensuring LLM outputs conform to precise schemas.

By the end of this chapter, you will understand how to build prompt pipelines that optimize themselves, constrain generation at the token level, and guarantee output format compliance.

---

## DSPy — Declarative Self-Improving Python

### Philosophy

DSPy, developed at Stanford NLP, represents a paradigm shift in how we interact with language models. The core insight is deceptively simple: **prompts are parameters, not code.** Just as neural network weights are learned rather than hand-specified, prompts should be optimized rather than hand-written.

In traditional prompt engineering, you write a prompt, test it, adjust wording, add few-shot examples, adjust again — an endless cycle that produces brittle, model-specific artifacts. DSPy replaces this with a programming model where you:

1. **Declare** what the LLM should do (signatures)
2. **Compose** modules into pipelines (programs)
3. **Optimize** the entire pipeline automatically (teleprompters/optimizers)

The key abstraction is the separation of concerns: the *what* (signatures) is decoupled from the *how* (prompts, few-shot examples, chain-of-thought reasoning). DSPy's optimizers discover the *how* automatically by searching over prompt strategies, few-shot example selections, and instruction phrasings.

This means you can swap the underlying LLM — from GPT-4 to Llama 3 to Mistral — and simply re-optimize, rather than rewriting every prompt from scratch.

### Signatures

A **signature** in DSPy is a declarative specification of input/output behavior. It describes *what* a module should do without specifying *how*.

The simplest form is an inline signature using the `"input -> output"` string notation:

```python
import dspy

# Configure the language model
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Inline signature — simplest form
qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print(result.answer)  # "Paris"
```

For more complex tasks, you define a class-based signature with field descriptions. These descriptions serve as documentation and are used by DSPy's optimizers to generate better prompts:

```python
class BasicQA(dspy.Signature):
    """Answer questions with short, factual answers."""

    question = dspy.InputField(desc="A factual question")
    answer = dspy.OutputField(desc="A concise answer, often 1-5 words")
```

Signatures can have multiple inputs and outputs, enabling complex multi-field interactions:

```python
class RAGSignature(dspy.Signature):
    """Answer questions using the provided context.
    If the context does not contain the answer, say 'I don't know'."""

    context = dspy.InputField(desc="Relevant passages from the knowledge base")
    question = dspy.InputField(desc="The user's question")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning based on context")
    answer = dspy.OutputField(desc="The final answer")
    confidence = dspy.OutputField(
        desc="Confidence level: high, medium, or low"
    )
```

Signatures are not prompts — they are specifications. DSPy compiles them into prompts (including instructions, few-shot examples, and formatting directives) during the optimization phase.

### Modules

DSPy provides a hierarchy of **modules** — pre-built LLM interaction patterns with increasing sophistication. Each module takes a signature and implements a specific prompting strategy.

#### Predict

The simplest module. It calls the LLM directly with the compiled signature:

```python
# Basic prediction — one LLM call
predictor = dspy.Predict(BasicQA)
result = predictor(question="Who wrote 'War and Peace'?")
print(result.answer)  # "Leo Tolstoy"
```

#### ChainOfThought

Automatically adds a `rationale` field before the answer, encouraging the model to reason step-by-step before answering. This consistently improves accuracy on complex tasks:

```python
# Chain-of-thought — adds reasoning before the answer
cot = dspy.ChainOfThought(BasicQA)
result = cot(question="If a train travels 120 km in 2 hours, what is its speed?")
print(result.rationale)  # "Speed = distance / time = 120 / 2 = 60 km/h"
print(result.answer)     # "60 km/h"
```

#### ReAct

Implements the Reason-Act loop pattern. The LLM alternates between *thinking* and *acting* (calling tools), enabling multi-step problem solving with external tool access:

```python
# Define tools for the ReAct agent
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a summary."""
    # Implementation here
    ...

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# ReAct module with tools
react = dspy.ReAct(
    BasicQA,
    tools=[search_wikipedia, calculate]
)

result = react(
    question="What is the population of France divided by 1000?"
)
# The agent will:
# 1. Think: I need to find France's population
# 2. Act: search_wikipedia("population of France")
# 3. Think: Now I need to divide by 1000
# 4. Act: calculate("67390000 / 1000")
# 5. Answer: 67390
```

#### ProgramOfThought

Instead of reasoning in natural language, this module generates executable code to solve the problem. Particularly effective for mathematical and data manipulation tasks:

```python
# Program-of-thought — generates code to solve the problem
pot = dspy.ProgramOfThought(BasicQA)
result = pot(
    question="What is the sum of the first 100 prime numbers?"
)
# Internally generates and executes Python code to compute the answer
print(result.answer)  # "24133"
```

### Building Pipelines

DSPy's real power emerges when you compose modules into multi-step pipelines. A pipeline is a `dspy.Module` subclass that wires together multiple LLM calls, retrieval steps, and control flow.

#### RAG Pipeline

```python
class RAGPipeline(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(RAGSignature)

    def forward(self, question):
        # Step 1: Retrieve relevant passages
        context = self.retrieve(question).passages

        # Step 2: Generate answer with chain-of-thought reasoning
        result = self.generate(
            context=context,
            question=question
        )
        return dspy.Prediction(
            answer=result.answer,
            reasoning=result.reasoning,
            confidence=result.confidence
        )
```

#### Self-Checking RAG

More sophisticated pipelines add verification steps. This pattern has the model check its own answer against the retrieved context:

```python
class SelfCheckRAG(dspy.Module):
    """RAG pipeline with self-verification: generates an answer,
    then checks it for faithfulness to the retrieved context."""

    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(RAGSignature)
        self.verify = dspy.ChainOfThought(
            "context, question, answer -> is_faithful, revised_answer"
        )

    def forward(self, question):
        # Retrieve
        context = self.retrieve(question).passages

        # Generate initial answer
        result = self.generate(context=context, question=question)

        # Self-check: is the answer faithful to the context?
        check = self.verify(
            context=context,
            question=question,
            answer=result.answer
        )

        # Return revised answer if verification found issues
        final_answer = (
            check.revised_answer
            if check.is_faithful.lower() == "no"
            else result.answer
        )

        return dspy.Prediction(
            answer=final_answer,
            reasoning=result.reasoning,
            is_faithful=check.is_faithful
        )
```

### Optimizers (Teleprompters)

Optimizers are the heart of DSPy. They automatically tune prompts, select few-shot examples, and refine instructions to maximize a given metric. Think of them as *compilers* that transform your declarative program into an optimized prompt pipeline.

#### BootstrapFewShot

The simplest optimizer. It generates few-shot examples by running the pipeline on training data and selecting successful examples as demonstrations:

```python
from dspy.teleprompt import BootstrapFewShot

# Define a metric: does the predicted answer match the gold answer?
def answer_match(example, prediction, trace=None):
    return example.answer.lower() == prediction.answer.lower()

# Create the optimizer
optimizer = BootstrapFewShot(
    metric=answer_match,
    max_bootstrapped_demos=4,   # Max few-shot examples to generate
    max_labeled_demos=4          # Max labeled examples to include
)

# Compile (optimize) the pipeline
compiled_rag = optimizer.compile(
    RAGPipeline(),
    trainset=train_examples
)

# The compiled pipeline now includes optimized few-shot examples
result = compiled_rag(question="What causes tides?")
```

#### BootstrapFewShotWithRandomSearch

Extends `BootstrapFewShot` by running multiple random trials and selecting the best combination of few-shot examples. More robust but slower:

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    metric=answer_match,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,  # Try 10 random combinations
    num_threads=4               # Parallelize evaluation
)

compiled_rag = optimizer.compile(
    RAGPipeline(),
    trainset=train_examples,
    valset=val_examples          # Validation set for selection
)
```

#### MIPRO (Multi-prompt Instruction Proposal Optimizer)

MIPRO optimizes not just few-shot examples but also the *instructions* themselves. It uses a Bayesian approach to search over both instruction phrasings and demonstration selections:

```python
from dspy.teleprompt import MIPRO

optimizer = MIPRO(
    metric=answer_match,
    num_candidates=20,          # Instruction candidates to generate
    init_temperature=1.0        # Creativity for instruction generation
)

compiled_rag = optimizer.compile(
    RAGPipeline(),
    trainset=train_examples,
    num_trials=50,              # Total optimization trials
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
```

#### BayesianSignatureOptimizer

The most sophisticated optimizer. It uses Bayesian optimization to jointly optimize instructions, field descriptions, and few-shot examples across all modules in the pipeline:

```python
from dspy.teleprompt import BayesianSignatureOptimizer

optimizer = BayesianSignatureOptimizer(
    metric=answer_match,
    n=20,                        # Candidates per iteration
    init_temperature=1.4,
    verbose=True
)

compiled_rag = optimizer.compile(
    RAGPipeline(),
    devset=train_examples,
    num_threads=4,
    eval_kwargs={"num_threads": 4}
)
```

### How Optimizers Work (Internal Algorithm)

Understanding the internal mechanics helps you choose the right optimizer and debug optimization failures.

#### BootstrapFewShot Algorithm

The bootstrap process follows five steps:

1. **Run the pipeline** on each training example, recording all intermediate LLM calls (the "trace").
2. **Filter traces** — keep only those where the final output passes the metric (successful executions).
3. **Extract demonstrations** — from successful traces, extract input/output pairs for each module in the pipeline.
4. **Select demonstrations** — choose up to `max_bootstrapped_demos` demonstrations per module. These become the few-shot examples in the optimized prompt.
5. **Combine with labeled demos** — if labeled demonstrations are available (from the training set directly), prepend them before the bootstrapped ones, up to `max_labeled_demos`.

The key insight is that bootstrapping is *self-referential*: the model generates its own few-shot examples by running on training data. This means the examples naturally match the model's own "voice" and reasoning style, which often outperforms hand-written examples.

#### MIPRO Algorithm

MIPRO adds instruction optimization on top of demonstration selection:

1. **Generate instruction candidates** — use the LLM to propose multiple phrasings for each module's instructions. MIPRO provides the module's signature, field descriptions, and a few examples as context for instruction generation.
2. **Build a search space** — the search space is the Cartesian product of instruction candidates and demonstration subsets for every module in the pipeline.
3. **Bayesian search** — use a surrogate model (typically Tree-structured Parzen Estimator) to efficiently explore this combinatorial space. Each trial evaluates a specific combination of instructions and demonstrations on the validation set.
4. **Select the best** — after all trials, return the combination with the highest validation metric.

MIPRO is significantly more expensive than BootstrapFewShot (it requires many evaluation runs), but it often discovers non-obvious instruction phrasings that substantially improve performance.

### Assertions and Constraints

DSPy provides `dspy.Assert` and `dspy.Suggest` to enforce constraints on LLM outputs within pipelines. This is critical for production systems where outputs must satisfy business rules.

- **`dspy.Assert`** — hard constraint. If violated, the pipeline retries the LLM call (with a corrective message) up to a configurable number of times. If all retries fail, it raises an exception.
- **`dspy.Suggest`** — soft constraint. If violated, it logs a warning and adds the corrective message to the next attempt, but does not raise an exception.

```python
class FactualQA(dspy.Module):
    """QA module with factual grounding constraints."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(
            "context, question -> answer"
        )

    def forward(self, context, question):
        result = self.generate(context=context, question=question)

        # Hard constraint: answer must not be empty
        dspy.Assert(
            len(result.answer) > 0,
            "The answer must not be empty. Please provide a substantive answer."
        )

        # Hard constraint: answer must be grounded in context
        dspy.Assert(
            any(
                word in context.lower()
                for word in result.answer.lower().split()
                if len(word) > 3
            ),
            "The answer must be based on the provided context. "
            "Please re-read the context and answer again."
        )

        # Soft constraint: prefer concise answers
        dspy.Suggest(
            len(result.answer.split()) < 50,
            "Please provide a more concise answer (under 50 words)."
        )

        return result
```

When an assertion fails, DSPy automatically:
1. Appends the error message to the prompt
2. Retries the LLM call with this additional guidance
3. Repeats up to the configured retry limit

This creates a self-correcting loop where the model learns from its own constraint violations within a single inference pass.

### Evaluation

DSPy provides built-in evaluation utilities to measure pipeline performance before and after optimization — the fundamental feedback loop that makes systematic improvement possible:

```python
from dspy.evaluate import Evaluate

# Configure evaluation
evaluator = Evaluate(
    devset=test_examples,
    metric=answer_match,
    num_threads=4,
    display_progress=True,
    display_table=5           # Show 5 example results in a table
)

# Evaluate the unoptimized pipeline
baseline_score = evaluator(RAGPipeline())
print(f"Baseline accuracy: {baseline_score:.1f}%")
# Output: Baseline accuracy: 42.0%

# Evaluate the optimized pipeline
optimized_score = evaluator(compiled_rag)
print(f"Optimized accuracy: {optimized_score:.1f}%")
# Output: Optimized accuracy: 67.0%

# Relative improvement
improvement = (optimized_score - baseline_score) / baseline_score * 100
print(f"Relative improvement: {improvement:.1f}%")
# Output: Relative improvement: 59.5%
```

These numbers are representative: DSPy optimization routinely yields 20-60% relative improvement over unoptimized baselines. The gains come from three sources: better few-shot example selection, improved instructions, and chain-of-thought prompting strategies that the optimizer discovers automatically.

---

## Guidance

**Guidance** (by Microsoft) takes a fundamentally different approach to prompt optimization. Rather than optimizing prompts at the *pipeline* level (like DSPy), Guidance provides **token-level control** over the generation process itself. You write templates that interleave text, generation calls, and constraints — producing a hybrid of prompt and program.

### Constrained Generation

The core idea is that you can constrain *what the model generates* at each step. Instead of generating freely and then parsing, you guide the generation token-by-token:

```python
from guidance import models, gen, select

# Load a model (local or API-based)
lm = models.OpenAI("gpt-4o-mini")

# Constrained generation using Guidance's template syntax
@guidance
def extract_info(lm, text):
    lm += f"""Extract structured information from the following text.

Text: {text}

Name: {gen('name', stop='\\n')}
Age: {gen('age', regex='[0-9]{{1,3}}', stop='\\n')}
Sentiment: {select(['positive', 'negative', 'neutral'], name='sentiment')}
"""
    return lm

# Execute
result = lm + extract_info("John Smith, 32 years old, loves his new job")
print(result["name"])       # "John Smith"
print(result["age"])        # "32"
print(result["sentiment"])  # "positive"
```

Key features of constrained generation:

- **`gen()`** — generate text with optional constraints: `stop` tokens, `regex` patterns, `max_tokens` limits.
- **`select()`** — force the model to choose from a predefined list of options. The model's logits are masked so only valid options receive probability mass.
- **`regex`** — constrain generation to match a regular expression. Invalid tokens are masked at each step.

This is not post-hoc validation. The constraints are enforced *during* generation: at each token position, the model can only emit tokens consistent with the constraints. This guarantees format compliance without retries.

### Token Healing

A subtle but important feature. Consider this prompt:

```
The URL is: https://
```

If the tokenizer splits `https://` across multiple tokens, the model may have been trained on the combined token `https://` but never seen the partial token `https:/` followed by a generation boundary. This mismatch can cause the model to generate poorly.

**Token healing** solves this by backing up the generation boundary to the nearest clean token boundary. Instead of starting generation at `https://`, Guidance rolls back to `https:` and lets the model regenerate `//` naturally. This consistently improves generation quality at prompt/generation boundaries.

### Templates with Logic

Guidance templates support full control flow — conditionals, loops, function calls — within the generation process:

```python
@guidance
def classify_and_respond(lm, user_input):
    lm += f"""Analyze the following user input and respond appropriately.

Input: {user_input}

Category: {select(['question', 'complaint', 'feedback', 'request'], name='category')}
Priority: {select(['low', 'medium', 'high'], name='priority')}
"""

    # Conditional generation based on classification
    if lm["category"] == "complaint":
        lm += f"""
Since this is a complaint, provide an empathetic response:
Response: {gen('response', max_tokens=150, stop='\\n\\n')}
Escalation needed: {select(['yes', 'no'], name='escalate')}
"""
    elif lm["category"] == "question":
        lm += f"""
Since this is a question, provide a helpful answer:
Answer: {gen('response', max_tokens=200, stop='\\n\\n')}
"""
    else:
        lm += f"""
Acknowledge the {lm['category']}:
Response: {gen('response', max_tokens=100, stop='\\n\\n')}
"""

    return lm

result = lm + classify_and_respond("Your product broke after one day!")
print(result["category"])   # "complaint"
print(result["priority"])   # "high"
print(result["response"])   # Empathetic response text
print(result["escalate"])   # "yes"
```

The critical distinction from DSPy is granularity. DSPy optimizes at the prompt/pipeline level — what instructions and examples to use. Guidance controls at the token level — what tokens the model is allowed to produce at each position. These approaches are complementary: you could use DSPy to optimize the template instructions while using Guidance to constrain the output format.

---

## LMQL

**LMQL** (Language Model Query Language) brings a SQL-like declarative syntax to LLM interaction. Developed at ETH Zurich, it combines natural language prompting with formal constraints and decoding control in a single query language.

### Query Language for LLMs

LMQL queries look like Python functions decorated with `@lmql.query`, mixing natural language prompts with typed variable declarations and constraint clauses:

```python
import lmql

@lmql.query
def classify_sentiment(text):
    '''lmql
    "Given the following text, classify its sentiment.\n"
    "Text: {text}\n"
    "Sentiment: [SENTIMENT]\n"
    "Confidence (0-100): [CONFIDENCE]\n"
    "Explanation: [EXPLANATION]"
    where
        SENTIMENT in ["positive", "negative", "neutral"]
        and int(CONFIDENCE) >= 0
        and int(CONFIDENCE) <= 100
        and len(EXPLANATION.split()) < 30
    return {
        "sentiment": SENTIMENT,
        "confidence": int(CONFIDENCE),
        "explanation": EXPLANATION
    }
    '''

result = classify_sentiment("This product exceeded all my expectations!")
# {"sentiment": "positive", "confidence": 95, "explanation": "..."}
```

Variables in square brackets (`[SENTIMENT]`, `[CONFIDENCE]`) are generation targets. The `where` clause specifies constraints that must hold. LMQL enforces these constraints during decoding — similar to Guidance's approach, but expressed declaratively.

### Constraints

LMQL supports a rich constraint language:

```python
@lmql.query
def generate_structured(topic):
    '''lmql
    "Write a structured analysis of {topic}.\n\n"
    "Category: [CATEGORY]\n"
    "Impact level: [IMPACT]\n"
    "Summary: [SUMMARY]\n"
    "Key points:\n"
    "1. [POINT1]\n"
    "2. [POINT2]\n"
    "3. [POINT3]\n"
    "Recommendation: [REC]"
    where
        CATEGORY in ["technology", "economics", "environment", "social"]
        and IMPACT in ["low", "medium", "high", "critical"]
        and len(SUMMARY.split()) in range(20, 51)
        and len(POINT1.split()) < 20
        and len(POINT2.split()) < 20
        and len(POINT3.split()) < 20
        and STOPS_AT(REC, "\n")
    '''
```

Available constraint types include:

- **Set membership:** `VAR in ["a", "b", "c"]`
- **Length constraints:** `len(VAR.split()) < N`
- **Type constraints:** `int(VAR) >= 0`
- **Stop conditions:** `STOPS_AT(VAR, "\n")`
- **Regular expressions:** `REGEX(VAR, r"[0-9]{3}-[0-9]{4}")`

### Decoding Strategies

LMQL provides fine-grained control over the decoding algorithm. Different strategies trade off between speed, diversity, and quality:

```python
@lmql.query
def beam_search_example(topic):
    '''lmql
    beam(n=3)               # Use beam search with 3 beams
    "Write a concise definition of {topic}: [DEFINITION]"
    where
        len(DEFINITION.split()) in range(10, 31)
    return DEFINITION
    '''

# Alternative decoding strategies:
# sample(temperature=0.7)   — temperature sampling
# argmax                     — greedy decoding
# beam(n=5)                  — beam search with 5 beams
# beam_var(n=3, num_return=2) — beam search returning top 2 results
```

Beam search is particularly useful when combined with constraints: it explores multiple generation paths simultaneously, increasing the likelihood of finding a high-quality output that satisfies all constraints.

---

## Structured Output Generation

Ensuring that LLM outputs conform to a precise schema is one of the most practical challenges in production systems. Whether you need a JSON object matching a Pydantic model, a phone number in a specific format, or a response that fits a database schema — structured output generation is essential.

### Comparison Table

| Approach | Format Guarantee | Speed | Flexibility |
|---|---|---|---|
| JSON mode (API) | High | Fast | Limited |
| Function calling | High | Fast | JSON schema |
| Instructor | High | Fast | Pydantic models |
| Constrained decoding (Outlines) | 100% | Slower | Any format |
| Post-hoc validation | No guarantee | Fast | Any format |

Each approach occupies a different point in the reliability-flexibility-speed trade-off space. The right choice depends on your model (API vs. open-source), reliability requirements, and output complexity.

### Instructor — Recommended Approach

**Instructor** is a lightweight library that patches LLM API clients to return validated Pydantic models instead of raw strings. It handles retries, validation, and schema communication transparently.

For most production use cases involving API models (OpenAI, Anthropic, etc.), Instructor is the recommended approach:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# Define the output schema as a Pydantic model
class UserInfo(BaseModel):
    """Structured representation of user information."""
    name: str = Field(description="Full name of the user")
    age: int = Field(description="Age in years", ge=0, le=150)
    email: Optional[str] = Field(
        default=None,
        description="Email address"
    )
    interests: List[str] = Field(
        description="List of user interests",
        min_length=1
    )
    sentiment: str = Field(
        description="Overall sentiment of the text about this user"
    )

# Extract structured data — returns a validated Pydantic model
user = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=UserInfo,
    messages=[
        {
            "role": "user",
            "content": """
                Extract user information from this text:
                'John Smith, 28, is an avid reader and Python developer.
                He recently started a blog at john@example.com and
                seems very enthusiastic about AI and machine learning.'
            """
        }
    ],
    max_retries=3  # Automatic retry on validation failure
)

print(user.model_dump_json(indent=2))
# {
#   "name": "John Smith",
#   "age": 28,
#   "email": "john@example.com",
#   "interests": ["reading", "Python development", "AI", "machine learning", "blogging"],
#   "sentiment": "positive"
# }
```

How Instructor works under the hood:

1. Converts the Pydantic model to a JSON schema
2. Passes the schema to the LLM via function calling or JSON mode
3. Parses the LLM's response into a Pydantic model instance
4. If validation fails, appends the validation error to the prompt and retries
5. Returns a fully validated Python object

This retry-on-validation-failure loop makes Instructor highly reliable in practice: even when the LLM produces slightly malformed output, the retry with error context almost always succeeds.

### Outlines — Constrained Generation for Open-Source Models

**Outlines** provides 100% format guarantee by constraining the generation process itself. Unlike Instructor (which validates after generation), Outlines modifies the token probabilities *during* generation so that only schema-valid tokens can be produced.

This approach requires direct access to model logits, making it suitable for locally hosted open-source models:

```python
import outlines
from pydantic import BaseModel, Field
from typing import List

# Define schema
class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(ge=0.0, le=10.0)
    genres: List[str]
    summary: str
    recommend: bool

# Load a local model
model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

# Create a generator constrained to the Pydantic schema
generator = outlines.generate.json(model, MovieReview)

# Generate — output is GUARANTEED to be valid JSON matching the schema
review = generator(
    "Write a review for the movie 'Inception' by Christopher Nolan."
)
print(review)
# MovieReview(title='Inception', rating=9.2, genres=['Sci-Fi', 'Thriller'],
#             summary='...', recommend=True)
```

Outlines also supports regex-constrained generation for arbitrary format patterns:

```python
# Generate a phone number in international format
phone_generator = outlines.generate.regex(
    model,
    r"\+[1-9]\d{0,2}\s?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,9}"
)
phone = phone_generator("Generate a US phone number:")
# "+1 (555) 123-4567"

# Generate a date in ISO format
date_generator = outlines.generate.regex(
    model,
    r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
)
date = date_generator("Generate today's date:")
# "2025-03-15"
```

The mechanism is conceptually simple but technically sophisticated: at each generation step, Outlines computes which tokens are valid continuations given the current partial output and the target schema/regex. Invalid tokens receive zero probability mass. The result is that every generated output is, by construction, valid — no retries needed.

The trade-off is speed: constrained decoding adds overhead at each generation step. For open-source models where you control the inference stack, this trade-off is often worthwhile. For API models where you cannot access logits, use Instructor or function calling instead.

---

## Framework Comparison: DSPy vs Guidance vs LMQL

| Dimension | DSPy | Guidance | LMQL |
|---|---|---|---|
| **Primary focus** | Prompt optimization | Controlled generation | Declarative queries |
| **Abstraction level** | Pipeline / module | Token / template | Query / constraint |
| **Optimization** | Automatic (teleprompters) | Manual | Manual |
| **Format guarantees** | Via assertions | Via token masking | Via constraint clauses |
| **Model support** | Any (API + local) | API + select local | API + select local |
| **Learning curve** | Moderate | Low-moderate | Low-moderate |
| **Best for** | Multi-step pipelines that need systematic improvement | Fine-grained output control, structured extraction | Constrained single-step generation |
| **Composability** | High (module nesting) | Medium (template nesting) | Low (single queries) |
| **Production readiness** | High | Medium | Medium |

**When to use each:**

- **DSPy** when you have a multi-step pipeline, training data, and want automatic optimization. The upfront cost of defining signatures and collecting evaluation data pays off through systematic, reproducible improvement.
- **Guidance** when you need fine-grained control over the generation format — extracting structured fields, enforcing regex patterns, or building complex conditional generation flows.
- **LMQL** when you want a clean, declarative way to express constrained generation with multiple output variables and formal constraints.

These tools are not mutually exclusive. A production system might use DSPy for pipeline-level optimization, Guidance for format-sensitive extraction steps within that pipeline, and Instructor for structured output from API models.

---

## Key Insights

> **DSPy vs Manual Prompting:** DSPy offers four fundamental advantages over hand-crafted prompts: **(1) Automatic optimization** — prompts are tuned by algorithms, not humans. **(2) Modularity** — swap the underlying LLM without rewriting prompts; just re-optimize. **(3) Reproducibility** — optimization is driven by explicit metrics, not subjective judgments. **(4) Scalability** — optimize complex multi-step pipelines that would be intractable to tune manually. DSPy is **not** suitable when: you have fewer than 20 training examples, you are building one-off prompts that do not justify the setup cost, or you need exact control over prompt wording (e.g., for regulatory compliance).

> **Ensuring Structured Output:** Use a multi-level approach: **(1) Instructor + Pydantic** for API models — highest developer ergonomics, automatic retries, production-proven. **(2) JSON mode** for simple structures where a full Pydantic schema is overkill. **(3) Outlines** for open-source models where you need 100% format guarantee without retries. **(4) Post-hoc validation** as a fallback — validate and retry if the primary mechanism fails. Layer these approaches: Instructor as the primary mechanism, with post-hoc validation as a safety net.

> **Constrained Decoding vs Post-hoc Validation:** These represent fundamentally different philosophies. **Constrained decoding** (Outlines, Guidance) restricts the LLM *during* generation — invalid tokens receive zero probability mass. The format guarantee is 100% by construction. No retries needed, but generation is slower due to per-token constraint checking. **Post-hoc validation** (Instructor, manual parsing) validates *after* generation — the model generates freely, and the output is checked against the schema. May require retries, but generation itself is faster. For API models where logits are inaccessible, use function calling + retry (Instructor). For local models where reliability is paramount, use constrained decoding (Outlines).

---

## References

- **DSPy Documentation:** [https://dspy-docs.vercel.app/](https://dspy-docs.vercel.app/)
- **DSPy GitHub:** [https://github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- **DSPy Paper:** Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" — [https://arxiv.org/abs/2310.03714](https://arxiv.org/abs/2310.03714)
- **Guidance:** [https://github.com/guidance-ai/guidance](https://github.com/guidance-ai/guidance)
- **LMQL:** [https://lmql.ai/](https://lmql.ai/)
- **Instructor:** [https://python.useinstructor.com/](https://python.useinstructor.com/)
- **Outlines:** [https://github.com/outlines-dev/outlines](https://github.com/outlines-dev/outlines)
