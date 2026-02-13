# Chapter 7: Python Mastery for Agent Systems

> *"Python is executable pseudocode."* — Bruce Eckel

---

Building production agent systems demands more than surface-level Python. You need to understand how the language itself works at a deep level — how classes are created, how attribute access is resolved, how the async event loop schedules coroutines. This chapter covers the advanced Python constructs that underpin every serious agent framework: metaclasses, descriptors, the MRO, decorators, and async patterns. These are not academic curiosities — they are the machinery behind LangChain's tool registration, Pydantic's validation, and every async agent runner in production today.

---

## Metaclasses

A metaclass is the class of a class. Just as an object is an instance of a class, a class is an instance of a metaclass. By default, that metaclass is `type`. Understanding metaclasses gives you the power to intercept and customize class creation itself — a pattern used extensively in ORMs, plugin systems, and agent registries.

### How Class Creation Works

When Python encounters a `class` statement, it does not simply define a namespace. It executes a well-defined protocol:

1. The class body is executed as a code block, producing a namespace dictionary.
2. Python calls `type(name, bases, namespace)` to create the class object.
3. The resulting class object is bound to the class name in the enclosing scope.

The built-in `type` serves double duty — as a function to inspect types and as the default metaclass:

```python
# type() as inspector
x = 42
print(type(x))          # <class 'int'>
print(type(int))         # <class 'type'>
print(type(type))        # <class 'type'>  — type is its own metaclass

# type() as class factory
# type(name, bases, namespace) -> new class
Dog = type('Dog', (), {
    'species': 'Canis familiaris',
    'speak': lambda self: 'Woof!'
})

rex = Dog()
print(rex.speak())       # Woof!
print(type(rex))         # <class '__main__.Dog'>
print(type(Dog))         # <class 'type'>
```

This is exactly what happens under the hood when you write a regular `class` statement. The `class` keyword is syntactic sugar for calling the metaclass.

### Creating a Metaclass

To customize class creation, subclass `type` and override `__new__` or `__init__`. The most common production use case is **auto-registration** — maintaining a global registry of classes that inherit from a base. Agent frameworks use this pattern to discover handlers, tools, and plugins without explicit registration.

```python
class RegistryMeta(type):
    """Metaclass that auto-registers all subclasses in a central registry."""

    _registry: dict[str, type] = {}

    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        cls = super().__new__(mcs, name, bases, namespace)
        # Don't register the base class itself
        if bases:
            key = namespace.get('handler_type', name.lower())
            mcs._registry[key] = cls
        return cls

    @classmethod
    def get_handler(mcs, handler_type: str) -> type | None:
        return mcs._registry.get(handler_type)

    @classmethod
    def list_handlers(mcs) -> dict[str, type]:
        return dict(mcs._registry)


class BaseHandler(metaclass=RegistryMeta):
    """Base class for all message handlers."""
    handler_type: str = ""

    def handle(self, message: str) -> str:
        raise NotImplementedError


class EmailHandler(BaseHandler):
    handler_type = "email"

    def handle(self, message: str) -> str:
        return f"Sending email: {message}"


class SMSHandler(BaseHandler):
    handler_type = "sms"

    def handle(self, message: str) -> str:
        return f"Sending SMS: {message}"


class SlackHandler(BaseHandler):
    handler_type = "slack"

    def handle(self, message: str) -> str:
        return f"Posting to Slack: {message}"


# No manual registration — the metaclass did it automatically
print(RegistryMeta.list_handlers())
# {'email': <class 'EmailHandler'>, 'sms': <class 'SMSHandler'>, 'slack': <class 'SlackHandler'>}

# Dynamic dispatch based on channel type
def dispatch_message(channel: str, message: str) -> str:
    handler_cls = RegistryMeta.get_handler(channel)
    if handler_cls is None:
        raise ValueError(f"Unknown channel: {channel}")
    return handler_cls().handle(message)

print(dispatch_message("email", "Hello!"))  # Sending email: Hello!
print(dispatch_message("sms", "Alert!"))    # Sending SMS: Alert!
```

This pattern is how Django discovers models, how Flask discovers blueprints, and how agent frameworks discover tool implementations.

### Singleton via Metaclass

The Singleton pattern ensures exactly one instance of a class exists. Implementing it as a metaclass is the cleanest approach because it is transparent to the class itself — no `__new__` override in the business logic, no decorator wrapper, just inheritance.

```python
class SingletonMeta(type):
    """Thread-safe singleton metaclass."""
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class AgentRuntime(metaclass=SingletonMeta):
    """Global agent runtime — exactly one instance."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.active_agents: list = []
        print(f"Runtime initialized with {model}")


# First call creates the instance
rt1 = AgentRuntime("claude-3-opus")    # Runtime initialized with claude-3-opus
rt2 = AgentRuntime("gpt-4o")          # No output — returns existing instance

print(rt1 is rt2)         # True
print(rt2.model)           # claude-3-opus  — original init preserved
```

The key insight is overriding `__call__` on the metaclass, which intercepts the `ClassName()` invocation before `__init__` ever runs on subsequent calls.

### `__init_subclass__` --- Lightweight Alternative

Python 3.6 introduced `__init_subclass__` as a simpler mechanism for the most common metaclass use cases. It is a hook called on the **parent** class whenever a new subclass is defined — no metaclass required.

```python
class Plugin:
    """Base class with automatic plugin registration."""
    _plugins: dict[str, type] = {}

    def __init_subclass__(cls, plugin_name: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        name = plugin_name or cls.__name__.lower()
        cls._plugins[name] = cls
        cls._plugin_name = name

    @classmethod
    def get_plugin(cls, name: str) -> type | None:
        return cls._plugins.get(name)

    @classmethod
    def list_plugins(cls) -> list[str]:
        return list(cls._plugins.keys())


class MarkdownParser(Plugin, plugin_name="markdown"):
    def parse(self, text: str) -> dict:
        return {"type": "markdown", "content": text}


class JSONParser(Plugin, plugin_name="json"):
    def parse(self, text: str) -> dict:
        import json
        return json.loads(text)


class XMLParser(Plugin, plugin_name="xml"):
    def parse(self, text: str) -> dict:
        return {"type": "xml", "content": text}


print(Plugin.list_plugins())          # ['markdown', 'json', 'xml']
parser = Plugin.get_plugin("json")()
print(parser.parse('{"key": "value"}'))  # {'key': 'value'}
```

**When to use which:**
- **`__init_subclass__`** — simple registration, validation, setting defaults. Covers 90% of real-world cases.
- **Metaclasses** — when you need to modify the class namespace before the class is created, control `__new__`, or intercept `__call__`.

---

## Descriptors

Descriptors are the mechanism behind `property`, `classmethod`, `staticmethod`, `__slots__`, and Pydantic's field validation. Any object that defines `__get__`, `__set__`, or `__delete__` is a descriptor. When that object is a class attribute, Python's attribute access machinery invokes the descriptor protocol instead of returning the object directly.

### The Descriptor Protocol

```python
from typing import Any


class TypedField:
    """A descriptor that enforces type checking on assignment."""

    def __init__(self, expected_type: type, default: Any = None):
        self.expected_type = expected_type
        self.default = default
        self.attr_name = None  # set by __set_name__

    def __set_name__(self, owner: type, name: str):
        """Called at class creation time. Receives the attribute name."""
        self.attr_name = name

    def __get__(self, obj: Any, objtype: type = None) -> Any:
        if obj is None:
            return self  # Class-level access returns the descriptor itself
        return obj.__dict__.get(self.attr_name, self.default)

    def __set__(self, obj: Any, value: Any):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.attr_name} must be {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        obj.__dict__[self.attr_name] = value

    def __repr__(self):
        return f"TypedField({self.expected_type.__name__}, default={self.default!r})"


class Config:
    model: str = TypedField(str, default="gpt-4o")
    temperature: float = TypedField(float, default=0.7)
    max_tokens: int = TypedField(int, default=4096)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


config = Config(model="claude-3-opus", temperature=0.9, max_tokens=8192)
print(config.model)          # claude-3-opus
print(config.temperature)    # 0.9

try:
    config.temperature = "hot"  # TypeError: temperature must be float, got str
except TypeError as e:
    print(e)
```

The `__set_name__` method (Python 3.6+) is called automatically when the descriptor is assigned as a class attribute. It eliminates the need to pass the field name explicitly — the descriptor discovers its own name.

### Data Descriptor vs Non-data Descriptor

Python distinguishes between two kinds of descriptors, and the distinction determines attribute lookup priority:

1. **Data descriptor** — defines `__get__` AND (`__set__` or `__delete__`)
2. **Non-data descriptor** — defines only `__get__`

The lookup order for `obj.attr` is:

1. **Data descriptors** on the class (highest priority)
2. The instance's `__dict__`
3. **Non-data descriptors** on the class (lowest priority)

This is why `property` (a data descriptor) always wins over instance attributes, while regular methods (non-data descriptors) can be shadowed by instance attributes.

```python
class NonDataDesc:
    """Only __get__ — can be overridden by instance __dict__."""
    def __get__(self, obj, objtype=None):
        return "from non-data descriptor"


class DataDesc:
    """__get__ + __set__ — always wins over instance __dict__."""
    def __get__(self, obj, objtype=None):
        return "from data descriptor"

    def __set__(self, obj, value):
        print(f"DataDesc.__set__ intercepted: {value!r}")
        # Intentionally not storing — demonstrating interception


class MyClass:
    non_data = NonDataDesc()
    data = DataDesc()


obj = MyClass()

# Non-data descriptor: instance dict wins
print(obj.non_data)              # from non-data descriptor
obj.__dict__['non_data'] = 42
print(obj.non_data)              # 42  ← instance dict overrides

# Data descriptor: descriptor always wins
print(obj.data)                  # from data descriptor
obj.__dict__['data'] = 42        # This goes into __dict__ but is ignored
print(obj.data)                  # from data descriptor  ← descriptor wins
obj.data = 99                    # DataDesc.__set__ intercepted: 99
```

### How property Works Internally

The built-in `property` is simply a data descriptor. Here is a minimal reimplementation that reveals its internals:

```python
class property_reimpl:
    """Reimplementation of the built-in property descriptor."""

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc or (fget.__doc__ if fget else None)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("property has no getter")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("property has no setter")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("property has no deleter")
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)


# Usage — identical to built-in property
class Circle:
    def __init__(self, radius: float):
        self._radius = radius

    @property_reimpl
    def radius(self) -> float:
        """The circle's radius."""
        return self._radius

    @radius.setter
    def radius(self, value: float):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property_reimpl
    def area(self) -> float:
        import math
        return math.pi * self._radius ** 2


c = Circle(5)
print(c.radius)    # 5
print(c.area)      # 78.539...
c.radius = 10
print(c.area)      # 314.159...
```

The decorator syntax `@property_reimpl` is equivalent to `radius = property_reimpl(radius)` — it replaces the function with a descriptor that calls the function on attribute access.

---

## Multiple Inheritance and MRO

Python supports multiple inheritance, which introduces the question: when a method is called on an object whose class has multiple parents, in what order are classes searched? The answer is the **Method Resolution Order (MRO)**, computed using the **C3 linearization algorithm**.

### C3 Linearization

C3 linearization produces a linear ordering of classes that respects two constraints:
1. **Monotonicity** — if class A appears before class B in the MRO of class C, then A appears before B in the MRO of every subclass of C.
2. **Local precedence** — the order in which bases are listed in the `class` statement is preserved.

Consider the following hierarchy:

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass
```

The MRO of `D` is computed step by step:

```
L[A] = [A, object]
L[B] = [B] + merge(L[A], [A])
     = [B] + merge([A, object], [A])
     = [B, A] + merge([object], [])
     = [B, A, object]

L[C] = [C, A, object]

L[D] = [D] + merge(L[B], L[C], [B, C])
     = [D] + merge([B, A, object], [C, A, object], [B, C])

Step 1: B is head of first and last list, not in tail of any → take B
     = [D, B] + merge([A, object], [C, A, object], [C])

Step 2: A is head of first list, but A is in tail of second → skip
         C is head of second list, not in tail of any → take C
     = [D, B, C] + merge([A, object], [A, object])

Step 3: A is head, not in tail → take A
     = [D, B, C, A] + merge([object], [object])

Step 4: object → take object
     = [D, B, C, A, object]
```

```python
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

d = D()
print(d.method())  # "B" — B comes before C in MRO
```

### super() with Multiple Inheritance

A critical insight: `super()` does **not** call the parent class. It calls the **next class in the MRO**. This enables **cooperative multiple inheritance**, where each class in the chain calls `super()` to pass control along the MRO.

```python
class Loggable:
    def setup(self, **kwargs):
        print(f"  Loggable.setup()")
        super().setup(**kwargs)

    def log(self, message: str):
        print(f"[LOG] {message}")


class Serializable:
    def setup(self, **kwargs):
        print(f"  Serializable.setup()")
        super().setup(**kwargs)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class Base:
    def setup(self, **kwargs):
        print(f"  Base.setup()")
        # End of MRO chain — do not call super().setup()
        for key, value in kwargs.items():
            setattr(self, key, value)


class Model(Loggable, Serializable, Base):
    def setup(self, **kwargs):
        print(f"  Model.setup()")
        super().setup(**kwargs)


m = Model()
print(f"MRO: {[c.__name__ for c in Model.__mro__]}")
# MRO: ['Model', 'Loggable', 'Serializable', 'Base', 'object']

m.setup(name="agent-1", version="2.0")
# Output:
#   Model.setup()
#   Loggable.setup()
#   Serializable.setup()
#   Base.setup()

print(m.to_dict())  # {'name': 'agent-1', 'version': '2.0'}
m.log("Ready")      # [LOG] Ready
```

The call chain follows the MRO exactly: `Model -> Loggable -> Serializable -> Base`. Each `super().setup()` call invokes the **next** class in the MRO, not the direct parent.

### Mixin Pattern

Mixins are small, focused classes designed for composition through multiple inheritance. They add specific capabilities without being standalone base classes. This pattern is ubiquitous in agent frameworks.

```python
import json
from datetime import datetime, timezone


class JSONMixin:
    """Adds JSON serialization/deserialization."""

    def to_json(self, indent: int = 2) -> str:
        data = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, datetime):
                    data[key] = value.isoformat()
                else:
                    data[key] = value
        return json.dumps(data, indent=indent)

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        instance = cls.__new__(cls)
        instance.__dict__.update(data)
        return instance


class TimestampMixin:
    """Adds automatic created_at and updated_at tracking."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__ if hasattr(cls, '__init__') else None

        def new_init(self, *args, **kw):
            self.created_at = datetime.now(timezone.utc)
            self.updated_at = datetime.now(timezone.utc)
            if original_init and original_init is not object.__init__:
                original_init(self, *args, **kw)

        cls.__init__ = new_init

    def touch(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)


class User(JSONMixin, TimestampMixin):
    def __init__(self, name: str, email: str, role: str = "user"):
        self.name = name
        self.email = email
        self.role = role


user = User("Alice", "alice@example.com", role="admin")
print(user.to_json())
# {
#   "created_at": "2025-01-15T10:30:00+00:00",
#   "updated_at": "2025-01-15T10:30:00+00:00",
#   "name": "Alice",
#   "email": "alice@example.com",
#   "role": "admin"
# }

user.touch()
print(f"Updated: {user.updated_at}")
```

---

## Advanced Decorators

### Decorator Factory

A decorator factory is a function that returns a decorator. It allows parametrization — passing arguments to control the decorator's behavior. The most production-relevant example for agent systems is a `retry` decorator that handles both synchronous and asynchronous functions.

```python
import asyncio
import functools
import random
import time
from typing import Callable, TypeVar

T = TypeVar('T')


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    jitter: bool = True,
):
    """Retry decorator factory supporting both sync and async functions.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier applied to delay after each retry.
        exceptions: Tuple of exception types to catch.
        jitter: Whether to add random jitter to delay.
    """
    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                    wait = current_delay + (random.uniform(0, current_delay) if jitter else 0)
                    print(f"[retry] {func.__name__} attempt {attempt} failed: {e}. "
                          f"Retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                    current_delay *= backoff
            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                    wait = current_delay + (random.uniform(0, current_delay) if jitter else 0)
                    print(f"[retry] {func.__name__} attempt {attempt} failed: {e}. "
                          f"Retrying in {wait:.1f}s...")
                    time.sleep(wait)
                    current_delay *= backoff
            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# --- Usage ---

@retry(max_attempts=3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
def call_api(endpoint: str) -> dict:
    """Sync API call with retries."""
    if random.random() < 0.7:
        raise ConnectionError("Connection refused")
    return {"status": "ok", "endpoint": endpoint}


@retry(max_attempts=5, delay=1.0, backoff=2.0, exceptions=(TimeoutError,))
async def call_llm(prompt: str) -> str:
    """Async LLM call with retries."""
    if random.random() < 0.5:
        raise TimeoutError("LLM timeout")
    return f"Response to: {prompt}"
```

### Class Decorator

A class decorator receives a class and returns a (modified) class. It is a lighter alternative to metaclasses for simple transformations.

```python
def singleton(cls):
    """Class decorator that makes a class a singleton."""
    instances = {}
    original_init = cls.__init__

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instance = object.__new__(cls)
            original_init(instance, *args, **kwargs)
            instances[cls] = instance
        return instances[cls]

    get_instance._class = cls       # Preserve reference to original class
    get_instance._instances = instances
    return get_instance


@singleton
class ConnectionPool:
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.active = 0
        print(f"Pool created with {max_connections} connections")


pool1 = ConnectionPool(20)    # Pool created with 20 connections
pool2 = ConnectionPool(50)    # No output — returns existing instance
print(pool1 is pool2)         # True
print(pool2.max_connections)  # 20
```

---

## `__getattr__` vs `__getattribute__`

These two hooks control attribute access, but they work at very different levels:

- **`__getattribute__`** — called on **every** attribute access, unconditionally. Overriding it is dangerous and rarely needed.
- **`__getattr__`** — called **only when normal lookup fails** (attribute not found in instance `__dict__`, class, or MRO). This is the safe hook for implementing fallback behavior.

A common agent pattern is a lazy-loading API client that defers endpoint discovery:

```python
import time


class LazyAPIClient:
    """API client that lazily loads and caches endpoint methods."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._cache: dict[str, callable] = {}
        self._endpoints: dict[str, str] | None = None

    def _discover_endpoints(self) -> dict[str, str]:
        """Simulate endpoint discovery from API schema."""
        print("Discovering endpoints...")
        time.sleep(0.1)  # Simulate network call
        return {
            "get_user": "/api/v1/users/{id}",
            "list_agents": "/api/v1/agents",
            "create_task": "/api/v1/tasks",
            "get_metrics": "/api/v1/metrics",
        }

    def __getattr__(self, name: str):
        # Only called when normal lookup fails
        if name.startswith('_'):
            raise AttributeError(f"No attribute {name!r}")

        # Lazy-load endpoints on first access
        if self._endpoints is None:
            self._endpoints = self._discover_endpoints()

        if name not in self._endpoints:
            raise AttributeError(
                f"Unknown endpoint: {name!r}. "
                f"Available: {list(self._endpoints.keys())}"
            )

        endpoint = self._endpoints[name]

        def method(**kwargs) -> dict:
            url = self.base_url + endpoint
            for key, value in kwargs.items():
                url = url.replace(f"{{{key}}}", str(value))
            return {"url": url, "method": "GET", "params": kwargs}

        # Cache the method on the instance — __getattr__ won't be called again
        self.__dict__[name] = method
        return method


client = LazyAPIClient("https://api.example.com")

# First call triggers endpoint discovery
result = client.get_user(id=123)
print(result)
# {'url': 'https://api.example.com/api/v1/users/123', 'method': 'GET', 'params': {'id': 123}}

# Second call uses cached method — no discovery
result = client.list_agents()
print(result)
# {'url': 'https://api.example.com/api/v1/agents', 'method': 'GET', 'params': {}}
```

The key technique here is **self-caching**: after `__getattr__` creates the method, it stores it in `self.__dict__`, which means subsequent accesses find it through normal lookup and never hit `__getattr__` again.

---

## Pydantic v2 for Agent Configuration

Pydantic v2 (built on pydantic-core in Rust) is the standard for data validation in Python agent systems. It provides type-safe configuration, serialization, and validation with excellent performance.

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class RetryConfig(BaseModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    base_delay: float = Field(default=1.0, gt=0)
    backoff_factor: float = Field(default=2.0, ge=1.0)


class AgentConfig(BaseModel):
    """Configuration for an LLM agent with full validation."""

    name: str = Field(..., min_length=1, max_length=100, description="Agent name")
    provider: ModelProvider = ModelProvider.OPENAI
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128_000)
    system_prompt: str = Field(default="You are a helpful assistant.")
    tools: list[str] = Field(default_factory=list)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    cost_budget_usd: float = Field(default=10.0, gt=0)

    @field_validator('model')
    @classmethod
    def validate_model(cls, v: str, info) -> str:
        provider_models = {
            ModelProvider.OPENAI: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'o1', 'o1-mini'],
            ModelProvider.ANTHROPIC: [
                'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku',
                'claude-3.5-sonnet', 'claude-3.5-haiku',
            ],
            ModelProvider.LOCAL: [],  # Any model name allowed
        }
        provider = info.data.get('provider', ModelProvider.OPENAI)
        allowed = provider_models.get(provider, [])
        if allowed and v not in allowed:
            raise ValueError(
                f"Model '{v}' not available for {provider.value}. "
                f"Choose from: {allowed}"
            )
        return v

    @model_validator(mode='after')
    def validate_config(self) -> 'AgentConfig':
        """Cross-field validation."""
        if self.temperature > 1.0 and 'code_generation' in self.tools:
            raise ValueError(
                "Temperature >1.0 is not recommended for code generation tasks"
            )
        if self.max_tokens > 16_000 and self.cost_budget_usd < 5.0:
            raise ValueError(
                "High max_tokens with low budget — increase cost_budget_usd or reduce max_tokens"
            )
        return self


# --- Usage ---

config = AgentConfig(
    name="support-agent",
    provider=ModelProvider.ANTHROPIC,
    model="claude-3.5-sonnet",
    temperature=0.3,
    max_tokens=8192,
    tools=["search", "calculator", "database_query"],
    retry=RetryConfig(max_attempts=5),
    cost_budget_usd=25.0,
)

print(config.model_dump_json(indent=2))
# {
#   "name": "support-agent",
#   "provider": "anthropic",
#   "model": "claude-3.5-sonnet",
#   "temperature": 0.3,
#   "max_tokens": 8192,
#   "system_prompt": "You are a helpful assistant.",
#   "tools": ["search", "calculator", "database_query"],
#   "retry": {"max_attempts": 5, "base_delay": 1.0, "backoff_factor": 2.0},
#   "cost_budget_usd": 25.0
# }

# Validation in action
try:
    bad_config = AgentConfig(
        name="test",
        provider=ModelProvider.OPENAI,
        model="nonexistent-model",
    )
except Exception as e:
    print(f"Validation error: {e}")
    # Model 'nonexistent-model' not available for openai.
```

---

## Async Programming for Agents

Agent systems are inherently I/O-bound: they wait for LLM responses, database queries, tool executions, and API calls. Asynchronous programming with `asyncio` enables a single thread to manage thousands of concurrent operations efficiently.

### asyncio Fundamentals

The core pattern for agent systems is controlled concurrency — running multiple agents or tool calls in parallel while limiting the number of simultaneous operations to avoid rate limits and resource exhaustion.

```python
import asyncio
import time
from typing import Any


async def call_agent(agent_id: str, query: str, semaphore: asyncio.Semaphore) -> dict:
    """Simulate an agent call with rate limiting."""
    async with semaphore:
        print(f"[{agent_id}] Starting: {query[:40]}...")
        await asyncio.sleep(1.0)  # Simulate LLM latency
        return {
            "agent_id": agent_id,
            "query": query,
            "response": f"Answer from {agent_id}",
            "tokens_used": 150,
        }


async def parallel_agents(
    queries: list[dict[str, str]],
    max_concurrent: int = 5,
) -> list[dict]:
    """Run multiple agent calls in parallel with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        call_agent(q["agent_id"], q["query"], semaphore)
        for q in queries
    ]

    start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter() - start

    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]

    print(f"Completed {len(successful)}/{len(results)} in {elapsed:.1f}s "
          f"({len(failed)} failures)")

    return successful


# --- Usage ---
async def main():
    queries = [
        {"agent_id": f"agent-{i}", "query": f"Process request {i}"}
        for i in range(20)
    ]
    results = await parallel_agents(queries, max_concurrent=5)
    print(f"Total tokens: {sum(r['tokens_used'] for r in results)}")


# asyncio.run(main())
```

**Python 3.11+ TaskGroup** provides structured concurrency with better error handling. If any task in the group raises an exception, all other tasks are cancelled:

```python
async def process_with_taskgroup(queries: list[dict]) -> list[dict]:
    """Structured concurrency with TaskGroup (Python 3.11+)."""
    results: list[dict] = []
    semaphore = asyncio.Semaphore(5)

    async def bounded_call(query: dict):
        result = await call_agent(query["agent_id"], query["query"], semaphore)
        results.append(result)

    try:
        async with asyncio.TaskGroup() as tg:
            for query in queries:
                tg.create_task(bounded_call(query))
    except* TimeoutError as eg:
        print(f"Some tasks timed out: {len(eg.exceptions)} failures")
    except* Exception as eg:
        print(f"Errors: {[str(e) for e in eg.exceptions]}")

    return results
```

The key difference: `gather` collects all results (including exceptions if `return_exceptions=True`), while `TaskGroup` cancels all tasks on first unhandled exception, enforcing an all-or-nothing semantics.

### Async Iterators for LLM Streaming

LLM APIs return tokens incrementally. Async iterators provide a clean abstraction for consuming streamed responses:

```python
import asyncio
import random
from dataclasses import dataclass, field


@dataclass
class StreamChunk:
    token: str
    finish_reason: str | None = None
    latency_ms: float = 0.0


class LLMStreamIterator:
    """Async iterator for streaming LLM responses."""

    def __init__(self, prompt: str, model: str = "gpt-4o"):
        self.prompt = prompt
        self.model = model
        self._buffer: asyncio.Queue[StreamChunk | None] = asyncio.Queue()
        self._started = False
        self._total_tokens = 0

    async def _generate(self):
        """Simulate token-by-token LLM generation."""
        tokens = f"Based on your query about '{self.prompt[:30]}', here is a detailed response with multiple tokens.".split()

        for i, token in enumerate(tokens):
            await asyncio.sleep(random.uniform(0.02, 0.08))  # Simulate latency
            chunk = StreamChunk(
                token=token + " ",
                finish_reason=None,
                latency_ms=random.uniform(20, 80),
            )
            await self._buffer.put(chunk)

        # Signal completion
        await self._buffer.put(StreamChunk(
            token="", finish_reason="stop", latency_ms=0
        ))
        await self._buffer.put(None)  # Sentinel

    def __aiter__(self):
        if not self._started:
            asyncio.create_task(self._generate())
            self._started = True
        return self

    async def __anext__(self) -> StreamChunk:
        chunk = await self._buffer.get()
        if chunk is None:
            raise StopAsyncIteration
        self._total_tokens += 1
        return chunk

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


async def stream_response(prompt: str):
    """Consume a streaming LLM response."""
    stream = LLMStreamIterator(prompt)
    full_response = []

    async for chunk in stream:
        if chunk.finish_reason == "stop":
            break
        full_response.append(chunk.token)
        print(chunk.token, end="", flush=True)

    print(f"\n--- {stream.total_tokens} tokens ---")
    return "".join(full_response)
```

### Threading vs Multiprocessing vs Asyncio

Choosing the right concurrency model depends on the nature of the workload:

| Aspect | `threading` | `multiprocessing` | `asyncio` |
|---|---|---|---|
| **Best for** | I/O-bound with blocking libs | CPU-bound computation | I/O-bound with async libs |
| **GIL** | Limited by GIL | Bypasses GIL (separate processes) | Single-threaded, no GIL issue |
| **Overhead** | Medium (OS threads) | High (process forking) | Low (coroutines) |
| **Shared state** | Easy (shared memory) | Hard (IPC required) | Easy (single thread) |
| **Scaling** | ~100s of threads | ~10s of processes | ~100,000s of coroutines |
| **Debugging** | Hard (race conditions) | Medium (isolated) | Easier (sequential within coroutines) |
| **Agent use case** | Legacy sync SDK wrappers | Embedding computation, batch ML | LLM calls, API calls, tool execution |

In practice, production agent systems use a **hybrid** approach:

```python
import asyncio
import concurrent.futures
import time
from functools import partial


def cpu_intensive_embedding(texts: list[str]) -> list[list[float]]:
    """Simulate CPU-intensive embedding computation."""
    time.sleep(0.5)  # Simulate computation
    return [[0.1 * i] * 384 for i, _ in enumerate(texts)]


async def io_intensive_llm_call(prompt: str) -> str:
    """Simulate async LLM API call."""
    await asyncio.sleep(1.0)
    return f"Response to: {prompt}"


async def hybrid_processing(
    texts: list[str],
    prompts: list[str],
) -> dict:
    """Combine asyncio for I/O with ProcessPoolExecutor for CPU work."""
    loop = asyncio.get_event_loop()

    # CPU-bound work in a process pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        # Split texts into chunks for parallel processing
        chunk_size = max(1, len(texts) // 4)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

        # Run CPU work in processes + I/O work in async tasks concurrently
        embedding_futures = [
            loop.run_in_executor(pool, cpu_intensive_embedding, chunk)
            for chunk in chunks
        ]
        llm_tasks = [io_intensive_llm_call(p) for p in prompts]

        # Wait for both CPU and I/O work
        all_results = await asyncio.gather(
            *embedding_futures,
            *llm_tasks,
            return_exceptions=True,
        )

    n_embed = len(chunks)
    embeddings = []
    for result in all_results[:n_embed]:
        if not isinstance(result, Exception):
            embeddings.extend(result)

    llm_responses = [
        r for r in all_results[n_embed:]
        if not isinstance(r, Exception)
    ]

    return {
        "embeddings_count": len(embeddings),
        "llm_responses_count": len(llm_responses),
    }
```

---

## Key Insights

> **Metaclasses real-world use:** Auto-registration, ORM field mapping, validation enforcement. For Python 3.6+, `__init_subclass__` is often simpler. Reserve full metaclasses for cases where you need to modify the class namespace before the class object is created.

> **MRO and C3 Linearization:** `super()` calls the NEXT class in MRO, not the parent. This enables cooperative multiple inheritance. Always use `**kwargs` to pass unexpected arguments along the MRO chain, and ensure exactly one class at the end of the chain consumes them.

> **Descriptors are the backbone of Python:** `property`, `classmethod`, `staticmethod`, `__slots__`, and Pydantic fields are all implemented as descriptors. The data vs non-data distinction determines whether the descriptor or the instance `__dict__` takes priority.

> **Async is non-negotiable for agents:** Agent workloads are I/O-bound — LLM calls, API requests, database queries. Asyncio handles tens of thousands of concurrent operations on a single thread. Use `Semaphore` for rate limiting, `TaskGroup` for structured concurrency, and `ProcessPoolExecutor` for CPU-bound work within an async context.

---

## References

- Python Data Model: [https://docs.python.org/3/reference/datamodel.html](https://docs.python.org/3/reference/datamodel.html)
- Descriptor HowTo Guide: [https://docs.python.org/3/howto/descriptor.html](https://docs.python.org/3/howto/descriptor.html)
- asyncio Documentation: [https://docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)
- Pydantic v2 Documentation: [https://docs.pydantic.dev/latest/](https://docs.pydantic.dev/latest/)
- Raymond Hettinger, "Super considered super!": [https://rhettinger.wordpress.com/2011/05/26/super-considered-super/](https://rhettinger.wordpress.com/2011/05/26/super-considered-super/)
- Luciano Ramalho, *Fluent Python*, 2nd Edition, O'Reilly Media, 2022
