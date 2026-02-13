# Chapter 8: Advanced SQL & PostgreSQL

> *"The question of whether a computer can think is no more interesting than the question of whether a submarine can swim."* --- Edsger Dijkstra

---

Agent systems generate, query, and analyze vast amounts of structured data --- conversation logs, tool invocations, performance metrics, user sessions, and audit trails. PostgreSQL is the database of choice for most production agent platforms, and mastering its advanced features is essential. This chapter covers window functions, recursive CTEs, query optimization, JSON operations, full-text search, and lateral joins --- the SQL toolkit that separates prototypes from production systems.

---

## Window Functions

Window functions perform calculations across a set of rows related to the current row, without collapsing the result set. Unlike `GROUP BY`, which reduces rows, window functions **preserve every row** while adding computed columns. They are indispensable for agent analytics: ranking agents, detecting performance trends, computing running totals, and identifying sessions.

### ROW_NUMBER, RANK, DENSE_RANK

These three ranking functions differ in how they handle ties:

- **ROW_NUMBER()** --- assigns a unique sequential integer to each row, regardless of ties.
- **RANK()** --- assigns the same rank to ties, then skips subsequent ranks (1, 2, 2, 4).
- **DENSE_RANK()** --- assigns the same rank to ties, without gaps (1, 2, 2, 3).

```sql
-- Agent performance ranking with all three functions
CREATE TABLE agent_logs (
    id          SERIAL PRIMARY KEY,
    agent_id    VARCHAR(50) NOT NULL,
    task_type   VARCHAR(50) NOT NULL,
    score       NUMERIC(5,2),
    tokens_used INTEGER,
    latency_ms  INTEGER,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Sample data
INSERT INTO agent_logs (agent_id, task_type, score, tokens_used, latency_ms) VALUES
('agent-1', 'support',   0.95, 1200, 850),
('agent-2', 'support',   0.95, 1100, 920),
('agent-3', 'support',   0.88, 1400, 1100),
('agent-4', 'support',   0.88, 900,  750),
('agent-5', 'support',   0.82, 1600, 1300),
('agent-1', 'analysis',  0.91, 2200, 1500),
('agent-2', 'analysis',  0.87, 1800, 1200),
('agent-3', 'analysis',  0.87, 2000, 1350);

-- Compare all three ranking functions
SELECT
    agent_id,
    task_type,
    score,
    ROW_NUMBER() OVER (PARTITION BY task_type ORDER BY score DESC) AS row_num,
    RANK()       OVER (PARTITION BY task_type ORDER BY score DESC) AS rank,
    DENSE_RANK() OVER (PARTITION BY task_type ORDER BY score DESC) AS dense_rank
FROM agent_logs
ORDER BY task_type, score DESC;
```

Result:

```
 agent_id | task_type | score | row_num | rank | dense_rank
----------+-----------+-------+---------+------+------------
 agent-1  | analysis  |  0.91 |       1 |    1 |          1
 agent-2  | analysis  |  0.87 |       2 |    2 |          2
 agent-3  | analysis  |  0.87 |       3 |    2 |          2
 agent-1  | support   |  0.95 |       1 |    1 |          1
 agent-2  | support   |  0.95 |       2 |    1 |          1
 agent-3  | support   |  0.88 |       3 |    3 |          2
 agent-4  | support   |  0.88 |       4 |    3 |          2
 agent-5  | support   |  0.82 |       5 |    5 |          3
```

Notice: for the tied `support` agents with score 0.88, `RANK` assigns 3 (skipping to position 3), `DENSE_RANK` assigns 2 (no gap), and `ROW_NUMBER` assigns distinct values (3 and 4, deterministic only with a tiebreaker).

### LAG/LEAD --- Accessing Adjacent Rows

`LAG(column, offset, default)` accesses a previous row; `LEAD(column, offset, default)` accesses a subsequent row. These are essential for trend analysis --- comparing each data point to the one before it.

```sql
-- Trend analysis: detect performance changes over time
WITH daily_metrics AS (
    SELECT
        agent_id,
        DATE(created_at) AS metric_date,
        AVG(score) AS avg_score,
        AVG(latency_ms) AS avg_latency,
        COUNT(*) AS task_count
    FROM agent_logs
    GROUP BY agent_id, DATE(created_at)
)
SELECT
    agent_id,
    metric_date,
    avg_score,
    LAG(avg_score, 1) OVER w AS prev_day_score,
    avg_score - LAG(avg_score, 1) OVER w AS score_change,
    CASE
        WHEN avg_score > LAG(avg_score, 1) OVER w THEN 'improving'
        WHEN avg_score < LAG(avg_score, 1) OVER w THEN 'degrading'
        ELSE 'stable'
    END AS trend,

    -- Look ahead: what will tomorrow's latency be?
    LEAD(avg_latency, 1) OVER w AS next_day_latency,

    -- 7-day lookback for weekly comparison
    LAG(avg_score, 7, NULL) OVER w AS score_7d_ago

FROM daily_metrics
WINDOW w AS (PARTITION BY agent_id ORDER BY metric_date)
ORDER BY agent_id, metric_date;
```

The `WINDOW` clause defines a named window specification, avoiding repetition when multiple functions use the same partitioning and ordering.

### Running Totals and Moving Averages

Frame specifications control exactly which rows are included in the window calculation. The three frame types behave differently:

- **ROWS** --- counts physical rows relative to the current row.
- **RANGE** --- includes all rows whose ORDER BY value falls within a specified range of the current row's value.
- **GROUPS** --- counts groups of rows with the same ORDER BY value.

```sql
-- Running total of tokens consumed
SELECT
    agent_id,
    created_at,
    tokens_used,
    SUM(tokens_used) OVER (
        PARTITION BY agent_id
        ORDER BY created_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total_tokens
FROM agent_logs
ORDER BY agent_id, created_at;

-- 7-day moving average of scores (using RANGE with dates)
SELECT
    agent_id,
    DATE(created_at) AS metric_date,
    score,
    AVG(score) OVER (
        PARTITION BY agent_id
        ORDER BY DATE(created_at)
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d,

    -- 3-row moving average (using ROWS)
    AVG(score) OVER (
        PARTITION BY agent_id
        ORDER BY created_at
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3rows,

    -- Moving average over 3 groups of tied values (using GROUPS)
    AVG(score) OVER (
        PARTITION BY agent_id
        ORDER BY DATE(created_at)
        GROUPS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3groups

FROM agent_logs
ORDER BY agent_id, created_at;
```

The distinction matters: with `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`, exactly 3 physical rows are included. With `RANGE BETWEEN INTERVAL '7 days' PRECEDING`, all rows within the last 7 calendar days are included --- which could be 0 rows or 100 rows depending on data density.

### Top-N per Group

A classic problem: find the top N performing agents within each task type.

```sql
-- Top 2 agents per task type by score
SELECT *
FROM (
    SELECT
        agent_id,
        task_type,
        score,
        tokens_used,
        latency_ms,
        ROW_NUMBER() OVER (
            PARTITION BY task_type
            ORDER BY score DESC, latency_ms ASC
        ) AS rn
    FROM agent_logs
) ranked
WHERE rn <= 2
ORDER BY task_type, rn;
```

Adding `latency_ms ASC` as a secondary sort ensures deterministic results when scores are tied --- the faster agent wins.

---

## CTEs and Recursive Queries

Common Table Expressions (CTEs) improve readability by breaking complex queries into named, composable steps. Recursive CTEs enable traversal of hierarchical data --- agent trees, organizational structures, dependency graphs.

### Recursive CTE --- Hierarchies

Agent systems often have hierarchical structures: a supervisor agent manages sub-agents, which may themselves manage further sub-agents.

```sql
CREATE TABLE agent_tree (
    agent_id    VARCHAR(50) PRIMARY KEY,
    agent_name  VARCHAR(100) NOT NULL,
    parent_id   VARCHAR(50) REFERENCES agent_tree(agent_id),
    agent_role  VARCHAR(50),
    created_at  TIMESTAMP DEFAULT NOW()
);

INSERT INTO agent_tree VALUES
('root',    'Supervisor',        NULL,       'supervisor',  NOW()),
('support', 'Support Lead',      'root',     'lead',        NOW()),
('sales',   'Sales Lead',        'root',     'lead',        NOW()),
('faq',     'FAQ Agent',         'support',  'worker',      NOW()),
('ticket',  'Ticket Agent',      'support',  'worker',      NOW()),
('escal',   'Escalation Agent',  'support',  'worker',      NOW()),
('outbound','Outbound Agent',    'sales',    'worker',      NOW()),
('inbound', 'Inbound Agent',     'sales',    'worker',      NOW()),
('vip-faq', 'VIP FAQ Specialist','faq',      'specialist',  NOW());

-- Traverse the full hierarchy with depth and path
WITH RECURSIVE agent_hierarchy AS (
    -- Base case: root nodes (no parent)
    SELECT
        agent_id,
        agent_name,
        parent_id,
        agent_role,
        0 AS depth,
        ARRAY[agent_id] AS path,
        agent_name::TEXT AS full_path
    FROM agent_tree
    WHERE parent_id IS NULL

    UNION ALL

    -- Recursive case: join children to their parents
    SELECT
        t.agent_id,
        t.agent_name,
        t.parent_id,
        t.agent_role,
        h.depth + 1,
        h.path || t.agent_id,
        h.full_path || ' > ' || t.agent_name
    FROM agent_tree t
    INNER JOIN agent_hierarchy h ON t.parent_id = h.agent_id
)
SELECT
    REPEAT('  ', depth) || agent_name AS display_name,
    agent_role,
    depth,
    full_path,
    path
FROM agent_hierarchy
ORDER BY path;
```

Result:

```
      display_name       | agent_role  | depth |               full_path
-------------------------+-------------+-------+----------------------------------------
 Supervisor              | supervisor  |     0 | Supervisor
   Sales Lead            | lead        |     1 | Supervisor > Sales Lead
     Inbound Agent       | worker      |     2 | Supervisor > Sales Lead > Inbound Agent
     Outbound Agent      | worker      |     2 | Supervisor > Sales Lead > Outbound Agent
   Support Lead          | lead        |     1 | Supervisor > Support Lead
     Escalation Agent    | worker      |     2 | Supervisor > Support Lead > Escalation Agent
     FAQ Agent           | worker      |     2 | Supervisor > Support Lead > FAQ Agent
       VIP FAQ Specialist| specialist  |     3 | Supervisor > Support Lead > FAQ Agent > VIP FAQ Specialist
     Ticket Agent        | worker      |     2 | Supervisor > Support Lead > Ticket Agent
```

The `ARRAY[agent_id]` column tracks the full path as an array, which is used for sorting (`ORDER BY path`) to produce a natural tree display. The `REPEAT('  ', depth)` creates visual indentation.

### Generating Date Series

When analyzing time-series data, gaps in dates cause misleading results. A date-generating CTE fills those gaps.

```sql
-- Generate a complete date series for gap-free reporting
WITH RECURSIVE dates AS (
    SELECT DATE '2025-01-01' AS d
    UNION ALL
    SELECT d + INTERVAL '1 day'
    FROM dates
    WHERE d < DATE '2025-01-31'
),
daily_counts AS (
    SELECT
        DATE(created_at) AS log_date,
        COUNT(*) AS task_count,
        AVG(score) AS avg_score
    FROM agent_logs
    WHERE created_at >= '2025-01-01' AND created_at < '2025-02-01'
    GROUP BY DATE(created_at)
)
SELECT
    dates.d AS date,
    COALESCE(dc.task_count, 0) AS task_count,
    COALESCE(dc.avg_score, 0) AS avg_score
FROM dates
LEFT JOIN daily_counts dc ON dates.d = dc.log_date
ORDER BY dates.d;
```

PostgreSQL also provides the `generate_series` function, which is more efficient for this specific case:

```sql
SELECT
    d::DATE AS date,
    COALESCE(dc.task_count, 0) AS task_count
FROM generate_series('2025-01-01'::DATE, '2025-01-31'::DATE, '1 day') AS d
LEFT JOIN (
    SELECT DATE(created_at) AS log_date, COUNT(*) AS task_count
    FROM agent_logs
    GROUP BY DATE(created_at)
) dc ON d::DATE = dc.log_date;
```

### Materialized vs Non-materialized CTEs

By default, PostgreSQL may inline CTEs (merge them into the main query) or materialize them (execute once and store results). Since PostgreSQL 12, you can control this explicitly:

```sql
-- Force materialization: compute once, reference multiple times
WITH expensive_stats AS MATERIALIZED (
    SELECT
        agent_id,
        AVG(score) AS avg_score,
        STDDEV(score) AS score_stddev,
        COUNT(*) AS total_tasks
    FROM agent_logs
    GROUP BY agent_id
)
SELECT
    es.agent_id,
    es.avg_score,
    es.score_stddev,
    al.task_type,
    al.score,
    al.score - es.avg_score AS deviation
FROM agent_logs al
JOIN expensive_stats es ON al.agent_id = es.agent_id
WHERE al.score < es.avg_score - es.score_stddev;

-- Force inlining: allow the optimizer to merge CTE into the main query
WITH simple_filter AS NOT MATERIALIZED (
    SELECT * FROM agent_logs WHERE score > 0.9
)
SELECT agent_id, AVG(latency_ms)
FROM simple_filter
GROUP BY agent_id;
```

**When to materialize:** When the CTE is referenced multiple times, or when it acts as an optimization fence (you want to force a specific execution plan). **When to inline:** When the CTE is referenced once and you want the optimizer to push predicates into it.

---

## Query Optimization

### Reading EXPLAIN ANALYZE

`EXPLAIN ANALYZE` executes the query and reports the actual execution plan with real timings. This is the single most important tool for understanding query performance.

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT agent_id, AVG(score), COUNT(*)
FROM agent_logs
WHERE created_at >= '2025-01-01'
  AND task_type = 'support'
GROUP BY agent_id
HAVING COUNT(*) > 10
ORDER BY AVG(score) DESC;
```

Example output:

```
Sort  (cost=245.30..245.80 rows=200 width=48)
      (actual time=12.450..12.460 rows=15 loops=1)
  Sort Key: (avg(score)) DESC
  Sort Method: quicksort  Memory: 26kB
  -> HashAggregate  (cost=230.00..240.00 rows=200 width=48)
                    (actual time=12.100..12.200 rows=15 loops=1)
        Group Key: agent_id
        Filter: (count(*) > 10)
        Rows Removed by Filter: 3
        Batches: 1  Memory Usage: 40kB
        -> Bitmap Heap Scan on agent_logs  (cost=8.50..210.00 rows=5000 width=18)
                                           (actual time=0.120..8.500 rows=4800 loops=1)
              Recheck Cond: (created_at >= '2025-01-01')
              Filter: (task_type = 'support')
              Rows Removed by Filter: 1200
              Heap Blocks: exact=150
              Buffers: shared hit=160
              -> Bitmap Index Scan on idx_logs_created  (cost=0.00..7.25 rows=6000 width=0)
                                                        (actual time=0.080..0.080 rows=6000 loops=1)
                    Index Cond: (created_at >= '2025-01-01')
                    Buffers: shared hit=10
Planning Time: 0.150 ms
Execution Time: 12.600 ms
```

Key metrics to examine:

| Metric | Meaning |
|---|---|
| **cost** (estimated) | Startup cost..total cost in abstract units |
| **actual time** | Real milliseconds: startup..total |
| **rows** | Estimated vs actual row count --- large discrepancy means stale statistics |
| **Buffers: shared hit** | Pages read from cache (good) |
| **Buffers: shared read** | Pages read from disk (slow) |
| **Rows Removed by Filter** | Rows fetched but discarded --- indicates a missing or suboptimal index |
| **loops** | Number of times this node was executed (important in nested loops) |

The most common performance issue is a large gap between estimated and actual rows, which causes the optimizer to choose a suboptimal plan. Fix with `ANALYZE tablename` to refresh statistics.

### Index Types

PostgreSQL offers multiple index types, each optimized for different access patterns:

```sql
-- B-tree: default, best for equality and range queries
-- Supports: =, <, >, <=, >=, BETWEEN, IN, IS NULL
CREATE INDEX idx_logs_created ON agent_logs (created_at);
CREATE INDEX idx_logs_agent_task ON agent_logs (agent_id, task_type);

-- Hash: equality-only lookups, smaller than B-tree
-- Supports: = only. Useful for UUID or text exact match.
CREATE INDEX idx_logs_agent_hash ON agent_logs USING hash (agent_id);

-- GIN (Generalized Inverted Index): best for composite values
-- Supports: array containment, JSONB, full-text search
CREATE INDEX idx_logs_metadata_gin ON agent_logs USING gin (metadata jsonb_path_ops);

-- GiST (Generalized Search Tree): best for geometric/range data
-- Supports: overlap, containment, nearest-neighbor
CREATE INDEX idx_logs_tsrange ON agent_logs USING gist (
    tsrange(created_at, created_at + INTERVAL '1 hour')
);

-- BRIN (Block Range Index): best for naturally ordered large tables
-- Very small index size. Works when data is physically ordered by the indexed column.
CREATE INDEX idx_logs_created_brin ON agent_logs USING brin (created_at)
    WITH (pages_per_range = 32);

-- Partial index: index only the rows you query
CREATE INDEX idx_logs_high_score ON agent_logs (agent_id, score)
    WHERE score >= 0.9;

-- Expression index: index a computed value
CREATE INDEX idx_logs_date ON agent_logs (DATE(created_at));
CREATE INDEX idx_logs_lower_type ON agent_logs (LOWER(task_type));

-- Composite index: multi-column for combined filters
-- Column order matters: put high-selectivity columns first
CREATE INDEX idx_logs_composite ON agent_logs (task_type, agent_id, created_at DESC);
```

**Index selection guide:**

| Use case | Index type |
|---|---|
| Equality + range on scalar columns | B-tree |
| Exact equality only (UUID, hash lookups) | Hash |
| JSONB containment (`@>`) | GIN with `jsonb_path_ops` |
| Full-text search (`@@`) | GIN on `tsvector` |
| Array containment (`@>`, `&&`) | GIN |
| Range overlap, geometric queries | GiST |
| Very large, append-only, time-series | BRIN |
| Queries filtering on a constant predicate | Partial index |
| Queries filtering on a function result | Expression index |

### Partitioning

For tables that grow indefinitely --- such as agent logs --- partitioning is essential for query performance and data management.

```sql
-- Range partitioning by date
CREATE TABLE agent_logs_partitioned (
    id          SERIAL,
    agent_id    VARCHAR(50) NOT NULL,
    task_type   VARCHAR(50) NOT NULL,
    score       NUMERIC(5,2),
    tokens_used INTEGER,
    latency_ms  INTEGER,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create partitions for each month
CREATE TABLE agent_logs_2025_01 PARTITION OF agent_logs_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE agent_logs_2025_02 PARTITION OF agent_logs_partitioned
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE agent_logs_2025_03 PARTITION OF agent_logs_partitioned
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- Default partition catches rows that don't fit any range
CREATE TABLE agent_logs_default PARTITION OF agent_logs_partitioned DEFAULT;

-- Indexes are created per partition
CREATE INDEX ON agent_logs_2025_01 (agent_id, created_at);
CREATE INDEX ON agent_logs_2025_02 (agent_id, created_at);
CREATE INDEX ON agent_logs_2025_03 (agent_id, created_at);

-- Queries automatically target relevant partitions (partition pruning)
EXPLAIN
SELECT * FROM agent_logs_partitioned
WHERE created_at >= '2025-02-01' AND created_at < '2025-03-01';
-- -> Seq Scan on agent_logs_2025_02 (only this partition is scanned)

-- Dropping old data is instant: detach the partition
ALTER TABLE agent_logs_partitioned DETACH PARTITION agent_logs_2025_01;
DROP TABLE agent_logs_2025_01;  -- Instant, no locking main table
```

---

## JSON/JSONB Operations

PostgreSQL's JSONB type is ideal for storing semi-structured agent data --- tool call arguments, LLM responses, metadata, configuration overrides.

```sql
-- Table with JSONB metadata
CREATE TABLE agent_events (
    id          SERIAL PRIMARY KEY,
    agent_id    VARCHAR(50),
    event_type  VARCHAR(50),
    payload     JSONB NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW()
);

INSERT INTO agent_events (agent_id, event_type, payload) VALUES
('agent-1', 'tool_call', '{
    "tool": "search",
    "args": {"query": "system design patterns", "top_k": 5},
    "result": {"hits": 5, "latency_ms": 120},
    "metadata": {"model": "gpt-4o", "tokens": 450}
}'),
('agent-1', 'llm_response', '{
    "model": "gpt-4o",
    "tokens": {"input": 500, "output": 200},
    "finish_reason": "stop",
    "tags": ["support", "technical"]
}');

-- Access nested values with -> (returns JSON) and ->> (returns text)
SELECT
    agent_id,
    payload->>'tool' AS tool_name,
    payload->'args'->>'query' AS search_query,
    (payload->'result'->>'hits')::INTEGER AS hit_count,
    payload->'metadata'->>'model' AS model_used
FROM agent_events
WHERE event_type = 'tool_call';

-- Filter by JSONB content using containment operator @>
SELECT * FROM agent_events
WHERE payload @> '{"tool": "search"}';

-- Access array elements
SELECT
    payload->'tags'->>0 AS first_tag,       -- "support"
    jsonb_array_length(payload->'tags') AS tag_count
FROM agent_events
WHERE payload ? 'tags';   -- ? checks key existence

-- Update JSONB values with jsonb_set
UPDATE agent_events
SET payload = jsonb_set(
    payload,
    '{metadata,processed}',     -- path
    'true'::jsonb               -- new value
)
WHERE agent_id = 'agent-1' AND event_type = 'tool_call';

-- Add a new key to JSONB
UPDATE agent_events
SET payload = payload || '{"reviewed": true}'::jsonb
WHERE id = 1;

-- Remove a key from JSONB
UPDATE agent_events
SET payload = payload - 'reviewed'
WHERE id = 1;

-- GIN index for fast JSONB queries
CREATE INDEX idx_events_payload ON agent_events USING gin (payload jsonb_path_ops);

-- With jsonb_path_ops, these queries use the index:
SELECT * FROM agent_events WHERE payload @> '{"tool": "search"}';
-- Without jsonb_path_ops (plain GIN), these additional operators are supported:
-- ?, ?|, ?&  (key existence)
```

---

## Full-Text Search

PostgreSQL provides built-in full-text search that is often sufficient for agent knowledge bases, eliminating the need for an external search engine.

```sql
-- Add full-text search to a knowledge base table
CREATE TABLE knowledge_base (
    id          SERIAL PRIMARY KEY,
    title       VARCHAR(200) NOT NULL,
    content     TEXT NOT NULL,
    category    VARCHAR(50),
    search_vec  TSVECTOR,  -- pre-computed search vector
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Automatically update search_vec on insert/update
CREATE OR REPLACE FUNCTION update_search_vec() RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vec :=
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_search_vec
    BEFORE INSERT OR UPDATE OF title, content
    ON knowledge_base
    FOR EACH ROW EXECUTE FUNCTION update_search_vec();

-- GIN index for fast full-text search
CREATE INDEX idx_kb_search ON knowledge_base USING gin (search_vec);

-- Search with ranking
SELECT
    id,
    title,
    ts_rank(search_vec, query) AS rank,
    ts_headline('english', content, query,
        'StartSel=<<, StopSel=>>, MaxWords=50, MinWords=20') AS snippet
FROM knowledge_base,
     to_tsquery('english', 'agent & orchestration & !testing') AS query
WHERE search_vec @@ query
ORDER BY rank DESC
LIMIT 10;

-- Phrase search (words must be adjacent)
SELECT * FROM knowledge_base
WHERE search_vec @@ phraseto_tsquery('english', 'multi agent system');

-- Hybrid search: combine full-text score with vector similarity
-- (Assumes pgvector extension with an embedding column)
SELECT
    kb.id,
    kb.title,
    0.7 * ts_rank(kb.search_vec, query) +
    0.3 * (1 - (kb.embedding <=> query_embedding)) AS hybrid_score
FROM knowledge_base kb,
     to_tsquery('english', 'agent & orchestration') AS query,
     (SELECT embedding AS query_embedding FROM get_query_embedding('agent orchestration')) qe
WHERE kb.search_vec @@ query
ORDER BY hybrid_score DESC
LIMIT 10;
```

The weighted `tsvector` (`'A'` for title, `'B'` for content) ensures title matches rank higher than content matches.

---

## LATERAL Joins

A `LATERAL` join allows a subquery to reference columns from preceding tables in the `FROM` clause --- something ordinary subqueries cannot do. This is invaluable for "top-N per group" queries and per-row computed aggregations.

```sql
-- Top 3 best-performing task types per agent
SELECT
    a.agent_id,
    t.task_type,
    t.avg_score,
    t.task_count
FROM (SELECT DISTINCT agent_id FROM agent_logs) a
CROSS JOIN LATERAL (
    SELECT
        task_type,
        AVG(score) AS avg_score,
        COUNT(*) AS task_count
    FROM agent_logs al
    WHERE al.agent_id = a.agent_id   -- references outer table
    GROUP BY task_type
    ORDER BY AVG(score) DESC
    LIMIT 3
) t
ORDER BY a.agent_id, t.avg_score DESC;

-- Per-agent running statistics (more efficient than window for complex computations)
SELECT
    a.agent_id,
    stats.total_tasks,
    stats.avg_score,
    stats.p95_latency,
    stats.first_task,
    stats.last_task
FROM (SELECT DISTINCT agent_id FROM agent_logs) a
CROSS JOIN LATERAL (
    SELECT
        COUNT(*) AS total_tasks,
        AVG(score) AS avg_score,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency,
        MIN(created_at) AS first_task,
        MAX(created_at) AS last_task
    FROM agent_logs al
    WHERE al.agent_id = a.agent_id
) stats
ORDER BY stats.avg_score DESC;
```

`LATERAL` is more efficient than window functions for top-N per group when N is small, because PostgreSQL can use an index to fetch just the top rows per group rather than computing a window over the entire partition.

---

## Practice Problems

### Problem 1: Agent Performance Degradation

Find agents whose average weekly score has degraded by more than 20% compared to the previous week.

```sql
WITH weekly_stats AS (
    SELECT
        agent_id,
        DATE_TRUNC('week', created_at) AS week_start,
        AVG(score) AS avg_score,
        COUNT(*) AS task_count,
        AVG(latency_ms) AS avg_latency
    FROM agent_logs
    WHERE created_at >= NOW() - INTERVAL '8 weeks'
    GROUP BY agent_id, DATE_TRUNC('week', created_at)
),
with_prev AS (
    SELECT
        agent_id,
        week_start,
        avg_score,
        task_count,
        avg_latency,
        LAG(avg_score) OVER (
            PARTITION BY agent_id ORDER BY week_start
        ) AS prev_week_score
    FROM weekly_stats
)
SELECT
    agent_id,
    week_start,
    ROUND(avg_score::NUMERIC, 4) AS current_score,
    ROUND(prev_week_score::NUMERIC, 4) AS previous_score,
    ROUND(
        ((avg_score - prev_week_score) / NULLIF(prev_week_score, 0) * 100)::NUMERIC,
        2
    ) AS pct_change,
    task_count,
    ROUND(avg_latency::NUMERIC, 0) AS avg_latency_ms
FROM with_prev
WHERE prev_week_score IS NOT NULL
  AND (avg_score - prev_week_score) / NULLIF(prev_week_score, 0) < -0.20
ORDER BY pct_change ASC;
```

### Problem 2: Session Detection

Identify user sessions from event data, where a new session starts after a 30-minute gap of inactivity.

```sql
WITH events_with_gap AS (
    SELECT
        user_id,
        event_type,
        created_at,
        LAG(created_at) OVER (
            PARTITION BY user_id ORDER BY created_at
        ) AS prev_event_at,
        EXTRACT(EPOCH FROM (
            created_at - LAG(created_at) OVER (
                PARTITION BY user_id ORDER BY created_at
            )
        )) / 60.0 AS gap_minutes
    FROM user_events
),
session_starts AS (
    SELECT
        *,
        CASE
            WHEN gap_minutes IS NULL THEN 1          -- First event = new session
            WHEN gap_minutes > 30 THEN 1             -- Gap > 30 min = new session
            ELSE 0
        END AS is_new_session
    FROM events_with_gap
),
sessions AS (
    SELECT
        *,
        SUM(is_new_session) OVER (
            PARTITION BY user_id
            ORDER BY created_at
        ) AS session_id
    FROM session_starts
)
SELECT
    user_id,
    session_id,
    MIN(created_at) AS session_start,
    MAX(created_at) AS session_end,
    COUNT(*) AS event_count,
    EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / 60.0 AS duration_minutes,
    ARRAY_AGG(DISTINCT event_type) AS event_types
FROM sessions
GROUP BY user_id, session_id
ORDER BY user_id, session_start;
```

The technique: use `LAG` to compute the gap, flag rows where the gap exceeds the threshold, then use a running `SUM` of those flags to assign session IDs.

### Problem 3: Pipeline Funnel Analysis

Analyze a multi-stage agent pipeline and compute drop-off rates at each stage.

```sql
WITH pipeline_events AS (
    SELECT DISTINCT
        conversation_id,
        stage,
        MIN(created_at) OVER (
            PARTITION BY conversation_id, stage
        ) AS entered_at
    FROM pipeline_logs
    WHERE created_at >= NOW() - INTERVAL '30 days'
),
stage_order AS (
    SELECT
        unnest(ARRAY['intake', 'classification', 'processing', 'review', 'completed']) AS stage,
        generate_series(1, 5) AS stage_num
),
funnel AS (
    SELECT
        so.stage,
        so.stage_num,
        COUNT(DISTINCT pe.conversation_id) AS conversations
    FROM stage_order so
    LEFT JOIN pipeline_events pe ON so.stage = pe.stage
    GROUP BY so.stage, so.stage_num
)
SELECT
    stage,
    conversations,
    LAG(conversations) OVER (ORDER BY stage_num) AS prev_stage_count,
    ROUND(
        conversations * 100.0 /
        NULLIF(FIRST_VALUE(conversations) OVER (ORDER BY stage_num), 0),
        1
    ) AS pct_of_initial,
    ROUND(
        conversations * 100.0 /
        NULLIF(LAG(conversations) OVER (ORDER BY stage_num), 0),
        1
    ) AS stage_completion_rate
FROM funnel
ORDER BY stage_num;
```

Example result:

```
    stage      | conversations | prev_stage_count | pct_of_initial | stage_completion_rate
---------------+---------------+------------------+----------------+----------------------
 intake        |         10000 |           (null) |          100.0 |                (null)
 classification|          9200 |            10000 |           92.0 |                 92.0
 processing    |          7800 |             9200 |           78.0 |                 84.8
 review        |          6500 |             7800 |           65.0 |                 83.3
 completed     |          5800 |             6500 |           58.0 |                 89.2
```

---

## Key Insights

> **ROWS vs RANGE:** `ROWS` counts physical rows. `RANGE` counts by values --- all rows with the same `ORDER BY` value form one "range." For dates: `RANGE BETWEEN INTERVAL '7 days' PRECEDING` captures all rows in the last 7 days, regardless of how many rows that is. `GROUPS` counts groups of tied values --- `GROUPS BETWEEN 2 PRECEDING` means "the current group and the two preceding groups."

> **LATERAL JOIN use cases:** When a subquery depends on the current row: top-N per group (more efficient than window functions for small N), computed columns, per-row function calls (e.g., `unnest`, `generate_series`). Think of `LATERAL` as a SQL `for` loop.

> **Index selection:** B-tree handles 80% of cases. Add GIN for JSONB and full-text search. Use BRIN for append-only time-series tables --- the index is orders of magnitude smaller than B-tree. Partial indexes are the most underused optimization: index only the rows you actually query.

> **CTE materialization:** In PostgreSQL 12+, CTEs referenced once may be inlined automatically. Use `MATERIALIZED` when the CTE is expensive and referenced multiple times; use `NOT MATERIALIZED` when you want predicate pushdown into the CTE.

---

## References

- PostgreSQL Window Functions: [https://www.postgresql.org/docs/current/tutorial-window.html](https://www.postgresql.org/docs/current/tutorial-window.html)
- PostgreSQL JSON Functions: [https://www.postgresql.org/docs/current/functions-json.html](https://www.postgresql.org/docs/current/functions-json.html)
- PostgreSQL Full Text Search: [https://www.postgresql.org/docs/current/textsearch.html](https://www.postgresql.org/docs/current/textsearch.html)
- PostgreSQL EXPLAIN: [https://www.postgresql.org/docs/current/using-explain.html](https://www.postgresql.org/docs/current/using-explain.html)
- Use The Index, Luke: [https://use-the-index-luke.com/](https://use-the-index-luke.com/)
- Markus Winand, *SQL Performance Explained*, 2012
