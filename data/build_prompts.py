"""Assemble the prompt corpus to feed into synthesis.

Sources (in order of quality):
  1. The 10 caveman repo eval prompts (canonical)
  2. Hand-curated short tech-Q&A templates expanded across topics
  3. (Optional) HuggingFace CodeAlpaca / dolly subsets filtered to short tech instructions

Output: data/prompts/tech_qa_seed.jsonl, one {"prompt": str, "topic": str} per line.
Target size: 3000-5000 prompts, all <500 tokens.

Run: python data/build_prompts.py [--n 3000] [--hf]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS_PROMPTS = ROOT / "data" / "seeds" / "eval_prompts_en.txt"
OUT = ROOT / "data" / "prompts" / "tech_qa_seed.jsonl"

TEMPLATES = [
    "Why does my {thing} {bad_behavior}?",
    "Explain {concept}.",
    "What's the difference between {a} and {b}?",
    "How do I fix a {problem} in {context}?",
    "What does the {tool} command tell me?",
    "How does {data_structure} handle {edge_case}?",
    "Why am I getting {error} in {environment}?",
    "What's the point of using {pattern} in {context}?",
    "How does {a} differ from {b}?",
    "When should I use {a} vs {b}?",
    "How do I debug a {problem}?",
    "What's the best way to {action} in {language}?",
    "Why is my {thing} slow under load?",
    "Walk me through {protocol}.",
    "What is {concept} and when do I need it?",
    "How does {language} handle {feature}?",
    "What are the trade-offs of {a} vs {b}?",
    "Show me how to {action}.",
    "What causes {error} and how do I prevent it?",
    "How do I write a {pattern} in {language}?",
]

TOPICS = {
    "thing": ["React component", "Node service", "Python script", "Postgres query",
              "Docker container", "CI build", "Kafka consumer", "API endpoint",
              "Redis client", "SQLAlchemy session", "GraphQL resolver", "Lambda function",
              "WebSocket connection", "gRPC client", "Spark job"],
    "bad_behavior": ["leak memory", "stall under load", "double-fire events",
                     "deadlock", "throw 502", "miss messages", "OOM",
                     "drop connections", "ignore env vars", "rate-limit itself"],
    "concept": ["database connection pooling", "CAP theorem", "eventual consistency",
                "vector clocks", "CRDTs", "Raft consensus", "TLS handshakes",
                "JWT signing", "OAuth PKCE", "B-tree indexes", "LSM trees",
                "MVCC", "two-phase commit", "circuit breakers", "leaky bucket",
                "exponential backoff", "service mesh", "sidecar pattern",
                "blue-green deploys", "feature flags", "shadow traffic",
                "consistent hashing", "bloom filters", "merkle trees",
                "WAL", "fsync", "page cache", "io_uring", "epoll",
                "copy-on-write", "garbage collection"],
    "a": ["TCP", "merge", "queue", "REST", "SQL", "monorepo", "REST", "OAuth", "PUT",
          "WebSocket", "Postgres", "Docker", "TypeScript", "yarn", "rebase",
          "promise", "process", "Kafka topic", "JSON Schema", "git stash"],
    "b": ["UDP", "rebase", "topic", "gRPC", "NoSQL", "polyrepo", "GraphQL", "SAML",
          "PATCH", "SSE", "MySQL", "Podman", "Flow", "pnpm", "merge",
          "callback", "thread", "RabbitMQ queue", "Protobuf", "git worktree"],
    "problem": ["memory leak", "race condition", "N+1 query", "deadlock",
                "infinite loop", "stale cache", "thundering herd", "flaky test",
                "OOM kill", "DNS resolution failure", "TLS expiry"],
    "context": ["a long-running Node.js process", "Python asyncio", "Go goroutines",
                "a Lambda function", "a Postgres-backed API", "Ruby on Rails",
                "Django views", "a Kubernetes deployment", "a React app",
                "a SwiftUI view", "an Android app", "a Spark pipeline"],
    "tool": ["SQL EXPLAIN", "git bisect", "strace", "perf", "tcpdump", "lsof",
             "kubectl describe", "docker stats", "py-spy", "node --inspect",
             "EXPLAIN ANALYZE", "vmstat", "iostat", "dtrace", "eBPF"],
    "data_structure": ["a hash table", "a B-tree", "a skip list", "a trie",
                       "a heap", "a bloom filter", "an LRU cache",
                       "a ring buffer", "a CRDT counter"],
    "edge_case": ["collisions", "deletes", "concurrent writes",
                  "resize", "overflow", "ordering", "partial failure"],
    "error": ["CORS errors", "ECONNREFUSED", "EADDRINUSE", "stale closure",
              "SIGPIPE", "EPIPE", "OOMKilled", "ETIMEDOUT", "DNS_PROBE_FINISHED_NXDOMAIN",
              "401 from Cognito", "TLS handshake failures", "broken pipe"],
    "environment": ["my browser console", "a Docker container", "production only",
                    "CI", "local dev", "a Lambda cold start", "after a deploy"],
    "pattern": ["a debouncer", "a circuit breaker", "rate limiting",
                "retries with backoff", "the saga pattern", "the outbox pattern",
                "CQRS", "event sourcing", "the strangler fig pattern",
                "feature flags", "shadow testing"],
    "action": ["paginate efficiently", "stream a large file", "implement a worker pool",
               "build a rate limiter", "write idempotent endpoints",
               "deduplicate a stream", "compress a payload", "verify a webhook",
               "rotate a secret", "implement healthchecks", "graceful shutdown"],
    "language": ["Go", "Python", "TypeScript", "Rust", "Swift", "Kotlin",
                 "Java", "Ruby", "Elixir", "C#"],
    "protocol": ["TCP", "TLS 1.3", "HTTP/2", "HTTP/3", "QUIC", "WebSocket",
                 "gRPC", "MQTT", "OAuth2 authorization code flow",
                 "OIDC", "DNS resolution", "BGP"],
    "feature": ["concurrency", "errors", "generics", "memory", "I/O",
                "modules", "macros", "iterators", "channels"],
}


def fill(template: str, rng: random.Random) -> tuple[str, str] | None:
    out = template
    topic_tag = "mixed"
    for key in list(TOPICS.keys()):
        token = "{" + key + "}"
        if token in out:
            choice = rng.choice(TOPICS[key])
            out = out.replace(token, choice, 1)
            if topic_tag == "mixed":
                topic_tag = key
    return (out, topic_tag) if "{" not in out else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="target prompt count")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    prompts: list[dict] = []
    seen: set[str] = set()

    # 1. canonical caveman eval prompts
    if SEEDS_PROMPTS.exists():
        for line in SEEDS_PROMPTS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and line not in seen:
                prompts.append({"prompt": line, "topic": "canonical"})
                seen.add(line)

    # 2. template expansion
    attempts = 0
    while len(prompts) < args.n and attempts < args.n * 20:
        attempts += 1
        t = rng.choice(TEMPLATES)
        filled = fill(t, rng)
        if filled is None:
            continue
        p, topic = filled
        if p in seen:
            continue
        seen.add(p)
        prompts.append({"prompt": p, "topic": topic})

    with OUT.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} prompts -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
