# 2. Use NetworkX Initially

Date: 2026-07-20

## Status
Accepted

## Context
We need to model threat intelligence relationships (MITRE ATT&CK, STIX, CVE) and compute semantic threat scores. A graph database or library is required.

## Decision
We decided to use **NetworkX** initially rather than a heavy graph database like Neo4j.

## Consequences
- **Pros:** Pure Python, very easy to integrate, zero operational overhead, fast for in-memory operations and algorithm prototyping (like PageRank or custom similarity algorithms).
- **Cons:** Not scalable for massive graphs exceeding memory limits. We can migrate to Neo4j later if the Knowledge Graph size demands it.
