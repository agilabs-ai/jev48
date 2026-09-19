# Public data sources

The first semantic de-risk run intentionally uses public labeled data so it can run with **zero paid teacher/API calls**.

| Dataset | Role | License used by this project |
|---|---|---|
| Banking77 (`PolyAI/banking77`) | seen-domain fine-grained 77-way intent routing | CC BY 4.0 |
| BoolQ (`google/boolq`) | seen-domain binary semantic decision | CC BY-SA 3.0 |
| DBpedia14 (`fancyzhx/dbpedia_14`) | completely held-out OOD topic/entity classification | CC BY-SA 3.0 |
| AG News benchmark copy (`szhuggingface/ag_news`) | completely held-out OOD news classification | Apache 2.0 metadata on that repository |

## Benchmark policy

- official/held-out test rows never enter training
- DBpedia14 and AG News contribute **zero** training examples
- candidate order is deterministically permuted in all frozen benchmark rows
- long examples are filtered before selection rather than truncated after seeing model behavior
- benchmark hash is frozen before NanoJev/Jev queries

At release time, preserve the required attribution/citation information from each upstream dataset card. If there is any uncertainty about redistribution rights, ship the builder script and hashes rather than republishing upstream text wholesale.
