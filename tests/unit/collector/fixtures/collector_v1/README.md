# Collector V1 attention fixtures

These compressed JSONL files are test-only physical-key baselines. They are
not loaded by Collector at runtime.

- Context and generation were generated from Collector V1 commit
  `a4827ce203e9fbc24fe6c6779a7eaa2a7dc79f1a`.
- Encoder attention uses its original hardcoded grid from
  `36808ecced9af9d0d71d944c716ae96d1d4a2a47` because that operation was added
  after Collector V1.

The first JSONL record is scope metadata; every following record is one
consumer-visible physical lookup key. Unit tests execute the current backend
getter bodies and require every frozen key to remain present.
