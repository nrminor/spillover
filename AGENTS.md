# Agent Instructions for spillover

## What This Project Is

spillover is a Rust workspace for sorting datasets that don't fit in memory. The core `spillover` crate provides a generic external sort engine — it knows nothing about what it's sorting, how to compare items, how to deduplicate them, or how to serialize them to disk. All of those decisions are made by the caller through trait implementations.

`spillover-bio` is the first consumer. It injects genomics-specific opinions: bloom filter dedup, FASTQ/FASTA-compatible record traits, sequence-aware sort keys, and the `dryice` format for temporary on-disk storage. But `spillover` itself must never know about any of this.

## Design Philosophy

The core crate exists to be boring. It should be the kind of library where someone reads the trait definitions, immediately understands how to plug their own types in, and never has to fight the abstractions. If a user needs to read the internals to understand how to use it, the API has failed.

Generality is the whole point. Every design decision in `spillover` should be evaluated against the question: "does this prevent someone from using this crate for a use case I haven't thought of?" If the answer is yes, the design is wrong. This means no hardcoded serialization formats, no assumptions about record size (fixed or variable), no baked-in comparison logic, and no opinions about what deduplication means.

The two crates have a strict dependency direction: `spillover-bio` depends on `spillover`, never the reverse. If you find yourself wanting to add something to the core crate that only makes sense for genomics, it belongs in `spillover-bio`. If you find something in `spillover-bio` that would be useful to any external sort user, consider promoting it — but only if it can be expressed without domain-specific types.

## Attitudes

Prefer fewer, more powerful abstractions over many narrow ones. A trait with one method that covers ten use cases is better than five traits that each cover two. But don't sacrifice clarity for generality — if a single trait is doing too much, it's a sign that the abstraction is wrong, not that you need more surface area.

Treat compiler errors and clippy warnings as information, not obstacles. They are telling you something about the design. A type error during refactoring often reveals that two things you thought were the same are actually different. Listen to that signal.

Be skeptical of convenience methods that obscure what's happening. A user who calls `.sort()` should be able to predict roughly what the library will do (buffer items, flush to disk when full, merge at the end). If the convenience hides behavior that would surprise a careful user, it needs a different name or shouldn't exist.

Memory budgets are the user's business. The library should respect them precisely, not approximately. "Approximately constant memory" is the promise — breaking it, even in edge cases, is a bug.

## Before You Start Working

1. Run `just setup` if `.agents/repos/` is empty. The reference implementations in `sra-taxa-rs` (particularly `sra-taxa-build/src/merge.rs` and `sra-taxa-build/src/engine.rs`) and `haveitnway` are essential context.
2. Run `just --list` and use the justfile recipes for all standard operations.
3. Run `just check` before every commit. jj does not run hooks.

## Prohibited Actions

1. **`cargo install`** — the user's responsibility
2. **`jj git push`** — the user's responsibility
3. **Skipping `just check`** — non-negotiable
4. **Creating documentation files** unless explicitly asked
5. **Adding domain-specific types to the `spillover` core crate**
