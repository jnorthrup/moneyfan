# Track Spec: Superproject Neighbor Interface Mitigation

## Overview
Define the canonical interface seams between this repo and its neighboring project surfaces so conductor can treat this repo as the superproject pinnacle without stale path assumptions or fuzzy ownership.

## Goals
1. Name the neighboring surfaces already evidenced from this repo.
2. Define one owner and one contract per interface seam.
3. Record current breakage, ambiguity, and mitigation priorities.
4. Route delegated execution through `kilo` when product slices are ready.

## Runtime Route
Delegated execution runtime: `kilo`.

## In Scope
- Local `/conductor/` truth for interface ownership, contract, and mitigation.
- Seams involving the root-level Torch HRM surface.
- Seams involving the Kotlin ANE bridge.
- Seams involving checkpoint, artifact, and future Core ML export handoff.

## Out of Scope
- Rewriting neighboring repos from this track.
- Private ANE runtime redesign.
- Cross-repo product edits without an explicit later slice.

## Acceptance Criteria
1. A local interface matrix exists with neighbor, owner, contract, evidence, and mitigation.
2. Superproject assumptions are corrected to match the current root-level repo layout.
3. The next cross-surface implementation slice can be named without rediscovery.
