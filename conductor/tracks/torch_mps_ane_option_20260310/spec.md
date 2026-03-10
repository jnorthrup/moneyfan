# Track Spec: Torch Option with MPS Fallback and ANE Gap Mapping

## Overview
Build a Torch-first HRM training and evaluation path that can downshift to MPS, export to Core ML for inference, and explicitly map ANE gaps and mitigation tactics.

## Goals
1. Define and implement a Torch training path for the HRM.
2. Add MPS fallback for Apple Silicon training.
3. Add Core ML export and verification for inference.
4. Maintain an ANE gap register with explicit mitigations.

## Runtime Route
Delegated execution runtime: `kilo`.

## In Scope
- Creation of a Torch HRM training surface where none currently exists in this repo.
- Device routing logic `cpu|mps|cuda` where available.
- Core ML export pipeline for inference-only checks.
- ANE gap mapping and mitigation tracking in track artifacts.

## Out of Scope
- Rewriting the private ANE runtime.
- Full trading runtime integration.
- Production deployment automation.

## Acceptance Criteria
1. Torch training runs with deterministic smoke profiles.
2. MPS fallback selects correctly on Apple Silicon.
3. Core ML export produces a loadable model and simple inference parity check.
4. ANE gap register exists with explicit mitigations and priorities.

## Current Repo Reality
- A root-level Torch HRM implementation now exists at `hrm/torch_hrm.py`.
- The concrete ANE surface currently present is the Kotlin native bridge and sample coder under `kotlin/`.
- The next implementation step is to harden the Torch surface with clean focused tests before expanding export or neighbor integrations.
