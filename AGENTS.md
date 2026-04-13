# SequentialPrisonersDilemma Agent Instructions

These instructions apply when working in this repository.

## Environment

- Use the project-local Python environment by default:
  - `./.conda/bin/python`
- Prefer running scripts and checks from repo root:
  - `/home/doesburg/Projects/SequentialPrisonersDilemma`

## Cross-Repo Mapping

- This repository (`SequentialPrisonersDilemma`) contains the canonical Python implementation for the code-backed learned-cooperation repeated Prisoner's Dilemma experiment.
- The public website `https://humanbehaviorpatterns.org/` is built from the sibling `human-cooperation-site` repo and serves as the documentation/presentation layer for this experiment.
- Required mapping:
  - `SequentialPrisonersDilemma/` in this repo <-> `docs/learned-cooperation/repeated-prisoners-dilemma/ppo-study.md` in `human-cooperation-site`
- The higher-level pages `docs/learned-cooperation/learned-cooperation.md`, `docs/learned-cooperation/prisoners-dilemma/prisoners-dilemma.md`, and `docs/learned-cooperation/repeated-prisoners-dilemma/repeated-prisoners-dilemma.md` provide conceptual framing for the same experiment family and should stay consistent with concrete claims made from this repo.
- When modifying the experiment behavior, learning setup, or reported results here, check whether the corresponding website pages must be updated there.
- When modifying the website pages there, preserve fidelity to the Python implementation and reported outputs here.
