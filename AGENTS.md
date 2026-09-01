# AGENTS.md

This repository is designed for multiple coding/research agents and human reviewers.

## Branch isolation

- Work on your own branch.
- Never push directly to `main`.
- Never auto-merge your own PR.
- Never delete another agent's branch.

## Area ownership and isolation

Modify only the task-relevant area unless explicitly authorized.

Examples:
- MediaPipe tasks: `pose/mediapipe/`
- WiLoR tasks: `pose/wilor/`
- Tracking tasks: `tracking/`
- Kinematics tasks: `hand_kinematics/`
- Evaluation tasks: `evaluation/`

Changes to shared interfaces (e.g., `pose/common/`) must be justified in the task report.

## Mandatory reports

No substantial task is complete without a Markdown report in `reports/`.
Follow the required report sections from `reports/README.md`.

## Evaluation-first development

Competing approaches are intentional. Implementations must be measurable under shared evaluation criteria before architectural selection.

## Reproducibility

Record dependency versions, configs, dataset/sample IDs, seeds, hardware, and commands in reports.

## Raw-data immutability

Maintain reproducible stages:

source RGB → raw pose → tracked/cleaned → normalized geometry → derived joint angles → evaluation

Do not destructively overwrite raw extraction outputs.

## Evidence-driven decisions

Prioritize:
1. primary papers
2. official project pages
3. official repositories
4. official dataset pages
5. official documentation

Avoid unsourced architectural claims.

## No unrelated refactoring

Do not rewrite unrelated modules just to match personal style.
Parallel alternative implementations must remain reviewable.

## Repository artifact policy

Do not commit large datasets, videos, checkpoints, model weights, or bulky generated outputs.
