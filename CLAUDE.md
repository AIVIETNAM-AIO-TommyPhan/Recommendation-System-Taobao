# Working with me on this project

## About me
I'm a software developer learning EDA. I'm comfortable with code; I'm newer to the
statistics and the data-analysis reasoning behind it. Teach me the *why* as you go.

## How to help me
- For any EDA task (exploring, cleaning, or preparing this data), use the
  `/eda-analysis` skill — it's the workflow this repo is built around.
- When you compute a statistic or make a data decision, explain in one or two plain
  sentences what it means and why it matters — don't just report the number.
- Define a statistical term the first time it appears (a short `📘` note is great,
  the way `EDA_REPORT.md` already does).
- Prefer the simplest approach that works: readable pandas over clever one-liners.
- State assumptions before acting. If a column's meaning or the goal is unclear, ask
  instead of guessing.
- Follow the EDA loop this repo already uses: observation → hypothesis → proposed
  change → how to validate it.

## Orientation
- `dataset/train.csv`, `dataset/test.csv` — CTR data, target `click`.
- `README.md`, `Data Foundation.md` — schema and the `0`-sentinel convention.
- `EDA_REPORT.md` (+ `EDA_REPORT.vi.md`) — reviewed findings, one section per topic.
- `eda_scripts/` — reusable analysis and figure scripts.
