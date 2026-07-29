<h1 align="center">
<b>Decoding Driver Behavior Through Multimodal Analytics in Virtual Reality</b>
</h1>

 <p align="center">
  <a href="#-abstract"> 📋 Abstract</a> •
  <a href="#-dependencies"> 📄 Dependencies</a> •
  <a href="#-code-directory-structure"> 📁 Code Directory Structure</a>
  <p align="center">
  <a href="#-license"> 🔑 License</a> •
  <a href="#-citation"> 📜 Citation</a> •
  <a href="#-acknowledgements"> 🙏🏼 Acknowledgements</a>

This repository contains the analytics pipeline code for the paper "Decoding Driver Behavior Through Multimodal Analytics in Virtual Reality".


## 📋 Abstract
> In-vehicle infotainment (IVI) systems impose significant cognitive and manual demands on drivers, yet their impact on driving performance and safety remains inadequately characterized due to the difficulty and cost of real-world testing. This paper presents a comprehensive analysis of the V-Cockpit analytics pipeline, a data processing and statistical modeling framework designed to quantify the effects of IVI interactions on driver behavior using multimodal data collected in a controlled virtual reality driving simulator. We analyzed sessions from 28 participants performing secondary IVI tasks (phone dialing in urban environments; playlist selection on highways) at varying speed conditions, with synchronized recordings of gaze, vehicle dynamics, physiological signals, and scenario events. Descriptive analysis revealed cognitive tunneling patterns during IVI engagement, characterized by improved lane control but catastrophic declines in safety-critical awareness metrics. Rate-of-change analysis exposed accumulating lane-departure risks masked by static averages. Predictive modeling compared three approaches: standard linear mixed-effects models (baseline R²m = 0.336), demographic clustering by driving experience (R²m = 0.516 on average, representing a 53.4% improvement, with the best individual model reaching R²m = 0.69), and Random Forest applied to the same demographic clusters (R²m = 0.927 on average). Random Forest achieved the highest raw predictive fit, but its best-performing models relied disproportionately on static demographic variables and session-phase context rather than on the dynamic, session-level descriptors a live dashboard can display. Demographic clustering offered the most directly interpretable account of driver behavior heterogeneity, indicating that it is organized, at least in part, by experience-related pathways: intermediate-experience drivers showed the highest predictive ceiling, while inexperienced and experienced drivers showed lower, comparable ceilings, with all three groups gaining predictive power gradually and similarly as additional predictors were added. Across all three approaches, three dynamic indicators — mean distraction, speed variance, and mean focus — recurred in the large majority of best-performing models, providing an evidence-based core set for adaptive dashboards. These findings support V-Cockpit's dynamic multimodal indicators as informative, interpretable proxies for driver state, contingent upon demographic stratification, and motivate the design of personalized, adaptive in-vehicle systems that surface this core indicator set while adjusting supplementary indicators based on driver experience profiles.

### Pipeline overview

```
User session (raw) -> Integrity check -> Data ingestion -> Extractor -> Analysis -> Results
```

1. **Integrity check** (`src/checker`) verifies that each session folder contains the
   expected raw files before further processing. This step is run separately, ahead of
   and independently from the pipeline in this repository (`src/main.py`), which
   assumes sessions already passed the check.
2. **Ingestion** (`src/data_ingester`) reorganizes raw recordings into a structured,
   per-scenario "user session mapped" representation.
3. **Extraction** (`src/extractor`, `src/features`) transforms raw signals into the
   indicators used in the paper:
   - **Involvement** — physiological engagement derived from PPG-based HRV (Poincaré
     plane analysis).
   - **Focus** — stability of the driver's gaze (fixation duration/consistency).
   - **Distraction** — cumulative, individually-calibrated measure of off-task visual
     attention (Eq. 1 in the paper).
   - **Awareness** and **Safety** — composite indices derived following the EuroNCAP
     Assessment Protocol.
   - Vehicle dynamics and event-related measures: safety distance mask, straight
     driving variance, reaction/execution time.
4. **Analysis** (`src/analysis/process_study.py`, `src/analysis/calculate_statistics.py`,
   `src/analysis/perform_experiments.py`, `src/analysis/perform_experiments_model.py`)
   runs descriptive statistics and predictive modeling (linear mixed-effects models,
   demographic clustering, Random Forest) on the extracted dataset.

All stages are orchestrated by a single pipeline entry point, `src/main.py`
(`run_pipeline()`), which is also the repository's CLI entry point.

## 📄 Dependencies
> ⚠️ The project is based on **Python 3.12** (pinned in [`.python-version`](.python-version); some pinned dependencies, e.g. `matplotlib==3.9.3`, have no prebuilt wheels on newer Python versions and would need to be compiled from source).

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

This creates a `.venv` using the pinned Python 3.12 and installs every dependency from
the lockfile (`uv.lock`). Prefix commands with `uv run` (e.g.
`uv run python src/main.py`) or activate the environment with
`source .venv/bin/activate`.

Or with a plain virtualenv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

The pipeline is configured through environment variables, loaded via `python-dotenv`
in `src/app_config/config.py`.

1. Copy the example file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set at least `BASE_ROOT` and `DATA` to absolute paths on your
   machine. See the comments in [`.env.example`](.env.example) for the meaning of
   each variable (`DEBUG`, `BASE_ROOT`, `DATA`, `LOGS_JSON`, `STUDY_FOLDER`,
   `STUDY_PICKLE_FILE_PATH`, `SCENARIO`, `USERS`).

### Usage

Run the full pipeline (ingestion, processing, statistics, and experiments) from the
repository root:

```bash
uv run python src/main.py \
  --analysis_type LMEM \
  --scenarios City Highway \
  --targets mean_awareness mean_safety
```

Key options (see `uv run python src/main.py --help` for the full list):

- `--analysis_type {LMEM,RF}` — linear mixed-effects models or Random Forest.
- `--scenarios {City,Highway}` — one or more scenarios to include.
- `--clustering` — enable demographic clustering (as described in the paper).
- `--min_pred` / `--max_pred` — min/max number of predictors in model search.

Individual stages can also be run independently, since each module in `src/analysis/`
keeps its own `if __name__ == "__main__"` entry point (e.g.
`uv run python src/analysis/calculate_statistics.py`).

### Tests

```bash
uv run pytest src/tests
```

## 📁 Code Directory Structure

```python
├── data                       <- Raw/processed data (gitignored; see Dependencies > Configuration)
├── logs                       <- Log output (app.log, errors.log)
├── notebooks                  <- Exploratory analysis notebooks
├── src                        <- Source code (importable package) - the pipeline lives here.
│   ├── __init__.py
│   ├── main.py                 <- Single pipeline entry point (run_pipeline): ingestion ->
│   │                              processing -> statistics -> experiments
│   ├── analysis                <- Pipeline stage modules
│   │   ├── process_study.py            <- Builds/updates the Study object from extracted data
│   │   ├── calculate_statistics.py     <- Descriptive statistics on the processed dataset
│   │   ├── perform_experiments.py      <- Linear mixed-effects model (LMEM) experiments
│   │   └── perform_experiments_model.py <- Random Forest experiments
│   ├── app_config              <- Environment-driven configuration (see Dependencies > Configuration)
│   ├── checker                 <- Raw session integrity checks (run standalone, ahead of
│   │                              this pipeline — see Pipeline overview)
│   ├── data_ingester           <- Raw -> structured session mapping
│   ├── extractor               <- Signal -> indicator extraction (involvement, focus,
│   │                              distraction, awareness, safety, gaze track, ...)
│   ├── features                <- Lower-level feature extraction (e.g. HRV features)
│   ├── classes                 <- Study data model
│   ├── utils                   <- Logging config, folder helpers, mapping dictionaries
│   └── tests                   <- Unit tests
│
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml
└── uv.lock
```

## 🔑 License
This project is released under [CC0 1.0 Universal](LICENSE) (public domain dedication) — see the [LICENSE](LICENSE) file for details.

## 📜 Citation
Please use:

```bibtex
It will be added here once the paper is published.
```
--------
