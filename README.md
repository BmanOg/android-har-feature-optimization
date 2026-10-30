# Multi-Objective Feature Optimization for Cross-Subject Android HAR

Reproducible participant-separated baselines and feature-sufficiency analysis for recognizing six daily activities and six postural transitions from Android inertial sensors.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20HAPT-teal.svg)

> **Project status:** This repository reproduces the Stage 3 development analysis for a COSC 750 Pattern Recognition project. All reported results use only the 21 official HAPT development participants. The nine-participant official test partition is excluded from every committed result, figure, tuning decision, and model-selection claim. The bounded NSGA-II wrapper is specified for the next stage and is not presented here as a completed result.

## Study overview

The project asks how aggressively the 561-feature HAPT representation can be reduced without losing the brief sensor patterns needed to recognize postural transitions in unseen participants.

The reproducible Stage 3 analysis includes:

1. A majority-class floor
2. A class-weighted linear Support Vector Machine (SVM)
3. Principal Component Analysis (PCA) followed by a class-weighted linear SVM
4. Mutual-information feature ranking followed by a class-weighted linear SVM
5. A class-weighted radial basis function (RBF) SVM
6. Participant-grouped versus random-window validation
7. A nested feature-count sweep and participant-level uncertainty analysis

Scaling, PCA, mutual-information ranking, and hyperparameter selection are fitted only inside the relevant participant-separated training folds.

## Development results

Five participant-grouped outer folds; values are mean ± sample standard deviation.

| Model | Overall macro-F1 | Daily-activity F1 | Transition F1 |
|---|---:|---:|---:|
| Linear SVM | 0.849 ± 0.046 | 0.936 ± 0.049 | 0.763 ± 0.046 |
| PCA + linear SVM | 0.813 ± 0.045 | 0.914 ± 0.043 | 0.712 ± 0.061 |
| Mutual-information + linear SVM | 0.816 ± 0.036 | 0.921 ± 0.032 | 0.710 ± 0.044 |
| RBF SVM | 0.848 ± 0.039 | 0.932 ± 0.042 | 0.764 ± 0.045 |

A fixed-hyperparameter random-window split produced an overall macro-F1 estimate 0.033 higher than participant-grouped validation. Reducing the linear model to 60 mutual-information-ranked features lowered overall macro-F1 from 0.849 to 0.756. Transition F1 declined by 0.157, compared with 0.029 for daily activities. These are development-only findings, not official-test results.

## Why participant-separated evaluation matters

- Windows from the same person are correlated, so the participant is the independent experimental unit.
- Randomly splitting windows can place a person's motion style on both sides of the evaluation.
- Fold-local preprocessing prevents held-out participants from influencing scaling, feature ranking, PCA, or tuning.
- Participant-level losses reveal whether a small average change hides a large decline for one person.
- The Stage 2 compactness rule requires the one-sided 95% participant-bootstrap upper bound on mean macro-F1 loss to be no more than 0.03. A stricter 0.02 value is reported as a sensitivity check.

## Repository structure

```text
run_stage3.py       Main experiment and figure-generation program
tests/              Unit tests for metrics and feature-family parsing
results/            Committed development-only outputs
figures/            Committed development-only plots
data/README.md      Dataset acquisition instructions
requirements.txt    Pinned pip environment
environment.yml     Reproducible Conda environment
CITATION.cff        Software citation metadata
LICENSE             MIT License for the repository code
```

## Data

The dataset is not redistributed in this repository.

Jorge Luis Reyes-Ortiz, Davide Anguita, Luca Oneto, and Xavier Parra. *Smartphone-Based Recognition of Human Activities and Postural Transitions*. UCI Machine Learning Repository, 2015.  
DOI: [10.24432/C54G7M](https://doi.org/10.24432/C54G7M)  
Dataset license: CC BY 4.0.

The downloaded 75.9 MB archive is verified against this SHA-256 checksum:

```text
4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09
```

The repository's MIT License applies to the code, not to the HAPT dataset.

## Environment

- Python 3.12
- CPU only
- Main random seed: 2026
- Package versions pinned in `requirements.txt` and `environment.yml`

Create an environment with either `pip` or Conda. With `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## One-command reproduction

From the repository root:

```bash
python run_stage3.py --download --data-dir data/hapt --results-dir results_stage3 --figures-dir figures_stage3 --cache-dir .cache_stage3
```

This command downloads and checksum-verifies the official archive, runs five outer participant-grouped folds with three grouped inner folds for hyperparameter selection, and fits development models on the 21 official training participants. It neither loads nor scores the official test files.

For a short pipeline check that should not be cited as a result:

```bash
python run_stage3.py --download --quick --data-dir data/hapt --results-dir smoke_results --figures-dir smoke_figures --cache-dir .cache
```

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

## Committed outputs

- `results/nested_grouped_cv_{folds,summary,tuning}.csv`
- `results/pca_nested_grouped_cv_{folds,participants,summary,tuning}.csv`
- `results/random_vs_grouped_{folds,summary}.csv`
- `results/feature_count_sweep_{folds,summary}.csv`
- `results/feature_count_participant_{scores,uncertainty}.csv`
- `results/top60_feature_stability.csv`
- `results/feature_stability_summary.json`
- `results/environment.json`
- `figures/`: PNG and vector-PDF plots generated from development-only results

## Bounded next-stage search

The planned NSGA-II wrapper evaluates at most 25 populations of 40 masks, with an absolute cap of 1,000 unique masks per outer fold. Candidate subsets will be compared with mutual-information top-`k` subsets and 30 seeded random subsets at matched feature counts.

Three representatives must be frozen from development evidence before any later external evaluation: the smallest non-inferior subset, a normalized knee-point subset, and a transition-protective subset.

## Official-test boundary

Official-test execution is intentionally outside the public Stage 3 reproduction workflow. The default command neither loads nor scores the nine official test participants. Earlier local debugging means that any future HAPT test score must be labeled descriptive rather than an untouched confirmatory result, or replaced by evaluation on a newly sealed external cohort. No official-test output is committed here.

## License

The source code is released under the [MIT License](LICENSE). The UCI HAPT dataset and third-party materials retain their original licenses.

## Citation

If you use this repository, cite the software using [`CITATION.cff`](CITATION.cff). GitHub also exposes the citation from the repository's **Cite this repository** menu.

## Author

**Elijah Ogunniran**  
Bowie State University

Questions and reproducibility reports are welcome through GitHub Issues.

## Transparency

AI tools assisted with drafting, debugging, formatting, and editing. The author designed the participant-separated protocol, ran the experiments, checked the outputs, and remains responsible for the analysis and repository.

