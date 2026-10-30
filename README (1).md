# Dataset location

The UCI HAPT dataset is not redistributed in this repository.

Run the command below from the repository root. It downloads the official archive, verifies its SHA-256 checksum, and extracts only the files required for the Stage 3 development experiments.

```bash
python run_stage3.py --download --data-dir data/hapt --results-dir results_stage3 --figures-dir figures_stage3 --cache-dir .cache_stage3
```

Do not commit the extracted dataset. The dataset is licensed under CC BY 4.0 and must be cited as described in the main README.

