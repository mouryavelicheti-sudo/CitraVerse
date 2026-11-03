# CitraVerse: Cross-Dataset Citrus Disease Monitoring

**An evidence-preserving, reliability-aware, explainable, and edge-optimized AI framework for citrus disease monitoring.**

## Overview

CitraVerse is a research framework designed to study citrus disease classification across multiple datasets. It combines dataset harmonization, multi-evidence feature extraction, cross-dataset evidence stability analysis, reliability scoring, knowledge distillation, model quantization, and visual explainability.

## Key Components

- **Dataset harmonization:** Standardizes disease classes across datasets and supports duplicate detection and leakage-aware splitting.
- **Evidence extraction:** Combines color, texture, shape/lesion, and deep semantic evidence.
- **Cross-dataset stability:** Analyzes disease evidence consistency across dataset domains.
- **Reliability estimation:** Evaluates prediction confidence and prototype agreement.
- **Knowledge distillation:** Trains a lightweight student model using a teacher model and evidence-preservation objectives.
- **Edge optimization:** Evaluates model quantization, model size, and inference performance.
- **Explainability:** Generates Grad-CAM visualizations for model inspection.
- **Evaluation:** Reports classification metrics, robustness measurements, and model efficiency.

## Datasets

The complete datasets are not included in this repository because of their size. Download them from their respective sources and follow the folder layout and configuration described below.

1. [Citrus dataset — Mendeley Data](https://data.mendeley.com/datasets/8d9fv6kpt3/2)
2. [Citrus dataset — Mendeley Data](https://data.mendeley.com/datasets/3f83gxmv57/2)
3. [Citrus dataset — Mendeley Data](https://data.mendeley.com/datasets/f7cr74mwpj/1)

Place the downloaded datasets under `Datasets/` using the dataset directory names and image organization expected by `configs/datasets.yaml`.

Example:

```text
Datasets/
├── Pakistan_Citrus/
├── Bangladesh_Sweet_Orange/
└── Bangladesh_lemon/
```

The actual folder names and class structure must match your configuration. Review `configs/datasets.yaml` and `configs/class_mapping.yaml` before running the pipeline.

## Project Structure

```text
CitraVerse/
├── Architecture/
├── configs/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Installation

1. Install Python compatible with the project's dependencies.
2. Create and activate a virtual environment.
3. Install the appropriate PyTorch and torchvision versions for your system from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).
4. Install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

Run the tests before starting experiments:

```bash
python -m pytest tests/ -v
```

## Running the Pipeline

Run the following stages in the order supported by the project:

```bash
python run_pipeline.py --profile
python run_pipeline.py --preprocess
python run_pipeline.py --extract-evidence
python run_pipeline.py --build-prototypes
python run_pipeline.py --train-teacher
python run_pipeline.py --train-student
python run_pipeline.py --quantize
python run_pipeline.py --robustness
python run_pipeline.py --evaluate
python run_pipeline.py --explain
```

Use the command-line help to inspect the available options:

```bash
python run_pipeline.py --help
```

These stages require the datasets and configuration files to be prepared. See the existing project documentation for the complete pipeline details.

## Experimental Results

Selected research outputs, evaluation metrics, training curves, confusion matrices, robustness analyses, and explainability figures may be included in `outputs/`.

Results should be interpreted alongside the dataset composition, experimental configuration, and evaluation protocol. Do not interpret model explanations as proof of biological causation.

## Model Checkpoints

Trained model checkpoints may be included when licensing, file size, and reproducibility requirements permit. See `models/` for the checkpoints that are actually published.

## Research Notes

The project proposes Disease Evidence Stability Score (DESS) and Perturbation Consistency Score (PCS) as research metrics. These should be described as project-proposed metrics, not established benchmark standards.

Dynamic INT8 quantization may leave convolutional layers in floating-point precision. Quantization claims should reflect the actual layer-level results reported by the implementation.

## Limitations

Performance and cross-dataset generalization must be established through experiments on the actual datasets. Results from synthetic test data do not establish real-world disease-classification performance.

Grad-CAM visualizations indicate model activation patterns and should not be presented as proof of causal disease reasoning.

## Citation and Dataset Attribution

Please cite the original dataset publications and relevant research papers when using the datasets or building upon this framework. Check each dataset's license and the permissions applicable to any included material.

## License

A repository license has not yet been selected. Add a license only after deciding how you want the source code to be reused.
