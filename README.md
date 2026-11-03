# CitraVerse: Cross-Dataset Citrus Disease Monitoring

**An Evidence-Preserving, Reliability-Aware, Explainable, and Edge-Optimized AI Framework for Citrus Disease Monitoring**

---

## 1. Overview

CitraVerse is an AI-based research framework designed for citrus disease classification across multiple datasets and domains.

The framework focuses on improving the reliability, explainability, stability, and deployment efficiency of citrus disease monitoring systems.

It integrates dataset harmonization, multi-evidence feature extraction, cross-dataset evidence stability analysis, reliability estimation, knowledge distillation, model quantization, and visual explainability into a unified pipeline.

Unlike conventional image classification approaches that primarily depend on deep learning predictions, CitraVerse investigates multiple forms of disease-related evidence, including color, texture, shape, and deep semantic features.

The framework also explores how consistently these features represent citrus diseases across different datasets and how lightweight models can preserve useful evidence while reducing computational requirements.

## 2. Research Objectives

The primary objectives of CitraVerse are:

- To develop a cross-dataset citrus disease classification framework.
- To harmonize disease classes across datasets with different class naming conventions.
- To extract and combine multiple forms of visual disease evidence.
- To evaluate the consistency of disease-related evidence across different dataset domains.
- To estimate prediction reliability using confidence and prototype agreement.
- To develop lightweight student models through knowledge distillation.
- To investigate model quantization for edge deployment.
- To improve model interpretability through Grad-CAM visualizations.
- To evaluate classification performance, robustness, reliability, and computational efficiency.

## 3. Key Components

### 3.1 Dataset Harmonization

Standardizes disease class labels across multiple citrus datasets.

Main operations include:
- Dataset scanning and inspection.
- Class mapping and harmonization.
- Duplicate image detection.
- Leakage-aware dataset splitting.
- Dataset provenance tracking.

### 3.2 Multi-Evidence Feature Extraction

Extracts different forms of visual evidence from citrus leaf images.

The framework considers:

- **Color Evidence:** Captures color characteristics associated with disease symptoms.
- **Texture Evidence:** Identifies surface patterns and texture variations.
- **Shape Evidence:** Captures lesion and shape-related characteristics.
- **Deep Evidence:** Extracts high-level semantic representations using deep learning models.

These evidence representations are combined through an evidence fusion mechanism.

### 3.3 Cross-Dataset Evidence Stability

Investigates whether disease-related evidence remains consistent across different dataset domains.

The framework evaluates evidence stability under different data conditions and perturbations.

### 3.4 Reliability Estimation

Evaluates prediction reliability using confidence information and prototype agreement.

The reliability component aims to distinguish predictions with stronger supporting evidence from predictions with weaker evidence.

### 3.5 Knowledge Distillation

Uses a teacher-student learning approach to develop a lightweight student model.

The objective is to transfer useful knowledge from the teacher model while preserving important disease-related evidence.

### 3.6 Edge Optimization

Investigates model optimization techniques for resource-constrained deployment.

Includes:
- Model quantization.
- Model size evaluation.
- Inference performance measurement.
- Computational efficiency analysis.

### 3.7 Explainable AI

Uses Grad-CAM visualizations to inspect image regions that influence model predictions.

These visualizations support model interpretation and help investigate whether predictions focus on relevant image regions.

### 3.8 Model Evaluation

Evaluates the framework using classification metrics, robustness measurements, reliability analysis, and model efficiency measurements.

---

## 4. Proposed Research Metrics

CitraVerse investigates two project-proposed research metrics.

### 4.1 Disease Evidence Stability Score (DESS)

DESS is proposed to study the consistency of disease-related evidence across different dataset domains.

It aims to evaluate whether the extracted evidence representations remain stable when the input data comes from different sources.

### 4.2 Perturbation Consistency Score (PCS)

PCS is proposed to investigate prediction and evidence consistency under input perturbations.

It helps analyze how model behavior changes when images undergo controlled modifications.

**Note:** DESS and PCS are project-proposed research metrics and should not be interpreted as established benchmark standards.

---

## 5. Datasets

The complete datasets are not included in this GitHub repository because of their size and dataset licensing considerations.

Researchers must download the datasets independently from their respective sources.

### Dataset Sources

1. [Citrus Dataset - Mendeley Data](https://data.mendeley.com/datasets/8d9fv6kpt3/2)
2. [Citrus Dataset - Mendeley Data](https://data.mendeley.com/datasets/3f83gxmv57/2)
3. [Citrus Dataset - Mendeley Data](https://data.mendeley.com/datasets/f7cr74mwpj/1)

### Dataset Organization

After downloading the datasets, organize them according to the directory names and image structure expected by the configuration files.

Example:

```text
Datasets/
├── Pakistan_Citrus/
├── Bangladesh_Sweet_Orange/
└── Bangladesh_lemon/
```

The actual directory names and class structure must match the configurations.

Before running the pipeline, review:

- `configs/datasets.yaml`
- `configs/class_mapping.yaml`

**Important:** Dataset availability, access permissions, and licensing conditions are determined by the original dataset providers. Verify these conditions before downloading or using the datasets.

---

## 6. Project Structure

The GitHub repository contains the source code, configuration files, tests, and documentation required to understand and work with the CitraVerse framework.

```text
CitraVerse/
│
├── configs/
│   ├── class_mapping.yaml
│   ├── datasets.yaml
│   ├── edge.yaml
│   ├── reliability.yaml
│   └── training.yaml
│
├── scripts/
│   ├── build_prototypes.py
│   ├── evaluate.py
│   ├── export_dataset.py
│   ├── extract_evidence.py
│   ├── generate_gradcam.py
│   ├── generate_visualizations.py
│   ├── harmonize_datasets.py
│   ├── profile_datasets.py
│   ├── quantize_model.py
│   ├── run_robustness.py
│   ├── train_student.py
│   └── train_teacher.py
│
├── src/
│   ├── data/
│   ├── distillation/
│   ├── evaluation/
│   ├── evidence/
│   ├── explainability/
│   ├── harmonization/
│   ├── models/
│   ├── preprocessing/
│   ├── prototypes/
│   ├── provenance/
│   ├── quantization/
│   ├── reliability/
│   ├── stability/
│   └── utils/
│
├── tests/
│
├── AgriVerse_representative_image_selection.py
├── generate_visualizations.py
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 7. Technologies Used

The project uses the following technologies and tools:

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Machine Learning:** Scikit-learn
- **Image Processing:** OpenCV and image-processing libraries
- **Data Processing:** NumPy and Pandas
- **Visualization:** Matplotlib
- **Explainable AI:** Grad-CAM
- **Testing:** Pytest
- **Configuration:** YAML
- **Version Control:** Git and GitHub

The exact dependency versions are specified in `requirements.txt`.

---

## 8. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/CitraVerse.git
```

Navigate into the project directory:

```bash
cd CitraVerse
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

For Linux or macOS:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

Install the appropriate PyTorch and torchvision versions for your system using the official installation guide:

https://pytorch.org/get-started/locally/

Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Datasets

Download the required datasets from their original sources.

Organize them according to the expected directory structure.

Review the configuration files:

```text
configs/datasets.yaml
configs/class_mapping.yaml
```

Update the dataset paths and class mappings if necessary.

### Step 5: Verify the Installation

Run the project tests:

```bash
python -m pytest tests/ -v
```

---

## 9. Running the Pipeline

CitraVerse provides a pipeline entry point through `run_pipeline.py`.

The research workflow includes the following stages:

### Stage 1: Dataset Profiling

```bash
python run_pipeline.py --profile
```

Analyzes dataset structure, class distributions, and dataset characteristics.

### Stage 2: Dataset Preprocessing

```bash
python run_pipeline.py --preprocess
```

Performs dataset preprocessing, harmonization, duplicate handling, and splitting.

### Stage 3: Evidence Extraction

```bash
python run_pipeline.py --extract-evidence
```

Extracts visual evidence from citrus leaf images.

### Stage 4: Prototype Building

```bash
python run_pipeline.py --build-prototypes
```

Builds disease-related prototypes from the extracted evidence.

### Stage 5: Teacher Model Training

```bash
python run_pipeline.py --train-teacher
```

Trains the teacher model.

### Stage 6: Student Model Training

```bash
python run_pipeline.py --train-student
```

Trains the lightweight student model using knowledge distillation.

### Stage 7: Model Quantization

```bash
python run_pipeline.py --quantize
```

Applies the configured model quantization procedure.

### Stage 8: Robustness Evaluation

```bash
python run_pipeline.py --robustness
```

Evaluates model behavior under the implemented perturbation conditions.

### Stage 9: Model Evaluation

```bash
python run_pipeline.py --evaluate
```

Evaluates classification performance and other configured metrics.

### Stage 10: Explainability

```bash
python run_pipeline.py --explain
```

Generates Grad-CAM-based visual explanations.

**Note:** The commands above describe the intended workflow. Available arguments, execution dependencies, and required intermediate files should be checked using:

```bash
python run_pipeline.py --help
```

Some stages may require outputs from earlier stages.

---

## 10. Testing

The repository includes unit tests and integration-oriented tests for important framework components.

Run all tests using:

```bash
python -m pytest tests/ -v
```

The tests cover components such as:

- Dataset scanning.
- Class mapping.
- Duplicate detection and splitting.
- Image dataset handling.
- Evidence extraction and fusion.
- Knowledge distillation losses.
- Prototype and stability components.
- Quantization.
- Reliability estimation.
- Visualization generation.

Testing helps verify the implemented components independently of the complete research pipeline.

---

## 11. Experimental Results

This GitHub repository is primarily intended to provide the CitraVerse source code and configuration files.

Generated experimental outputs, training logs, model checkpoints, and visualization results are not included in this source-code-only repository.

Actual experimental performance should be evaluated using the original datasets and the configured research pipeline.

Relevant evaluation measures may include:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Robustness measurements.
- Reliability measurements.
- Model size.
- Inference performance.

Results should always be interpreted alongside the dataset composition, experimental configuration, and evaluation protocol.

Synthetic test results alone do not establish real-world citrus disease classification performance.

---

## 12. Model Checkpoints

Pretrained teacher and student model checkpoints are not provided in this repository.

Model checkpoints must be generated by executing the corresponding training stages.

The availability of trained models depends on successful execution of the training pipeline and the required datasets.

---

## 13. Limitations

The CitraVerse framework has several research considerations:

1. Cross-dataset generalization must be established through experiments using the actual datasets.
2. Dataset differences, class imbalance, image quality, and domain variations may affect classification performance.
3. Reliability estimates depend on the quality of the underlying models and evidence representations.
4. Grad-CAM visualizations indicate model activation patterns and should not be interpreted as proof of causal disease reasoning.
5. Quantization performance depends on the model architecture and the implementation.
6. Dynamic INT8 quantization may leave convolutional layers in floating-point precision. Quantization claims should reflect actual layer-level results.
7. DESS and PCS require appropriate experimental evaluation before broader adoption.

---

## 14. Future Scope

Potential future extensions of CitraVerse include:

- Evaluation on additional citrus disease datasets.
- Improved cross-domain generalization.
- More advanced evidence fusion mechanisms.
- Further development and validation of DESS and PCS.
- Improved prediction reliability estimation.
- Hardware-specific edge optimization.
- Deployment on resource-constrained devices.
- Real-world citrus disease monitoring applications.

---

## 15. Research and Dataset Attribution

CitraVerse builds upon existing research in deep learning, computer vision, explainable AI, knowledge distillation, model compression, and agricultural disease monitoring.

Researchers using this framework should appropriately cite:

- Original dataset publications.
- Relevant research papers.
- External libraries and frameworks.
- Any additional methods incorporated into their work.

Please verify the licensing and attribution requirements of each dataset before using or redistributing it.

---

## 16. Contributing

Contributions to CitraVerse are welcome.

Potential areas for contribution include:

- Dataset harmonization.
- Evidence extraction methods.
- Model architecture improvements.
- Reliability estimation.
- Robustness evaluation.
- Explainability techniques.
- Edge optimization.
- Testing and documentation.

For significant changes, please describe the proposed improvement and its experimental justification.

---

## 17. License

A repository license has not yet been selected.

Until a license is added, the repository should not be assumed to grant permission for unrestricted reuse, modification, or redistribution.

---

## 18. Acknowledgement

CitraVerse is a research-oriented framework exploring cross-dataset, explainable, reliability-aware, and edge-optimized artificial intelligence for citrus disease monitoring.

The project aims to investigate how multiple forms of visual evidence can support more reliable and interpretable agricultural AI systems.

---

**Project:** CitraVerse  
**Domain:** Artificial Intelligence | Machine Learning | Computer Vision | Precision Agriculture  
**Focus:** Citrus Disease Monitoring | Explainable AI | Cross-Dataset Learning | Edge AI