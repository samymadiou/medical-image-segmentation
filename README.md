# ACDC Cardiac MRI Segmentation

A PyTorch-based deep learning project for multiclass cardiac segmentation on the ACDC dataset. The repository implements U-Net and residual U-Net architectures to segment the left ventricle, myocardium, and right ventricle from short-axis cardiac MR images.

## Overview

This project was developed for medical image segmentation research and is structured to be easy to understand, reproduce, and present as a personal GitHub portfolio project. It combines:

- a patient-level preprocessing pipeline,
- custom medical image dataset loading,
- multiple segmentation model variants,
- custom loss functions for class imbalance,
- validation/inference utilities,
- notebook-based experimentation and evaluation.

The work focuses on the ACDC challenge setup, where target masks encode 4 classes: background, right ventricle, myocardium, and left ventricle.

## Project highlights

- U-Net baseline implementation with skip connections
- Residual U-Net variant designed for stronger feature propagation
- Multiclass Dice and focal-style segmentation losses
- Dataset handling for train/validation/test splits
- Notebook-based exploratory workflow for training and evaluation
- Reproducible Python environment metadata

## Repository structure

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── Makefile
├── CONTRIBUTING.md
├── .gitignore
├── scripts/
│   └── check_setup.py
├── medicalDataLoader.py          # Data pipeline and PyTorch dataset
├── UNet_Base.py                  # Standard U-Net architecture
├── Model_Samy.py                 # Residual U-Net implementation
├── residualunet.py               # Residual encoder/decoder blocks
├── f_loss.py                     # Loss functions for segmentation
├── utils.py                      # Evaluation and prediction helpers
├── progressBar.py                # Lightweight console progress bar
├── mainSegmentation.ipynb        # Training workflow notebook
├── Eval.ipynb                    # Evaluation notebook
└── Data/                         # Not included in repo; expected at runtime
```

## Dataset format

Place the dataset in a local `Data/` folder with the structure below:

```text
Data/
├── train/
│   ├── Img/
│   └── GT/
├── val/
│   ├── Img/
│   └── GT/
└── test/
    ├── Img/
    └── GT/
```

The loader expects image and mask files to be paired by filename order. Ground-truth masks follow the ACDC convention where class labels are encoded as grayscale intensities such as 0, 85, 170, and 255, and the project converts them to class indices internally.

## Environment setup

### Option 1: Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option 2: project metadata

```bash
pip install -e .
```

### Validate the environment

```bash
python scripts/check_setup.py
```

## Training workflow

1. Open `mainSegmentation.ipynb` from the project root.
2. Ensure the working directory is the repository folder so relative imports resolve correctly.
3. Confirm the dataset is available under `./Data`.
4. Choose the model architecture (U-Net or Residual U-Net).
5. Train with the notebook configuration or adapt the hyperparameters for your experiments.

Typical training uses:

- Adam optimizer
- learning rate around `1e-4` to `1e-3`
- multiclass segmentation objective combining cross-entropy and Dice-style terms
- validation checkpoints and saved outputs under `models/`, `Results/`, and `Stats/`

## Evaluation workflow

Use `Eval.ipynb` to:

1. load a saved checkpoint,
2. run model inference on the test set,
3. save predicted masks,
4. compute Dice scores for the foreground classes,
5. compare predictions against ground truth masks.

The scripts in `utils.py` provide the core support for:

- label conversion,
- segmentation inference,
- Dice-based metric evaluation,
- saving predictions,
- generic image utilities.

## Model architecture

This project includes two main model families:

- `UNet_Base.py`: classic encoder-decoder segmentation model
- `Model_Samy.py`: residual U-Net variant with skip connections and deeper feature aggregation

The residual version is designed to improve feature reuse across the encoder-decoder path while maintaining a compact segmentation network for medical imaging tasks.

## Loss functions

The codebase contains multiple segmentation losses in `f_loss.py`, including:

- Dice loss
- Focal loss
- Tversky loss
- Focal Tversky loss
- generalized / multiclass Dice formulations
- binary cross-entropy variants

These losses are useful for handling foreground/background imbalance, which is common in medical segmentation tasks.

## Notes for reproducibility

- Run notebooks with the repository root as the working directory.
- Do not commit the dataset, checkpoint files, or generated outputs.
- The repository intentionally omits `Data/`, `Results/`, `Stats/`, and `models/` from version control.
- The project is designed as a research prototype and can be extended with better logging, experiment tracking, and model packaging.

## Roadmap

This project can be improved further with:

- a clean training CLI using `argparse`
- experiment tracking via MLflow or Weights & Biases
- automated metrics logging and plots
- Docker support for consistent deployment
- unit tests around dataset loading and evaluation utilities
- a modular model registry and configuration manager

## Contributing

Contributions are welcome. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for development guidance and pull request expectations.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This project is based on the ACDC cardiac segmentation challenge and uses a U-Net-inspired deep learning approach for medical image segmentation.