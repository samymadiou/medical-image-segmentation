# Contributing

Contributions are welcome for improving the model, refining the training pipeline, or improving the documentation.

## Development setup

1. Create and activate a virtual environment.
2. Install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Validate the environment:

```bash
python scripts/check_setup.py
```

## Workflow

- Keep the project reproducible and easy to understand.
- Prefer small, well-documented changes.
- Document model assumptions and dataset-specific preprocessing clearly.
- Keep notebooks focused on experimentation, while reusable logic should live in Python modules.

## Pull request expectations

- Describe the problem and the intended improvement.
- Include training/evaluation metrics when a model change is made.
- Note any required dataset path or configuration updates.
