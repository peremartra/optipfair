# OptiPFair Library Maintenance Guide

This guide provides essential information for developers maintaining the OptiPFair library codebase. It covers project structure, development setup, testing, code standards, and key areas for development.

## Table of Contents

1.  **Project Structure**
2.  **Setting up the Development Environment**
3.  **Code Standards and Formatting**
4.  **Testing**
5.  **Understanding Core Modules**
    *   Pruning (`optipfair/pruning/`)
    *   Bias Visualization (`optipfair/bias/`)
    *   Evaluation (`optipfair/evaluation/`)
    *   CLI (`optipfair/cli.py`)
6.  **Adding New Features**
7.  **Updating Dependencies**
8.  **Releasing New Versions**
9.  **Troubleshooting**
10. **Contribution Process**

---

## 1. Project Structure

The OptiPFair library follows a standard Python package structure.

```
optipfair/
├── optipfair/
│   ├── __init__.py         # Package initialization
│   ├── cli.py              # Command-Line Interface entry point
│   ├── pruning/            # Pruning related modules
│   │   ├── __init__.py
│   │   ├── mlp_glu_pruning.py # Logic for MLP GLU pruning
│   │   ├── methods.py         # Neuron selection methods (MAW, VOW, PON)
│   │   └── utils.py           # Pruning utilities
│   ├── bias/               # Bias visualization and analysis modules
│   │   ├── __init__.py
│   │   ├── activations.py     # Activation capturing logic
│   │   ├── metrics.py         # Bias metric calculation
│   │   ├── visualization.py   # Visualization plotting functions
│   │   ├── defaults.py        # Default prompts, settings
│   │   └── utils.py           # Bias utilities
│   ├── evaluation/         # Evaluation benchmarks (e.g., inference timing)
│   │   ├── __init__.py
│   │   └── benchmarks.py
│   └── utils.py            # General utilities
├── tests/                  # Test scripts
│   ├── test_pruning.py     # (Hypothetical) Tests for pruning
│   └── completebias_test.py  # Comprehensive bias visualization test
├── setup.py                # Package setup script
├── README.md               # Project overview and quick start
├── optipfair_llm_reference_manual.txt # Detailed reference for LLMs
└── ... other files (LICENSE, .gitignore, etc.)
```

*   The `optipfair/` directory contains the main library code.
*   `setup.py` manages packaging and dependencies.
*   `tests/` contains test scripts.
*   `README.md` and `optipfair_llm_reference_manual.txt` serve as user-facing documentation.

## 2. Setting up the Development Environment

To contribute to OptiPFair, set up a development environment:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/peremartra/optipfair.git
    cd optipfair
    ```
2.  **Install in editable mode with development dependencies:**
    ```bash
    pip install -e ".[dev,eval,viz]"
    ```
    This installs the core library, plus dependencies for development tools (`pytest`, `black`, `flake8`, `isort`, `mypy`), evaluation (`datasets`, `numpy`, `pandas`, `matplotlib`, `seaborn`), and bias visualization (`matplotlib`, `seaborn`, `numpy`, `pandas`, `scikit-learn`).

## 3. Code Standards and Formatting

OptiPFair uses automated tools to maintain code quality and consistency.

*   **Formatting:** `black` and `isort` are used for code formatting and import sorting.
    ```bash
    # Auto-format code
    black .
    # Auto-sort imports
    isort .
    ```
    Run these tools before committing changes.
*   **Linting:** `flake8` is used for linting (checking for style guide violations and potential errors).
    ```bash
    flake8 .
    ```
    Address any warnings or errors reported by `flake8`.
*   **Type Checking:** `mypy` is used for static type checking.
    ```bash
    mypy optipfair/
    ```
    Ensure type hints are correct and `mypy` passes without errors.

Ideally, these checks should be integrated into pre-commit hooks or CI/CD pipelines.

## 4. Testing

Tests are crucial for ensuring the library's correctness and preventing regressions.

*   Tests are located in the `tests/` directory.
*   `pytest` is the test runner.
*   Run tests from the project root directory:
    ```bash
    pytest
    ```
*   The `completebias_test.py` script provides a comprehensive test for the bias visualization module using a real model. Ensure this test passes.
*   When adding new features or fixing bugs, write corresponding tests.

## 5. Understanding Core Modules

Familiarize yourself with the main components of the library:

*   **Pruning (`optipfair/pruning/`)**:
    *   `mlp_glu_pruning.py`: Contains the core logic for applying structured pruning specifically to MLP layers in GLU architectures. This is where the model modification happens.
    *   `methods.py`: Implements the different neuron importance calculation methods (PPM/MAW, VOW, PON). If adding a new method, this is the place. Note: PPM (Peak-to-Peak Magnitude) is the formal name; \"MAW\" is maintained as parameter for backward compatibility.
    *   `utils.py`: Helper functions for pruning, like identifying GLU layers, calculating parameter counts, etc.
*   **Bias Visualization (`optipfair/bias/`)**:
    *   `activations.py`: Handles the complex task of setting up PyTorch hooks to capture intermediate layer activations. Modifying this requires careful understanding of PyTorch hooks and Transformer model internals.
    *   `metrics.py`: Contains functions to calculate quantitative bias metrics from activation differences. If adding new metrics, modify this file.
    *   `visualization.py`: Responsible for generating plots (matplotlib/seaborn). If adding new visualization types, add functions here.
    *   `defaults.py`: Stores default configurations, including example prompt pairs and layer selection options.
    *   `utils.py`: Utility functions specific to the bias module.
*   **Evaluation (`optipfair/evaluation/`)**:
    *   `benchmarks.py`: Contains functions for evaluating model performance, currently focused on inference timing. Add new evaluation methods here.
*   **CLI (`optipfair/cli.py`)**:
    *   Uses the `click` library to define command-line commands (`prune`, `analyze`). If adding new CLI functionality, extend this file.

## 6. Adding New Features

When adding new features (e.g., a new pruning method, a new visualization type, support for a new model architecture):

1.  **Identify the relevant module(s)**: Based on the feature, determine which files in `optipfair/` need modification or new files need to be created.
2.  **Write code following standards**: Adhere to the established code style, formatting, and type hinting.
3.  **Add tests**: Write tests in the `tests/` directory to cover the new functionality.
4.  **Update documentation**:
    *   Modify `README.md` for user-facing quick starts and overviews.
    *   Update `optipfair_llm_reference_manual.txt` with details about the new API or CLI options.
    *   Consider adding more detailed documentation if the feature is complex (e.g., in a separate `docs/` directory if Sphinx is used, although not present in the provided files).
5.  **Update `setup.py`**: If new dependencies are required, add them to `install_requires` or the appropriate `extras_require` section in `/Users/pere/Documents/GitHub/optipfair/setup.py`.

## 7. Updating Dependencies

Regularly review and update project dependencies listed in `/Users/pere/Documents/GitHub/optipfair/setup.py`.

1.  Check for newer versions of `torch`, `transformers`, `tqdm`, `click`, and the dependencies in `extras_require`.
2.  Update the version specifiers in `setup.py`.
3.  Run `pip install -e ".[dev,eval,viz]"` to install the updated dependencies.
4.  Run all tests (`pytest`) to ensure compatibility with the new dependency versions.
5.  Be mindful of potential breaking changes in major dependency updates (especially `torch` and `transformers`).

## 8. Releasing New Versions

Releasing a new version involves updating the version number and publishing to PyPI.

1.  Update the `version` string in `/Users/pere/Documents/GitHub/optipfair/setup.py`. Follow semantic versioning (e.g., `major.minor.patch`).
2.  Ensure all changes are committed and pushed to the main branch.
3.  Create a Git tag corresponding to the new version number.
    ```bash
    git tag vX.Y.Z
    git push origin vX.Y.Z
    ```
4.  Build the distribution packages (sdist and wheel):
    ```bash
    python setup.py sdist bdist_wheel
    ```
5.  Upload the packages to PyPI using `twine`:
    ```bash
    twine upload dist/*
    ```
    (You may need to install `twine` first: `pip install twine`)

## 9. Troubleshooting

Refer to the "Troubleshooting" section in `/Users/pere/Documents/GitHub/optipfair/optipfair_llm_reference_manual.txt` for common issues and their solutions. Maintainers should keep this section updated as new issues are identified and resolved.

## 10. Contribution Process

Refer to the `CONTRIBUTING.md` file (mentioned in `README.md`) for detailed guidelines on contributing, including branching strategy, pull request process, and code review expectations.

---

By following this guide, developers can effectively maintain and contribute to the OptiPFair library, ensuring its quality, stability, and continued development.