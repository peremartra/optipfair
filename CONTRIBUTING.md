# Contributing to OptiPFair

Thank you for your interest in contributing to OptiPFair! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Future Roadmap](#future-roadmap)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. We expect all contributors to be respectful, considerate, and constructive in their communications and actions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/optipfair.git
   cd optipfair
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Process

1. **Check existing issues**: Look for existing issues or create a new one to discuss your proposed changes.
2. **Design your changes**: For significant changes, consider discussing the design in the issue first.
3. **Implement your changes**: Follow the coding standards below.
4. **Write tests**: Add or update tests for your changes.
5. **Update documentation**: Ensure documentation is updated to reflect your changes.
6. **Submit a pull request**: After your changes are complete and tested, submit a pull request.

## Pull Request Process

1. Ensure your code follows the project's coding standards and passes all tests.
2. Update the documentation with details of changes to the interface, new features, or important notes.
3. Update the CHANGELOG.md with details of your changes.
4. Submit a pull request against the `main` branch and request a review.
5. Address any feedback or comments from reviewers.

## Coding Standards

OptiPFair follows these coding standards:

1. **PEP 8**: Follow Python PEP 8 style guide.
2. **Type Hints**: Use type hints for function parameters and return values.
3. **Docstrings**: Document all functions, classes, and modules using Google style docstrings.
4. **Formatting**: Code is formatted using `black` with a line length of 100 characters.
5. **Imports**: Use `isort` to sort imports consistently.
6. **Naming Conventions**:
   - Use `snake_case` for functions and variables
   - Use `PascalCase` for classes
   - Use `UPPER_CASE` for constants

We use the following tools to enforce these standards:

- `black`: Code formatting
- `isort`: Import sorting
- `flake8`: Linting
- `mypy`: Type checking

You can run all of these checks with:

```bash
# Format code
black optipfair tests examples
isort optipfair tests examples

# Check code
flake8 optipfair tests examples
mypy optipfair
```

## Testing

All new features or bug fixes should include tests. We use `pytest` for testing.

To run the tests:

```bash
pytest tests/
```

For new features:
- Add unit tests for each function or method
- Add integration tests for interactions between components
- Ensure tests cover both normal behavior and error cases

## Documentation

Documentation is a crucial part of the project. Please follow these guidelines:

1. **Docstrings**: Every public function, method, and class should have a docstring explaining:
   - What it does
   - Parameters and their types
   - Return values and their types
   - Exceptions raised
   - Examples (where appropriate)

2. **Markdown Documentation**: Update the relevant markdown files in the `docs/` directory.

3. **README**: Update the README.md if your changes affect the installation, basic usage, or other key aspects.

## Future Roadmap

OptiPFair is an evolving project with plans for several future enhancements. If you're interested in contributing to these areas, please join the discussion in the related issues:

1. **Attention Layer Pruning**: Implementation of structured pruning for attention mechanisms.
2. **Bias visualisations**: Implement visualizations of bias in pair prompts. 
3. **Block Pruning**: Methods for pruning entire transformer blocks.
4. **Evaluation Framework**: Comprehensive evaluation suite for pruned models.
5. **Fine-tuning Integration**: Tools for fine-tuning after pruning.

## Questions?

If you have any questions or need help, please create an issue or reach out to the maintainers.

Thank you for contributing to OptiPFair!