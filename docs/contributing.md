# Contributing

Thank you for your interest in contributing to OptiPFair! We welcome contributions from everyone.

For detailed guidelines on how to contribute, please see our [CONTRIBUTING.md](https://github.com/peremartra/optipfair/blob/main/CONTRIBUTING.md) file in the repository.

## Quick Start

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/peremartra/optipfair.git
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
5. Make your changes, add tests, and ensure all tests pass:
   ```bash
   pytest tests/
   ```
6. Submit a pull request

## Development Workflow

Our development workflow follows these steps:

1. **Open an Issue**: Start by opening an issue describing the feature or bug
2. **Discussion**: Discuss the approach with maintainers and community
3. **Implementation**: Make your changes with tests and documentation
4. **Pull Request**: Submit a PR referencing the original issue
5. **Review**: Address any feedback from code review
6. **Merge**: Once approved, your PR will be merged

## Documentation

When adding new features, please update the documentation as well. We use MkDocs for our documentation.

To preview documentation changes locally:

```bash
# Install MkDocs and required plugins
pip install mkdocs mkdocs-material mkdocstrings

# Serve the documentation
mkdocs serve
```

Then open your browser to http://127.0.0.1:8000/ to see the documentation site.

## Code Style

We use the following tools to enforce code style:

- `black` for code formatting
- `isort` for import sorting
- `flake8` for linting
- `mypy` for type checking

You can run these checks with:

```bash
# Format code
black optipfair tests examples
isort optipfair tests examples

# Check code
flake8 optipfair tests examples
mypy optipfair
```