# Contributing to Emotion-XAI

We welcome contributions to the Emotion-XAI project! This document provides guidelines for contributing.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find that you don't need to create one. When creating a bug report, please include as many details as possible:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed and what behavior you expected
- Include details about your environment (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- Use a clear and descriptive title
- Provide a detailed description of the suggested enhancement
- Explain why this enhancement would be useful
- List some other packages where this enhancement exists, if applicable

### Pull Requests

1. Fork the repo and create your branch from `develop`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue the pull request

## Development Process

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature branches
- `fix/*`: Bug fix branches

### Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

### Testing

- Write tests for new functionality
- Maintain or improve test coverage
- Use pytest for testing framework
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(models): add BERT-based emotion classifier

Add support for BERT model fine-tuning with custom
attention mechanisms for better emotion detection.

Closes #123
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-xai-project.git
cd emotion-xai-project
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

5. Run tests to ensure everything works:
```bash
pytest
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create pull request to `main`
4. After merge, tag the release
5. GitHub Actions will handle the rest

Thank you for contributing!