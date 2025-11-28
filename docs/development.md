# Development Guide

## Setting Up Development Environment

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/emotion-xai-project.git
cd emotion-xai-project
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Install Pre-commit Hooks

```bash
pre-commit install
```

## Code Quality Standards

### Formatting
We use Black for code formatting:
```bash
black emotion_xai tests
```

### Import Sorting
We use isort for import organization:
```bash
isort emotion_xai tests
```

### Linting
We use Flake8 for linting:
```bash
flake8 emotion_xai tests
```

### Type Checking
We use mypy for type checking:
```bash
mypy emotion_xai
```

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=emotion_xai --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest tests/unit/     # Unit tests only
pytest tests/integration/  # Integration tests only
```

### Writing Tests
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use fixtures from `tests/conftest.py`
- Follow the AAA pattern (Arrange, Act, Assert)

## Contributing

### Branching Strategy
- `main`: Stable release branch
- `develop`: Development branch
- Feature branches: `feature/your-feature-name`
- Bug fixes: `fix/bug-description`

### Pull Request Process
1. Create feature branch from `develop`
2. Make changes with tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit PR with clear description

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Documentation

### Building Docs Locally
```bash
cd docs
sphinx-build -b html . _build/html
```

### Writing Documentation
- Use clear, concise language
- Include code examples
- Add type hints to all functions
- Document parameters and return values