# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with professional Python package layout
- Modular architecture with separate packages for data, models, explainability, and clustering
- Comprehensive test suite with unit and integration tests
- Configuration management system with YAML support
- CLI interface for common tasks
- Pre-commit hooks for code quality
- CI/CD pipeline with GitHub Actions
- Docker support for containerized deployment
- Comprehensive documentation structure

### Changed
- Reorganized source code from flat `src/` structure to modular `emotion_xai/` package
- Updated dependencies to use flexible version ranges
- Improved project metadata with pyproject.toml

## [0.1.0] - 2024-11-28

### Added
- Initial release with basic emotion analysis functionality
- Baseline TF-IDF + Logistic Regression model
- Transformer fine-tuning utilities (placeholder)
- Explainability framework (SHAP/LIME placeholders)
- Clustering analysis utilities (placeholder)
- Gradio web interface
- Basic data preprocessing utilities