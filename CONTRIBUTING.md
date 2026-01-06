# Contributing to Image-to-Text Generator

Thank you for your interest in contributing to this project!

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Include Python version and OS
3. Provide steps to reproduce
4. Include error messages/stack traces

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes following PEP 8 style guide
4. Write/update tests as needed
5. Submit PR with clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/image-to-text-generator.git
cd image-to-text-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Commit Messages

Use conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding/updating tests

Example: `feat: add nucleus sampling to caption generation`

## Questions?

Open an issue or contact tharunponnam007@gmail.com

Thank you for helping improve this project! 🙏
