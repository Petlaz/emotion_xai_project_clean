"""Test configuration and fixtures."""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from typing import Generator


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample emotion data for testing."""
    return pd.DataFrame({
        'text': [
            "I am so happy today!",
            "This is really frustrating.",
            "I feel excited about the project.",
            "I'm disappointed with the results.",
            "This brings me great joy."
        ],
        'emotions': [
            'joy',
            'anger',
            'excitement',
            'disappointment', 
            'joy'
        ]
    })


@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return [
        "I love this product!",
        "This is the worst service ever.",
        "I'm feeling great today.",
        "This is okay, nothing special.",
        "Amazing experience, highly recommend!"
    ]