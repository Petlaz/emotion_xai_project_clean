"""
Emotion-XAI Hugging Face Spaces Application
Phase 6: Production-ready Gradio Interface for HF Spaces

This is the main entry point for the Hugging Face Spaces deployment.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main gradio app
from app.gradio_app import main

if __name__ == "__main__":
    main()