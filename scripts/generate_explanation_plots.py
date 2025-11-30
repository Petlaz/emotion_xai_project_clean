#!/usr/bin/env python3
"""
Create comprehensive explainability plots for the emotion classification model.

This script builds and saves various explanation visualizations to results/plots/explainability/
including SHAP explanations, LIME interpretations, and comparative analyses.

Usage:
    python scripts/generate_explanation_plots.py [--text "Custom text to analyze"]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from emotion_xai.explainability.explanations import SHAPExplainer, TransformerExplainer
from emotion_xai.explainability.lime_utils import LIMEExplainer, MultiLabelLIME
from emotion_xai.explainability.visualizations import ExplanationVisualizer, InteractiveVisualizer


def generate_comprehensive_plots(model_path: str, custom_text: str = None):
    """Generate comprehensive explanation plots."""
    
    print("🚀 Initializing explainability framework...")
    
    # Initialize explainers
    shap_explainer = SHAPExplainer(model_path)
    lime_explainer = LIMEExplainer(model_path)
    
    # Initialize visualizers
    static_viz = ExplanationVisualizer(save_plots=True)
    interactive_viz = InteractiveVisualizer(save_plots=True)
    
    # Sample texts for analysis
    sample_texts = {
        "joy_example": "I just got accepted to my dream university! This is the best day of my life!",
        "mixed_emotions": "I'm excited about the new job but nervous about leaving my current team.",
        "complex_narrative": "After years of hard work and rejections, I finally achieved my goal.",
        "custom": custom_text
    }
    
    if custom_text:
        texts_to_analyze = {"custom": custom_text}
        print(f"📝 Analyzing custom text: {custom_text}")
    else:
        texts_to_analyze = {k: v for k, v in sample_texts.items() if k != "custom"}
        print(f"📝 Analyzing {len(texts_to_analyze)} sample texts")
    
    all_results = {}
    
    for name, text in texts_to_analyze.items():
        print(f"\n🔍 Processing: {name}")
        print(f"Text: {text}")
        
        # Generate SHAP explanation
        print("  📊 SHAP analysis...")
        shap_result = shap_explainer.explain_text(text, num_samples=100)
        
        # Generate LIME explanation  
        print("  🍋 LIME analysis...")
        lime_result = lime_explainer.explain_top_emotions(text, top_k=3, num_samples=500)
        
        # Get top emotion for focused visualization
        top_emotion = list(shap_result.predictions.keys())[0]
        
        # Store results
        all_results[name] = {
            'text': text,
            'shap': shap_result,
            'lime': lime_result,
            'top_emotion': top_emotion
        }
        
        print(f"  🎯 Top emotion: {top_emotion} ({shap_result.predictions[top_emotion]:.3f})")
        
        # Generate static visualizations
        print("  📈 Creating static plots...")
        
        # Feature importance (SHAP)
        static_viz.plot_feature_importance(shap_result, top_emotion, method='shap')
        plt.close()
        
        # Feature importance (LIME)
        static_viz.plot_feature_importance(lime_result, top_emotion, method='lime') 
        plt.close()
        
        # Prediction confidence
        static_viz.plot_prediction_confidence(shap_result)
        plt.close()
        
        # SHAP vs LIME comparison
        static_viz.plot_comparison_chart(shap_result, lime_result, top_emotion)
        plt.close()
        
        # Generate interactive visualizations
        print("  🌟 Creating interactive plots...")
        
        # Interactive importance
        interactive_viz.create_interactive_importance(shap_result, top_emotion)
        
        # Interactive predictions
        interactive_viz.create_interactive_predictions(shap_result)
        
        # Comparison dashboard
        comparison_results = {'SHAP': shap_result, 'LIME': lime_result}
        interactive_viz.create_comparison_dashboard(comparison_results, text)
        
        print(f"  ✅ Completed analysis for: {name}")
    
    # Generate summary report
    print(f"\n📋 Creating summary report...")
    
    plots_dir = Path("results/plots/explainability")
    plot_files = list(plots_dir.glob("*"))
    
    print(f"\n🎊 Plot creation complete!")
    print(f"📂 Location: {plots_dir.absolute()}")
    print(f"📊 Total files: {len(plot_files)}")
    
    # File type breakdown
    from collections import Counter
    extensions = [f.suffix for f in plot_files]
    ext_counts = Counter(extensions)
    
    print(f"\n📈 Generated files by type:")
    for ext, count in ext_counts.items():
        print(f"  {ext.upper()[1:]:>4}: {count} files")
    
    # Show recent files
    plot_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"\n📄 Most recent files:")
    for plot_file in plot_files[:10]:  # Show top 10 most recent
        file_size = plot_file.stat().st_size / 1024  # KB
        print(f"  {plot_file.name} ({file_size:.1f} KB)")
    
    return all_results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate explainability plots for emotion classification model"
    )
    parser.add_argument(
        "--text", 
        type=str,
        help="Custom text to analyze (if not provided, will use sample texts)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/distilroberta_production_20251130_044054",
        help="Path to the trained model"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first or provide correct model path")
        return 1
    
    try:
        # Generate plots
        results = generate_comprehensive_plots(str(model_path), args.text)
        print(f"\n🎉 Success! Generated explanations for {len(results)} text(s)")
        print("📊 All plots saved to results/plots/explainability/")
        return 0
    except Exception as e:
        print(f"❌ Error creating plots: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())