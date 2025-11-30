#!/usr/bin/env python3
"""
Test script for improved explainability plotting system.

This script demonstrates the new plot manager with deduplication,
creates clear comparison plots, and removes duplicate files.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from emotion_xai.explainability.explanations import SHAPExplainer, TransformerExplainer
from emotion_xai.explainability.lime_utils import LIMEExplainer
from emotion_xai.explainability.plot_manager import (
    ImprovedExplanationVisualizer, 
    clean_duplicate_plots,
    create_clear_comparison,
    create_enhanced_importance
)

def main():
    """Main function to test improved plotting system."""
    print("🎭 Testing Improved Explainability Plotting System")
    print("=" * 60)
    
    # Clean up existing duplicates first
    print("\n1. Cleaning up duplicate plots...")
    main_plots_dir = clean_duplicate_plots()
    
    # Initialize explainers
    print("\n2. Initializing explainers...")
    MODEL_PATH = "models/distilroberta_production_20251130_044054"
    
    try:
        shap_explainer = SHAPExplainer(MODEL_PATH)
        lime_explainer = LIMEExplainer(MODEL_PATH)
        print("✅ Explainers initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing explainers: {e}")
        return
    
    # Test with sample text
    print("\n3. Creating explanations...")
    test_text = "I'm so excited about this new opportunity, but also nervous about the challenges ahead."
    
    try:
        # Get explanations
        print(f"   Analyzing: {test_text}")
        shap_result = shap_explainer.explain_text(test_text, num_samples=50)
        lime_result = lime_explainer.explain_top_emotions(test_text, top_k=3)
        
        # Get top emotion
        top_emotion = list(shap_result.predictions.keys())[0]
        print(f"   Top emotion detected: {top_emotion}")
        
    except Exception as e:
        print(f"❌ Error creating explanations: {e}")
        return
    
    # Create improved visualizations
    print("\n4. Creating improved visualizations...")
    visualizer = ImprovedExplanationVisualizer()
    
    try:
        # Test 1: Clear comparison plot (this addresses the original issue)
        print("   📊 Creating clear SHAP vs LIME comparison...")
        comparison_fig = visualizer.plot_clear_comparison(
            shap_result, lime_result, top_emotion, save_plot=True
        )
        
        # Test 2: Enhanced feature importance plots
        print("   📈 Creating enhanced SHAP importance plot...")
        shap_fig = visualizer.plot_feature_importance_enhanced(
            shap_result, top_emotion, method='shap', save_plot=True
        )
        
        print("   🍋 Creating enhanced LIME importance plot...")
        lime_fig = visualizer.plot_feature_importance_enhanced(
            lime_result, top_emotion, method='lime', save_plot=True
        )
        
        # Test 3: Summary dashboard
        print("   📋 Creating summary dashboard...")
        dashboard_fig = visualizer.create_summary_dashboard(
            shap_result, lime_result, top_emotions=2, save_plot=True
        )
        
        print("✅ All visualizations created successfully")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        return
    
    # Test deduplication
    print("\n5. Testing deduplication...")
    try:
        print("   Attempting to create duplicate comparison plot...")
        duplicate_fig = visualizer.plot_clear_comparison(
            shap_result, lime_result, top_emotion, save_plot=True
        )
        print("   (Should show 'Plot already exists' message)")
        
    except Exception as e:
        print(f"❌ Error testing deduplication: {e}")
    
    # Show final results
    print(f"\n6. Final results:")
    print(f"   📁 Plots saved to: {main_plots_dir}")
    
    # List all files
    if main_plots_dir.exists():
        plot_files = list(main_plots_dir.glob("*"))
        print(f"   📊 Total plots: {len(plot_files)}")
        
        for plot_file in sorted(plot_files):
            if plot_file.suffix in ['.png', '.svg', '.html']:
                size_kb = plot_file.stat().st_size / 1024
                print(f"      {plot_file.name} ({size_kb:.1f} KB)")
    
    print("\n✨ Testing complete! The improved system should:")
    print("   ✅ Prevent duplicate plots")
    print("   ✅ Create clearer comparison charts")
    print("   ✅ Use consistent directory structure")
    print("   ✅ Provide better file organization")


def test_convenience_functions():
    """Test the convenience functions."""
    print("\n🧪 Testing convenience functions...")
    
    MODEL_PATH = "models/distilroberta_production_20251130_044054"
    test_text = "This is amazing news! I couldn't be happier about this outcome."
    
    try:
        # Initialize explainers
        shap_explainer = SHAPExplainer(MODEL_PATH)
        lime_explainer = LIMEExplainer(MODEL_PATH)
        
        # Get explanations
        shap_result = shap_explainer.explain_text(test_text)
        lime_result = lime_explainer.explain_top_emotions(test_text, top_k=2)
        
        top_emotion = list(shap_result.predictions.keys())[0]
        
        # Test convenience functions
        print("   📊 Testing create_clear_comparison()...")
        fig1 = create_clear_comparison(shap_result, lime_result, top_emotion)
        
        print("   📈 Testing create_enhanced_importance()...")
        fig2 = create_enhanced_importance(shap_result, top_emotion, method='shap')
        
        print("✅ Convenience functions work correctly")
        
    except Exception as e:
        print(f"❌ Error testing convenience functions: {e}")


if __name__ == "__main__":
    main()
    test_convenience_functions()