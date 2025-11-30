"""
Plot Manager for Explainable AI Visualizations

This module provides a centralized plot management system that prevents duplicates,
ensures consistent directory structure, and provides clear visualization utilities.
"""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .explanations import ExplanationResult


class PlotManager:
    """Centralized plot management with deduplication."""
    
    def __init__(self, base_output_dir: str = "results/plots/explainability"):
        """
        Initialize the plot manager.
        
        Args:
            base_output_dir: Base directory for saving plots (relative to project root)
        """
        # Ensure we use project root, not notebook directory
        project_root = Path.cwd()
        if project_root.name == "notebooks":
            project_root = project_root.parent
            
        self.output_dir = project_root / base_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    # Keep track of created plots to prevent duplicates
        self.plot_registry = {}
        self._load_registry()
        
    def _create_plot_hash(self, plot_type: str, **kwargs) -> str:
        """Create a unique hash for plot parameters to prevent duplicates."""
        # Create a consistent hash based on plot type and parameters
        hash_data = {
            'plot_type': plot_type,
            **{k: str(v) for k, v in kwargs.items() if k != 'fig'}  # Exclude figure objects
        }
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()[:12]
    
    def _load_registry(self):
        """Load existing plot registry."""
        registry_file = self.output_dir / "plot_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    self.plot_registry = json.load(f)
            except Exception:
                self.plot_registry = {}
    
    def _save_registry(self):
        """Save plot registry."""
        registry_file = self.output_dir / "plot_registry.json"
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.plot_registry, f, indent=2)
        except Exception:
            pass  # Don't fail if registry can't be saved
    
    def save_plot(self, fig, plot_type: str, formats: List[str] = None, 
                  force_save: bool = False, **kwargs) -> Optional[str]:
        """
        Save a plot with deduplication.
        
        Args:
            fig: Matplotlib or Plotly figure
            plot_type: Type of plot (for naming)
            formats: List of formats to save ['png', 'svg', 'html']
            force_save: Force save even if duplicate exists
            **kwargs: Additional parameters for naming and deduplication
            
        Returns:
            Path to saved file (primary format) or None if duplicate
        """
        if formats is None:
            formats = ['png', 'svg']

        # Compute hash for deduplication
        plot_hash = self._create_plot_hash(plot_type, **kwargs)

        # Check if this plot already exists
        if not force_save and plot_hash in self.plot_registry:
            existing_file = self.plot_registry[plot_hash]
            if Path(existing_file).exists():
                print(f"📋 Plot already exists: {Path(existing_file).name}")
                return existing_file

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include important parameters in filename
        filename_parts = [plot_type]
        if 'emotion' in kwargs:
            filename_parts.append(kwargs['emotion'])
        if 'method' in kwargs:
            filename_parts.append(kwargs['method'])

        base_filename = "_".join(filename_parts)
        filename = f"{base_filename}_{timestamp}"

        # Save in specified formats
        saved_files = []
        primary_file = None

        for fmt in formats:
            file_path = self.output_dir / f"{filename}.{fmt}"

            try:
                if fmt == 'png':
                    if hasattr(fig, 'savefig'):  # Matplotlib
                        fig.savefig(file_path, dpi=300, bbox_inches='tight', 
                                   facecolor='white', edgecolor='none')
                    elif hasattr(fig, 'write_image'):  # Plotly
                        fig.write_image(str(file_path), width=1200, height=800)

                elif fmt == 'svg':
                    if hasattr(fig, 'savefig'):  # Matplotlib
                        fig.savefig(file_path, format='svg', bbox_inches='tight', 
                                   facecolor='white', edgecolor='none')
                    elif hasattr(fig, 'write_image'):  # Plotly
                        fig.write_image(str(file_path), format='svg', width=1200, height=800)

                elif fmt == 'html' and hasattr(fig, 'write_html'):  # Plotly only
                    fig.write_html(str(file_path))

                saved_files.append(str(file_path))
                if primary_file is None:  # First successful save is primary
                    primary_file = str(file_path)

            except Exception as e:
                print(f"⚠️  Could not save {fmt} format: {e}")

        if saved_files:
            # Update registry
            self.plot_registry[plot_hash] = primary_file
            self._save_registry()

            print(f"💾 Plot saved: {Path(primary_file).name}")
            return primary_file
        else:
            print("⚠️  No plot was saved.")
            return None
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Helper function to plot feature importance
        def plot_importance(ax, result, method_name, color_positive, color_negative):
            if (result.feature_importance and emotion in result.feature_importance):
                importance = result.feature_importance[emotion]
                sorted_features = sorted(importance.items(), 
                                       key=lambda x: abs(x[1]), reverse=True)[:top_k]
                
                if sorted_features:
                    words, scores = zip(*sorted_features)
                    
                    # Create colors based on positive/negative
                    colors = [color_positive if s > 0 else color_negative for s in scores]
                    
                    # Create horizontal bar plot
                    bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.8, 
                                  edgecolor='black', linewidth=0.5)
                    
                    # Customize axis
                    ax.set_yticks(range(len(words)))
                    ax.set_yticklabels(words, fontsize=11)
                    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
                    ax.set_title(f'{method_name} - {emotion.title()}', 
                               fontsize=14, fontweight='bold', pad=20)
                    
                    # Add value labels on bars
                    for i, (bar, score) in enumerate(zip(bars, scores)):
                        label_x = score + (0.01 if score > 0 else -0.01)
                        ha = 'left' if score > 0 else 'right'
                        ax.text(label_x, i, f'{score:.3f}', 
                               ha=ha, va='center', fontsize=9, fontweight='bold')
                    
                    # Add vertical line at x=0
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                    
                    # Add grid for better readability
                    ax.grid(True, alpha=0.3, axis='x')
                    ax.set_axisbelow(True)
                    
                else:
                    ax.text(0.5, 0.5, f'No {method_name} data\nfor {emotion}', 
                           ha='center', va='center', transform=ax.transAxes, 
                           fontsize=12, style='italic')
            else:
                ax.text(0.5, 0.5, f'No {method_name} data\nfor {emotion}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, style='italic')
        
        # Plot SHAP (left subplot)
        plot_importance(ax1, shap_result, 'SHAP', '#2E8B57', '#DC143C')  # SeaGreen, Crimson
        
        # Plot LIME (right subplot)  
        plot_importance(ax2, lime_result, 'LIME', '#4169E1', '#FF6347')  # RoyalBlue, Tomato
        
        # Synchronize x-axis limits for better comparison
        all_scores = []
        for result in [shap_result, lime_result]:
            if result.feature_importance and emotion in result.feature_importance:
                scores = list(result.feature_importance[emotion].values())
                all_scores.extend(scores)
        
        if all_scores:
            max_abs = max(abs(s) for s in all_scores)
            xlim = [-max_abs * 1.1, max_abs * 1.1]
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)
        
        # Add main title
        fig.suptitle(f'SHAP vs LIME Comparison for "{emotion.title()}" Emotion', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add prediction confidence info
        shap_conf = shap_result.predictions.get(emotion, 0)
        lime_conf = lime_result.predictions.get(emotion, 0)
        
        fig.text(0.5, 0.02, f'Model Confidence: {shap_conf:.3f} | Text: "{shap_result.text[:60]}..."', 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.87, bottom=0.12)
        
        # Save the plot
        if save_plot:
            self.plot_manager.save_plot(
                fig, 
                plot_type="clear_comparison_shap_lime",
                emotion=emotion,
                formats=['png', 'svg']
            )
        
    
    def plot_feature_importance_enhanced(self,
                                       explanation_result: ExplanationResult,
                                       emotion: str, 
                                       method: str = 'shap',
                                       top_k: int = 12,
                                       save_plot: bool = True) -> plt.Figure:
        """
        Create an enhanced feature importance plot.
        
        Args:
            explanation_result: Result from explainer
            emotion: Emotion to visualize
            method: Method used ('shap' or 'lime')
            top_k: Number of features to show
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        if (not explanation_result.feature_importance or 
            emotion not in explanation_result.feature_importance):
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, f'No {method.upper()} data for emotion: {emotion}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Extract and sort features
        importance = explanation_result.feature_importance[emotion]
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        
        if not sorted_features:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, f'No feature data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        words, scores = zip(*sorted_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create color mapping
        colors = ['#2E8B57' if score > 0 else '#DC143C' for score in scores]  # Green/Red
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=0.8)
        
        # Customize appearance
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=12)
        ax.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
        ax.set_title(f'{method.upper()} Feature Importance for "{emotion.title()}"', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            label_x = score + (max(scores) * 0.02 if score > 0 else min(scores) * 0.02)
            ha = 'left' if score > 0 else 'right'
            ax.text(label_x, i, f'{score:.3f}', ha=ha, va='center', 
                   fontsize=10, fontweight='bold', color='black')
        
        # Add reference line and styling
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        # Add confidence info
        confidence = explanation_result.predictions.get(emotion, 0)
        ax.text(0.98, 0.02, f'Confidence: {confidence:.3f}', 
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        # Save the plot
        if save_plot:
            self.plot_manager.save_plot(
                fig, 
                plot_type="enhanced_feature_importance",
                method=method,
                emotion=emotion,
                formats=['png', 'svg']
            )
        
        return fig
    
    def create_summary_dashboard(self,
                               shap_result: ExplanationResult,
                               lime_result: ExplanationResult,
                               top_emotions: int = 3,
                               save_plot: bool = True) -> plt.Figure:
        """
        Create a comprehensive dashboard showing multiple emotions.
        
        Args:
            shap_result: SHAP results
            lime_result: LIME results 
            top_emotions: Number of top emotions to show
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get top predicted emotions
        top_preds = sorted(shap_result.predictions.items(), 
                          key=lambda x: x[1], reverse=True)[:top_emotions]
        
        fig, axes = plt.subplots(top_emotions, 2, figsize=(18, 6 * top_emotions))
        if top_emotions == 1:
            axes = axes.reshape(1, -1)
        
        for i, (emotion, confidence) in enumerate(top_preds):
            # SHAP subplot
            self._plot_subplot_importance(axes[i, 0], shap_result, emotion, 'SHAP', '#2E8B57', '#DC143C')
            
            # LIME subplot  
            self._plot_subplot_importance(axes[i, 1], lime_result, emotion, 'LIME', '#4169E1', '#FF6347')
        
        fig.suptitle(f'Multi-Emotion Explanation Dashboard\nText: "{shap_result.text[:80]}..."', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_plot:
            self.plot_manager.save_plot(
                fig,
                plot_type="summary_dashboard", 
                formats=['png', 'svg']
            )
        
        return fig
    
    def _plot_subplot_importance(self, ax, result, emotion, method, color_pos, color_neg, top_k=6):
        """Helper method to plot feature importance in a subplot."""
        if (result.feature_importance and emotion in result.feature_importance):
            importance = result.feature_importance[emotion]
            sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
            
            if sorted_features:
                words, scores = zip(*sorted_features)
                colors = [color_pos if s > 0 else color_neg for s in scores]
                
                bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.8)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words)
                ax.set_title(f'{method} - {emotion.title()} ({result.predictions.get(emotion, 0):.3f})')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.7)
                ax.grid(True, alpha=0.3, axis='x')
            else:
                ax.text(0.5, 0.5, f'No data for {emotion}', ha='center', va='center', 
                       transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, f'No data for {emotion}', ha='center', va='center', 
                   transform=ax.transAxes)


def clean_duplicate_plots(base_dir: str = "results/plots/explainability"):
    """
    Clean up duplicate plot files and consolidate to main directory.
    
    Args:
        base_dir: Base directory to clean
    """
    project_root = Path.cwd()
    if project_root.name == "notebooks":
        project_root = project_root.parent
    
    main_plots_dir = project_root / base_dir
    notebooks_plots_dir = project_root / "notebooks" / base_dir
    
    print(f"🧹 Cleaning duplicate plots...")
    print(f"   Main directory: {main_plots_dir}")
    print(f"   Notebooks directory: {notebooks_plots_dir}")
    
    # Create main directory if it doesn't exist
    main_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Move files from notebooks directory to main directory
    moved_count = 0
    if notebooks_plots_dir.exists():
        for file_path in notebooks_plots_dir.rglob("*"):
            if file_path.is_file():
                # Create corresponding path in main directory
                relative_path = file_path.relative_to(notebooks_plots_dir)
                target_path = main_plots_dir / relative_path
                
                # Create parent directories
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file if it doesn't exist in target
                if not target_path.exists():
                    file_path.rename(target_path)
                    moved_count += 1
                    print(f"   Moved: {relative_path}")
                else:
                    # Remove duplicate
                    file_path.unlink()
                    print(f"   Removed duplicate: {relative_path}")
        
        # Remove empty directories
        try:
            notebooks_plots_dir.rmdir()
        except OSError:
            pass  # Directory not empty
    
    print(f"✅ Cleanup complete. Moved {moved_count} files.")
    return main_plots_dir


# Convenience functions
def create_clear_comparison(shap_result: ExplanationResult, 
                          lime_result: ExplanationResult,
                          emotion: str,
                          save_plots: bool = True) -> plt.Figure:
    """Create a clear SHAP vs LIME comparison plot."""
    visualizer = PlotManager()
    return visualizer.plot_clear_comparison(shap_result, lime_result, emotion, save_plot=save_plots)


def create_enhanced_importance(explanation_result: ExplanationResult,
                             emotion: str, 
                             method: str = 'shap',
                             save_plots: bool = True) -> plt.Figure:
    """Create an enhanced feature importance plot."""
    visualizer = PlotManager()
    return visualizer.plot_feature_importance_enhanced(explanation_result, emotion, method, save_plot=save_plots)