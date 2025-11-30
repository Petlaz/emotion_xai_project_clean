"""
Visualization utilities for explainable AI in emotion classification.

This module provides comprehensive visualization tools for SHAP and LIME explanations,
including interactive plots, attention heatmaps, and feature importance visualizations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .explanations import ExplanationResult


class ExplanationVisualizer:
    """Main visualizer class for explanation results."""
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (12, 8), 
                 save_plots: bool = True, output_dir: str = "results/plots/explainability"):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
            save_plots: Whether to automatically save plots
            output_dir: Directory to save plots to
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        if self.save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_figure(self, fig: plt.Figure, filename: str, timestamp: bool = True) -> str:
        """
        Save figure to the output directory.
        
        Args:
            fig: Matplotlib figure to save
            filename: Base filename (without extension)
            timestamp: Whether to add timestamp to filename
            
        Returns:
            Full path to saved file
        """
        if not self.save_plots:
            return ""
        
        if timestamp:
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp_str}"
        
        # Save as both PNG and SVG for flexibility
        png_path = self.output_dir / f"{filename}.png"
        svg_path = self.output_dir / f"{filename}.svg"
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
        
        print(f"📊 Plot saved: {png_path}")
        return str(png_path)
        
    def plot_feature_importance(self, 
                               explanation_result: ExplanationResult, 
                               emotion: str,
                               top_k: int = 10,
                               method: str = 'shap') -> plt.Figure:
        """
        Plot feature importance for a specific emotion.
        
        Args:
            explanation_result: Result from explainer
            emotion: Emotion to visualize
            top_k: Number of top features to show
            method: Method used ('shap' or 'lime')
            
        Returns:
            Matplotlib figure
        """
        if not explanation_result.feature_importance or emotion not in explanation_result.feature_importance:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, f'No feature importance data for {emotion}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Extract feature importance
        importance = explanation_result.feature_importance[emotion]
        
        # Sort by absolute importance
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        words, scores = zip(*sorted_features) if sorted_features else ([], [])
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Color bars based on positive/negative
        colors = ['green' if score > 0 else 'red' for score in scores]
        
        bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.7)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel(f'{method.upper()} Importance Score')
        ax.set_title(f'{method.upper()} Feature Importance for \"{emotion}\"', fontsize=14, fontweight='bold')
        
        # Add score labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + (0.01 if score > 0 else -0.01), i, f'{score:.3f}', 
                   ha='left' if score > 0 else 'right', va='center')
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # Save the plot
        if self.save_plots:
            self._save_figure(fig, f"feature_importance_{method}_{emotion}")
        
        return fig
    
    def plot_prediction_confidence(self, 
                                  explanation_result: ExplanationResult, 
                                  top_k: int = 10) -> plt.Figure:
        """
        Plot prediction confidence scores.
        
        Args:
            explanation_result: Result from explainer
            top_k: Number of top predictions to show
            
        Returns:
            Matplotlib figure
        """
        predictions = explanation_result.predictions
        
        # Sort predictions
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
        emotions, scores = zip(*sorted_preds) if sorted_preds else ([], [])
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.barh(range(len(emotions)), scores, color=self.colors[:len(emotions)])
        ax.set_yticks(range(len(emotions)))
        ax.set_yticklabels(emotions)
        ax.set_xlabel('Prediction Confidence')
        ax.set_title('Model Prediction Confidence', fontsize=14, fontweight='bold')
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.005, i, f'{score:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        # Save the plot
        if self.save_plots:
            self._save_figure(fig, "prediction_confidence")
        
        return fig
    
    def plot_attention_heatmap(self, 
                              tokens: List[str], 
                              attention_weights: np.ndarray,
                              layer: int = -1,
                              head: int = -1) -> plt.Figure:
        """
        Plot attention weights as a heatmap.
        
        Args:
            tokens: List of tokens
            attention_weights: Attention weights array
            layer: Layer to visualize (-1 for average across all)
            head: Head to visualize (-1 for average across all)
            
        Returns:
            Matplotlib figure
        """
        if attention_weights.ndim == 4:  # [batch, layers, heads, seq_len, seq_len]
            attention = attention_weights[0]  # Take first batch
        else:
            attention = attention_weights
        
        # Select layer and head
        if layer >= 0 and layer < attention.shape[0]:
            attention = attention[layer]
        else:
            attention = attention.mean(axis=0)  # Average across layers
            
        if attention.ndim == 3:  # Still have heads dimension
            if head >= 0 and head < attention.shape[0]:
                attention = attention[head]
            else:
                attention = attention.mean(axis=0)  # Average across heads
        
        # Limit tokens for readability
        max_tokens = 20
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            attention = attention[:max_tokens, :max_tokens]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), max(8, len(tokens) * 0.5)))
        
        sns.heatmap(attention, 
                   xticklabels=tokens,
                   yticklabels=tokens,
                   cmap='Blues',
                   ax=ax,
                   cbar_kws={'label': 'Attention Weight'})
        
        ax.set_title(f'Attention Heatmap (Layer {layer if layer >= 0 else "Average"}, '
                    f'Head {head if head >= 0 else "Average"})', fontweight='bold')
        ax.set_xlabel('Tokens (To)')
        ax.set_ylabel('Tokens (From)')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save the plot
        if self.save_plots:
            self._save_figure(fig, f"attention_heatmap_layer{layer}_head{head}")
        
        return fig
    
    def plot_comparison_chart(self, 
                             shap_result: ExplanationResult, 
                             lime_result: ExplanationResult,
                             emotion: str,
                             top_k: int = 8) -> plt.Figure:
        """
        Compare SHAP and LIME explanations side by side.
        
        Args:
            shap_result: SHAP explanation result
            lime_result: LIME explanation result
            emotion: Emotion to compare
            top_k: Number of features to compare
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # SHAP subplot
        if (shap_result.feature_importance and 
            emotion in shap_result.feature_importance):
            
            shap_importance = shap_result.feature_importance[emotion]
            sorted_shap = sorted(shap_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
            words, scores = zip(*sorted_shap) if sorted_shap else ([], [])
            
            colors = ['green' if s > 0 else 'red' for s in scores]
            ax1.barh(range(len(words)), scores, color=colors, alpha=0.7)
            ax1.set_yticks(range(len(words)))
            ax1.set_yticklabels(words)
            ax1.set_title(f'SHAP - {emotion}', fontweight='bold')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # LIME subplot
        if (lime_result.feature_importance and 
            emotion in lime_result.feature_importance):
            
            lime_importance = lime_result.feature_importance[emotion]
            sorted_lime = sorted(lime_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
            words, scores = zip(*sorted_lime) if sorted_lime else ([], [])
            
            colors = ['green' if s > 0 else 'red' for s in scores]
            ax2.barh(range(len(words)), scores, color=colors, alpha=0.7)
            ax2.set_yticks(range(len(words)))
            ax2.set_yticklabels(words)
            ax2.set_title(f'LIME - {emotion}', fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # Save the plot
        if self.save_plots:
            self._save_figure(fig, f"comparison_shap_lime_{emotion}")
        
        return fig


class InteractiveVisualizer:
    """Interactive visualizations using Plotly."""
    
    def __init__(self, save_plots: bool = True, output_dir: str = "results/plots/explainability"):
        """
        Initialize interactive visualizer.
        
        Args:
            save_plots: Whether to automatically save plots
            output_dir: Directory to save plots to
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive visualizations. "
                            "Install with: pip install plotly")
        
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        if self.save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_plotly_figure(self, fig: go.Figure, filename: str, timestamp: bool = True) -> str:
        """
        Save Plotly figure to the output directory.
        
        Args:
            fig: Plotly figure to save
            filename: Base filename (without extension)
            timestamp: Whether to add timestamp to filename
            
        Returns:
            Full path to saved file
        """
        if not self.save_plots:
            return ""
        
        if timestamp:
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp_str}"
        
        # Save as HTML for interactivity
        html_path = self.output_dir / f"{filename}.html"
        fig.write_html(str(html_path))
        
        # Also save as PNG for static use
        png_path = self.output_dir / f"{filename}.png"
        try:
            fig.write_image(str(png_path), width=1200, height=800)
        except Exception:
            # If kaleido is not available, skip PNG export
            pass
        
        print(f"📊 Interactive plot saved: {html_path}")
        return str(html_path)
    
    def create_interactive_importance(self, 
                                    explanation_result: ExplanationResult,
                                    emotion: str,
                                    top_k: int = 15) -> go.Figure:
        """
        Create interactive feature importance plot.
        
        Args:
            explanation_result: Result from explainer
            emotion: Emotion to visualize
            top_k: Number of features to show
            
        Returns:
            Plotly figure
        """
        if (not explanation_result.feature_importance or 
            emotion not in explanation_result.feature_importance):
            fig = go.Figure()
            fig.add_annotation(text=f"No data for emotion: {emotion}", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        importance = explanation_result.feature_importance[emotion]
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        words, scores = zip(*sorted_features) if sorted_features else ([], [])
        
        colors = ['rgba(0, 128, 0, 0.7)' if s > 0 else 'rgba(255, 0, 0, 0.7)' for s in scores]
        
        fig = go.Figure(data=[
            go.Bar(
                y=words,
                x=scores,
                orientation='h',
                marker=dict(color=colors),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f'Feature Importance for "{emotion}"',
            xaxis_title='Importance Score',
            yaxis_title='Words',
            height=max(400, len(words) * 25),
            showlegend=False
        )
        
        # Add vertical line at x=0
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        
        # Save the plot
        if self.save_plots:
            self._save_plotly_figure(fig, f"interactive_importance_{emotion}")
        
        return fig
    
    def create_interactive_predictions(self, 
                                     explanation_result: ExplanationResult,
                                     top_k: int = 10) -> go.Figure:
        """
        Create interactive prediction confidence plot.
        
        Args:
            explanation_result: Result from explainer
            top_k: Number of predictions to show
            
        Returns:
            Plotly figure
        """
        predictions = explanation_result.predictions
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
        emotions, scores = zip(*sorted_preds) if sorted_preds else ([], [])
        
        fig = go.Figure(data=[
            go.Bar(
                y=emotions,
                x=scores,
                orientation='h',
                marker=dict(color='rgba(55, 126, 184, 0.7)'),
                hovertemplate='<b>%{y}</b><br>Confidence: %{x:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Model Prediction Confidence',
            xaxis_title='Confidence Score',
            yaxis_title='Emotions',
            height=max(400, len(emotions) * 30),
            showlegend=False
        )
        
        # Save the plot
        if self.save_plots:
            self._save_plotly_figure(fig, "interactive_predictions")
        
        return fig
    
    def create_comparison_dashboard(self, 
                                  results: Dict[str, ExplanationResult],
                                  text: str) -> go.Figure:
        """
        Create a dashboard comparing multiple explanation methods.
        
        Args:
            results: Dictionary mapping method names to results
            text: Original text being explained
            
        Returns:
            Plotly figure with subplots
        """
        methods = list(results.keys())
        n_methods = len(methods)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=n_methods,
            subplot_titles=[f'{method.upper()} Predictions' for method in methods] + 
                          [f'{method.upper()} Feature Importance' for method in methods],
            specs=[[{"type": "bar"}] * n_methods] + [[{"type": "bar"}] * n_methods],
            vertical_spacing=0.15
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, (method, result) in enumerate(results.items(), 1):
            # Top predictions subplot
            predictions = result.predictions
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
            emotions, scores = zip(*sorted_preds) if sorted_preds else ([], [])
            
            fig.add_trace(
                go.Bar(y=emotions, x=scores, orientation='h', 
                      name=f'{method}_pred', showlegend=False,
                      marker_color=colors[i % len(colors)]),
                row=1, col=i
            )
            
            # Feature importance subplot (for top emotion)
            if emotions and result.feature_importance:
                top_emotion = emotions[0]
                if top_emotion in result.feature_importance:
                    importance = result.feature_importance[top_emotion]
                    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                    words, imp_scores = zip(*sorted_features) if sorted_features else ([], [])
                    
                    bar_colors = ['rgba(0, 128, 0, 0.7)' if s > 0 else 'rgba(255, 0, 0, 0.7)' 
                                 for s in imp_scores]
                    
                    fig.add_trace(
                        go.Bar(y=words, x=imp_scores, orientation='h',
                              name=f'{method}_imp', showlegend=False,
                              marker_color=bar_colors),
                        row=2, col=i
                    )
        
        # Update layout
        fig.update_layout(
            title=f'Multi-Method Explanation Dashboard<br><sub>Text: {text[:100]}...</sub>',
            height=800,
            showlegend=False
        )
        
        # Update axes
        for i in range(1, n_methods + 1):
            fig.update_xaxes(title_text="Confidence", row=1, col=i)
            fig.update_xaxes(title_text="Importance", row=2, col=i)
        
        # Save the plot
        if self.save_plots:
            self._save_plotly_figure(fig, "comparison_dashboard")
        
        return fig


def create_explanation_report(results: Dict[str, ExplanationResult], 
                            text: str, 
                            save_path: Optional[str] = None) -> str:
    """
    Create a comprehensive HTML report of explanations.
    
    Args:
        results: Dictionary mapping method names to explanation results
        text: Original text
        save_path: Optional path to save the report
        
    Returns:
        HTML content as string
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emotion Classification Explanation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
            .text-box {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; 
                        border-left: 4px solid #007acc; }}
            .method-section {{ margin: 30px 0; padding: 20px; 
                             border: 1px solid #ddd; border-radius: 8px; }}
            .predictions {{ display: flex; flex-wrap: wrap; gap: 10px; }}
            .emotion-tag {{ padding: 5px 10px; background-color: #e7f3ff; 
                          border-radius: 15px; font-size: 0.9em; }}
            .importance-list {{ list-style-type: none; padding: 0; }}
            .importance-item {{ padding: 8px; margin: 2px 0; border-radius: 5px; }}
            .positive {{ background-color: #d4edda; }}
            .negative {{ background-color: #f8d7da; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎭 Emotion Classification Explanation Report</h1>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="text-box">
            <h3>📝 Analyzed Text:</h3>
            <p><em>"{text}"</em></p>
        </div>
    """
    
    for method, result in results.items():
        html_content += f"""
        <div class="method-section">
            <h2>🔍 {method.upper()} Analysis</h2>
            
            <h3>🎯 Top Predictions:</h3>
            <div class="predictions">
        """
        
        # Add predictions
        sorted_preds = sorted(result.predictions.items(), key=lambda x: x[1], reverse=True)[:5]
        for emotion, score in sorted_preds:
            html_content += f'<span class="emotion-tag">{emotion}: {score:.3f}</span>'
        
        html_content += """
            </div>
            
            <h3>📊 Feature Importance:</h3>
        """
        
        # Add feature importance
        if result.feature_importance:
            for emotion, importance in list(result.feature_importance.items())[:2]:
                html_content += f"<h4>{emotion.title()}:</h4><ul class='importance-list'>"
                
                sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                for word, score in sorted_features:
                    css_class = "positive" if score > 0 else "negative"
                    direction = "↗️" if score > 0 else "↘️"
                    html_content += (f'<li class="importance-item {css_class}">'
                                   f'{word}: {score:.3f} {direction}</li>')
                
                html_content += "</ul>"
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    if save_path:
        Path(save_path).write_text(html_content)
        print(f"Report saved to: {save_path}")
    
    return html_content


# Convenience functions
def quick_visualize(explanation_result: ExplanationResult, 
                   emotion: str, 
                   method: str = 'shap',
                   save_plots: bool = True) -> plt.Figure:
    """Quick visualization of explanation results."""
    visualizer = ExplanationVisualizer(save_plots=save_plots)
    return visualizer.plot_feature_importance(explanation_result, emotion, method=method)


def compare_methods(shap_result: ExplanationResult, 
                   lime_result: ExplanationResult, 
                   emotion: str,
                   save_plots: bool = True) -> plt.Figure:
    """Quick comparison of SHAP and LIME results."""
    visualizer = ExplanationVisualizer(save_plots=save_plots)
    return visualizer.plot_comparison_chart(shap_result, lime_result, emotion)