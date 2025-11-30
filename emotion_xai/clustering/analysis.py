"""
Advanced cluster analysis and theme interpretation utilities.

This module provides comprehensive analysis tools for understanding clusters,
extracting themes, and evaluating clustering quality for emotion analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter, defaultdict
import logging
from pathlib import Path
from datetime import datetime
import json

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Text analysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Setup logging
logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    """Comprehensive cluster analysis and interpretation."""
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize cluster analyzer.
        
        Args:
            output_dir: Directory for saving analysis results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("results/plots/clustering")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis results storage
        self.cluster_stats = {}
        self.theme_analysis = {}
        self.quality_metrics = {}
        
        logger.info(f"✅ Cluster analyzer initialized, output: {self.output_dir}")
    
    def analyze_clusters(
        self,
        embeddings_2d: np.ndarray,
        cluster_labels: np.ndarray,
        texts: List[str],
        emotions: Optional[np.ndarray] = None,
        emotion_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive cluster analysis.
        
        Args:
            embeddings_2d: 2D embeddings for visualization
            cluster_labels: Cluster assignments
            texts: Original text data
            emotions: Emotion labels/probabilities
            emotion_names: Names of emotion categories
            
        Returns:
            Complete analysis results
        """
        logger.info(f"🔍 Analyzing {len(np.unique(cluster_labels))} clusters")
        
        # Basic cluster statistics
        self.cluster_stats = self._calculate_cluster_stats(
            embeddings_2d, cluster_labels, texts
        )
        
        # Quality metrics
        self.quality_metrics = self._calculate_quality_metrics(
            embeddings_2d, cluster_labels
        )
        
        # Theme extraction
        self.theme_analysis = self._extract_themes(cluster_labels, texts)
        
        # Emotion analysis if provided
        emotion_analysis = None
        if emotions is not None:
            emotion_analysis = self._analyze_cluster_emotions(
                cluster_labels, emotions, emotion_names
            )
        
        # Compile results
        results = {
            'cluster_stats': self.cluster_stats,
            'quality_metrics': self.quality_metrics,
            'theme_analysis': self.theme_analysis,
            'emotion_analysis': emotion_analysis,
            'summary': self._compile_summary()
        }
        
        logger.info("✅ Cluster analysis completed")
        return results
    
    def _calculate_cluster_stats(
        self,
        embeddings_2d: np.ndarray,
        cluster_labels: np.ndarray,
        texts: List[str]
    ) -> Dict:
        """Calculate basic cluster statistics."""
        stats = {}
        
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_points = embeddings_2d[mask]
            cluster_texts = [texts[i] for i in np.where(mask)[0]]
            
            # Basic stats
            cluster_stats = {
                'size': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(cluster_labels) * 100),
                'centroid': np.mean(cluster_points, axis=0).tolist(),
                'std': np.std(cluster_points, axis=0).tolist(),
                'sample_texts': cluster_texts[:10],  # First 10 examples
                'text_length_stats': {
                    'mean': float(np.mean([len(t) for t in cluster_texts])),
                    'std': float(np.std([len(t) for t in cluster_texts])),
                    'min': int(min([len(t) for t in cluster_texts])),
                    'max': int(max([len(t) for t in cluster_texts]))
                }
            }
            
            cluster_name = "noise" if label == -1 else f"cluster_{label}"
            stats[cluster_name] = cluster_stats
        
        return stats
    
    def _calculate_quality_metrics(
        self,
        embeddings_2d: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict:
        """Calculate clustering quality metrics."""
        # Filter out noise points for some metrics
        non_noise_mask = cluster_labels != -1
        n_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise_points': int(np.sum(~non_noise_mask)),
            'noise_percentage': float(np.sum(~non_noise_mask) / len(cluster_labels) * 100)
        }
        
        # Calculate silhouette score for non-noise points
        if np.sum(non_noise_mask) > 1 and n_clusters > 1:
            try:
                silhouette_avg = silhouette_score(
                    embeddings_2d[non_noise_mask], 
                    cluster_labels[non_noise_mask]
                )
                
                # Individual silhouette scores
                silhouette_samples_scores = silhouette_samples(
                    embeddings_2d[non_noise_mask],
                    cluster_labels[non_noise_mask]
                )
                
                metrics.update({
                    'silhouette_score': float(silhouette_avg),
                    'silhouette_samples': silhouette_samples_scores.tolist()
                })
                
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score: {e}")
                metrics['silhouette_score'] = 0.0
        else:
            metrics['silhouette_score'] = 0.0
        
        # Cluster density and separation
        if n_clusters > 1:
            metrics.update(self._calculate_cluster_density_separation(embeddings_2d, cluster_labels))
        
        return metrics
    
    def _calculate_cluster_density_separation(
        self,
        embeddings_2d: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict:
        """Calculate cluster density and separation metrics."""
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        
        intra_distances = []
        inter_distances = []
        
        # Calculate intra-cluster distances (within clusters)
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_points = embeddings_2d[mask]
            
            if len(cluster_points) > 1:
                # Average pairwise distance within cluster
                distances = []
                for i in range(len(cluster_points)):
                    for j in range(i + 1, len(cluster_points)):
                        dist = np.linalg.norm(cluster_points[i] - cluster_points[j])
                        distances.append(dist)
                
                if distances:
                    intra_distances.extend(distances)
        
        # Calculate inter-cluster distances (between cluster centroids)
        centroids = []
        for label in unique_labels:
            mask = cluster_labels == label
            centroid = np.mean(embeddings_2d[mask], axis=0)
            centroids.append(centroid)
        
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                inter_distances.append(dist)
        
        return {
            'avg_intra_cluster_distance': float(np.mean(intra_distances)) if intra_distances else 0.0,
            'avg_inter_cluster_distance': float(np.mean(inter_distances)) if inter_distances else 0.0,
            'separation_ratio': float(np.mean(inter_distances) / np.mean(intra_distances)) if intra_distances and inter_distances else 0.0
        }
    
    def _extract_themes(self, cluster_labels: np.ndarray, texts: List[str]) -> Dict:
        """Extract key themes from clusters using TF-IDF."""
        themes = {}
        
        # Initialize TF-IDF
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_texts = [texts[i] for i in np.where(mask)[0]]
            
            if len(cluster_texts) < 3:  # Skip small clusters
                continue
            
            try:
                # Fit TF-IDF on cluster texts
                tfidf_matrix = tfidf.fit_transform(cluster_texts)
                feature_names = tfidf.get_feature_names_out()
                
                # Get average TF-IDF scores
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                
                # Top keywords
                top_indices = np.argsort(mean_scores)[-20:][::-1]
                top_keywords = [
                    {
                        'word': feature_names[i],
                        'score': float(mean_scores[i])
                    }
                    for i in top_indices
                ]
                
                themes[f"cluster_{label}"] = {
                    'top_keywords': top_keywords,
                    'size': len(cluster_texts),
                    'sample_texts': cluster_texts[:5]
                }
                
            except Exception as e:
                logger.warning(f"Could not extract themes for cluster {label}: {e}")
        
        return themes
    
    def _analyze_cluster_emotions(
        self,
        cluster_labels: np.ndarray,
        emotions: np.ndarray,
        emotion_names: Optional[List[str]] = None
    ) -> Dict:
        """Analyze emotion distributions within clusters."""
        emotion_analysis = {}
        
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_emotions = emotions[mask]
            
            if len(cluster_emotions.shape) == 2:  # Multi-label emotions
                # Average emotion scores
                emotion_means = np.mean(cluster_emotions, axis=0)
                emotion_stds = np.std(cluster_emotions, axis=0)
                
                # Top emotions
                top_indices = np.argsort(emotion_means)[-5:][::-1]
                
                emotion_stats = {
                    'emotion_means': emotion_means.tolist(),
                    'emotion_stds': emotion_stds.tolist(),
                    'top_emotions': [
                        {
                            'emotion': emotion_names[i] if emotion_names else f"emotion_{i}",
                            'score': float(emotion_means[i]),
                            'std': float(emotion_stds[i])
                        }
                        for i in top_indices
                    ]
                }
            else:  # Single-label emotions
                # Count occurrences
                emotion_counts = Counter(cluster_emotions)
                total = len(cluster_emotions)
                
                emotion_stats = {
                    'emotion_distribution': {
                        emotion_names[k] if emotion_names else f"emotion_{k}": v / total
                        for k, v in emotion_counts.items()
                    },
                    'dominant_emotion': emotion_names[emotion_counts.most_common(1)[0][0]] if emotion_names else f"emotion_{emotion_counts.most_common(1)[0][0]}"
                }
            
            emotion_analysis[f"cluster_{label}"] = emotion_stats
        
        return emotion_analysis
    
    def _compile_summary(self) -> Dict:
        """Compile high-level summary of cluster analysis."""
        n_clusters = len([k for k in self.cluster_stats.keys() if k != 'noise'])
        total_points = sum(stats['size'] for stats in self.cluster_stats.values())
        
        # Largest clusters
        largest_clusters = sorted(
            [(k, v['size']) for k, v in self.cluster_stats.items() if k != 'noise'],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        summary = {
            'total_clusters': n_clusters,
            'total_points': total_points,
            'noise_points': self.cluster_stats.get('noise', {}).get('size', 0),
            'silhouette_score': self.quality_metrics.get('silhouette_score', 0.0),
            'largest_clusters': largest_clusters,
            'avg_cluster_size': np.mean([v['size'] for k, v in self.cluster_stats.items() if k != 'noise']),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def visualize_clusters(
        self,
        embeddings_2d: np.ndarray,
        cluster_labels: np.ndarray,
        texts: Optional[List[str]] = None,
        save_plots: bool = True
    ) -> Dict[str, go.Figure]:
        """Develop comprehensive cluster visualizations."""
        logger.info("🎨 Developing cluster visualizations")
        
        plots = {}
        
        # 1. Main cluster scatter plot
        plots['cluster_scatter'] = self._plot_cluster_scatter(
            embeddings_2d, cluster_labels, texts
        )
        
        # 2. Cluster size distribution
        plots['cluster_sizes'] = self._plot_cluster_sizes()
        
        # 3. Silhouette analysis
        if self.quality_metrics.get('silhouette_samples'):
            plots['silhouette_analysis'] = self._plot_silhouette_analysis(cluster_labels)
        
        # 4. Quality metrics dashboard
        plots['quality_dashboard'] = self._plot_quality_dashboard()
        
        # Save plots if requested
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for name, fig in plots.items():
                # Save as HTML
                html_path = self.output_dir / f"{name}_{timestamp}.html"
                fig.write_html(html_path)
                
                # Save as PNG
                png_path = self.output_dir / f"{name}_{timestamp}.png"
                fig.write_image(png_path)
                
                logger.info(f"📊 Saved {name} to {html_path}")
        
        return plots
    
    def _plot_cluster_scatter(
        self,
        embeddings_2d: np.ndarray,
        cluster_labels: np.ndarray,
        texts: Optional[List[str]] = None
    ) -> go.Figure:
        """Develop interactive cluster scatter plot."""
        # Prepare data
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': cluster_labels,
            'text': texts[:len(embeddings_2d)] if texts else [f"Point {i}" for i in range(len(embeddings_2d))]
        })
        
        # Develop figure
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['text'],
            title="Cluster Visualization (UMAP 2D Projection)",
            labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _plot_cluster_sizes(self) -> go.Figure:
        """Plot cluster size distribution."""
        cluster_names = []
        sizes = []
        
        for name, stats in self.cluster_stats.items():
            cluster_names.append(name)
            sizes.append(stats['size'])
        
        fig = go.Figure(data=[
            go.Bar(x=cluster_names, y=sizes, text=sizes, textposition='auto')
        ])
        
        fig.update_layout(
            title="Cluster Size Distribution",
            xaxis_title="Cluster",
            yaxis_title="Number of Points",
            width=800,
            height=400
        )
        
        return fig
    
    def _plot_silhouette_analysis(self, cluster_labels: np.ndarray) -> go.Figure:
        """Develop silhouette analysis plot."""
        silhouette_samples_scores = self.quality_metrics.get('silhouette_samples', [])
        
        if not silhouette_samples_scores:
            return go.Figure()
        
        # Filter non-noise points
        non_noise_mask = cluster_labels != -1
        filtered_labels = cluster_labels[non_noise_mask]
        
        fig = go.Figure()
        
        # Plot silhouette scores for each cluster
        y_lower = 0
        unique_labels = sorted(np.unique(filtered_labels))
        
        for label in unique_labels:
            cluster_silhouette_scores = [
                score for i, score in enumerate(silhouette_samples_scores)
                if filtered_labels[i] == label
            ]
            
            cluster_silhouette_scores.sort()
            y_upper = y_lower + len(cluster_silhouette_scores)
            
            fig.add_trace(go.Scatter(
                x=cluster_silhouette_scores,
                y=list(range(y_lower, y_upper)),
                fill='tonexty' if label > unique_labels[0] else 'tozeroy',
                name=f'Cluster {label}',
                line=dict(width=0),
                mode='lines'
            ))
            
            y_lower = y_upper + 10
        
        # Add average silhouette score line
        avg_score = self.quality_metrics.get('silhouette_score', 0)
        fig.add_vline(
            x=avg_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {avg_score:.3f}"
        )
        
        fig.update_layout(
            title="Silhouette Analysis",
            xaxis_title="Silhouette Score",
            yaxis_title="Cluster Points",
            width=800,
            height=600
        )
        
        return fig
    
    def _plot_quality_dashboard(self) -> go.Figure:
        """Develop quality metrics dashboard."""
        metrics = self.quality_metrics
        
        # Develop subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Cluster Distribution",
                "Quality Metrics",
                "Noise Analysis",
                "Separation Analysis"
            ],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Cluster distribution pie chart
        cluster_names = [k for k in self.cluster_stats.keys() if k != 'noise']
        cluster_sizes = [self.cluster_stats[k]['size'] for k in cluster_names]
        
        fig.add_trace(
            go.Pie(labels=cluster_names, values=cluster_sizes, name="Clusters"),
            row=1, col=1
        )
        
        # 2. Quality metrics bar chart
        quality_names = ['Silhouette Score', 'Separation Ratio']
        quality_values = [
            metrics.get('silhouette_score', 0),
            metrics.get('separation_ratio', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=quality_names, y=quality_values, name="Quality"),
            row=1, col=2
        )
        
        # 3. Noise analysis
        total_points = metrics.get('total_points', 10000)  # Default fallback
        noise_points = metrics.get('n_noise_points', 0)
        valid_points = total_points - noise_points
        
        fig.add_trace(
            go.Bar(
                x=['Valid Points', 'Noise Points'],
                y=[valid_points, noise_points],
                name="Noise"
            ),
            row=2, col=1
        )
        
        # 4. Distance analysis
        fig.add_trace(
            go.Bar(
                x=['Intra-cluster', 'Inter-cluster'],
                y=[
                    metrics.get('avg_intra_cluster_distance', 0),
                    metrics.get('avg_inter_cluster_distance', 0)
                ],
                name="Distances"
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title="Clustering Quality Dashboard")
        return fig
    
    def save_analysis_report(self, results: Dict, filename: Optional[str] = None) -> Path:
        """
        Save comprehensive analysis report to JSON.
        
        Args:
            results: Analysis results from analyze_clusters()
            filename: Optional custom filename
            
        Returns:
            Path to saved report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cluster_analysis_report_{timestamp}.json"
        
        report_path = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._make_json_serializable(results)
        
        with open(report_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"📄 Analysis report saved to {report_path}")
        return report_path
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

def analyze_clustering_results(
    embeddings_2d: np.ndarray,
    cluster_labels: np.ndarray,
    texts: List[str],
    emotions: Optional[np.ndarray] = None,
    emotion_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Convenience function for complete cluster analysis.
    
    Args:
        embeddings_2d: 2D embeddings
        cluster_labels: Cluster assignments
        texts: Original texts
        emotions: Optional emotion data
        emotion_names: Optional emotion category names
        output_dir: Output directory for results
        
    Returns:
        Comprehensive analysis results
    """
    analyzer = ClusterAnalyzer(output_dir)
    
    results = analyzer.analyze_clusters(
        embeddings_2d=embeddings_2d,
        cluster_labels=cluster_labels,
        texts=texts,
        emotions=emotions,
        emotion_names=emotion_names
    )
    
    # Develop visualizations
    plots = analyzer.visualize_clusters(
        embeddings_2d=embeddings_2d,
        cluster_labels=cluster_labels,
        texts=texts,
        save_plots=True
    )
    
    results['plots'] = plots
    
    # Save report
    analyzer.save_analysis_report(results)
    
    return results

if __name__ == "__main__":
    # Example usage
    logger.info("🧪 Testing cluster analysis")
    
    # Prepare sample data
    np.random.seed(42)
    embeddings_2d = np.random.randn(500, 2)
    cluster_labels = np.random.randint(-1, 5, 500)  # 5 clusters + noise
    texts = [f"Sample text {i}" for i in range(500)]
    
    # Test analysis
    analyzer = ClusterAnalyzer()
    results = analyzer.analyze_clusters(embeddings_2d, cluster_labels, texts)
    
    print("✅ Cluster analysis test completed")
    print(f"   Found {results['summary']['total_clusters']} clusters")
    print(f"   Silhouette score: {results['quality_metrics']['silhouette_score']:.3f}")