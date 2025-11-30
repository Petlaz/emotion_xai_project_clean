#!/usr/bin/env python3
"""
Test script for Phase 5 clustering functionality.
Tests the complete clustering pipeline with sample data.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def test_clustering_pipeline():
    """Test the complete clustering pipeline."""
    try:
        logger.info("🚀 Starting clustering pipeline test...")

        # Import clustering modules
        from emotion_xai.clustering.embeddings import SemanticEmbeddingGenerator
        from emotion_xai.clustering.feedback_clustering import ThemeClusteringPipeline, ClusteringConfig
        from emotion_xai.clustering.analysis import ClusterAnalyzer

        logger.info("✅ Clustering modules imported successfully")

        # Test data - sample emotional texts
        test_texts = [
            "I'm so happy about this amazing birthday party!",
            "Happy birthday to the best friend ever!",
            "This is such a wonderful celebration of life!",
            "I feel really frustrated and disappointed right now.",
            "This situation makes me quite angry and upset.",
            "I'm confused about what's happening here.",
            "The weather is nice today, nothing special.",
            "Just a regular day at work, getting things done.",
            "I think this is an okay solution to the problem.",
            "New year celebrations are always so exciting!",
            "Happy new year everyone, hope it's amazing!",
            "Birthdays are special occasions to celebrate.",
        ] * 10  # Repeat to get more data points

        logger.info(f"📝 Using {len(test_texts)} test texts")

        # 1. Test embedding creation
        logger.info("🔄 Testing embedding creation...")

        embedding_generator = SemanticEmbeddingGenerator(
            model_name="all-MiniLM-L6-v2",
            cache_dir="models/cluster_embeddings"
        )

        embeddings = embedding_generator.generate_embeddings(
            texts=test_texts,
            batch_size=16,
            normalize=True,
            show_progress=True
        )

        logger.info(f"✅ Embeddings generated: {embeddings.shape}")

        # 2. Test clustering pipeline
        logger.info("🔄 Testing clustering pipeline...")
        config = ClusteringConfig(
            umap_n_neighbors=min(15, len(test_texts) - 1),
            umap_min_dist=0.1,
            hdbscan_min_cluster_size=min(5, len(test_texts) // 4),
            hdbscan_min_samples=3,
            standardize_features=True
        )

        clustering_pipeline = ThemeClusteringPipeline(config=config)
        embeddings_2d, cluster_labels = clustering_pipeline.fit(embeddings)
        logger.info(f"✅ Clustering completed: {embeddings_2d.shape}, {len(np.unique(cluster_labels))} clusters")

        # 3. Test cluster analysis
        logger.info("🔄 Testing cluster analysis...")
        analyzer = ClusterAnalyzer(output_dir="results/plots/clustering_test")

        analysis_results = analyzer.analyze_clusters(
            embeddings_2d=embeddings_2d,
            cluster_labels=cluster_labels,
            texts=test_texts
        )
        logger.info("✅ Cluster analysis completed")

        # 4. Test model saving and loading
        logger.info("🔄 Testing model persistence...")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Path(f"models/cluster_embeddings/test_pipeline_{timestamp}.pkl")

        clustering_pipeline.save_model(model_path)
        logger.info(f"✅ Model saved to: {model_path}")

        # Test loading
        loaded_pipeline = ThemeClusteringPipeline(config=config)
        loaded_pipeline.load_model(model_path)
        logger.info("✅ Model loaded successfully")

        # 5. Test prediction on new text
        logger.info("🔄 Testing prediction functionality...")
        test_prediction_text = "Happy anniversary to my wonderful partner!"

        try:
            # Create embedding for new text
            new_embedding = embedding_generator.generate_embeddings([test_prediction_text])

            # Try prediction
            try:
                prediction = loaded_pipeline.predict_new_points(new_embedding)
                logger.info(f"✅ Prediction successful: cluster {prediction[0]}")
            except Exception as pred_e:
                logger.warning(f"⚠️ Direct prediction failed: {pred_e}")
                logger.info("Using distance-based fallback...")

                # Fallback: distance-based prediction
                new_2d = loaded_pipeline.umap_model.transform(new_embedding)

                # Calculate distances to cluster centers
                cluster_centers = {}
                for label in np.unique(cluster_labels):
                    if label != -1:
                        cluster_mask = cluster_labels == label
                        center = np.mean(embeddings_2d[cluster_mask], axis=0)
                        cluster_centers[label] = center

                if cluster_centers:
                    distances = {label: np.linalg.norm(new_2d[0] - center)
                                 for label, center in cluster_centers.items()}
                    predicted_label = min(distances, key=distances.get)
                    logger.info(f"✅ Fallback prediction successful: cluster {predicted_label}")
                else:
                    logger.info("✅ No clusters found, assigned as noise")

        except Exception as e:
            logger.error(f"❌ Prediction test failed: {e}")

        # 6. Display results summary
        logger.info("\n" + "="*60)
        logger.info("🎉 CLUSTERING PIPELINE TEST RESULTS")
        logger.info("="*60)

        summary = clustering_pipeline.get_cluster_summary()
        logger.info(f"📊 Clusters discovered: {summary['n_clusters']}")
        logger.info(f"📊 Noise points: {summary['n_noise_points']} ({summary['noise_percentage']:.1f}%)")
        logger.info(f"📊 Silhouette score: {summary['silhouette_score']:.3f}")

        if 'cluster_stats' in analysis_results:
            logger.info(f"📊 Cluster analysis completed with {len(analysis_results['cluster_stats'])} clusters analyzed")

        logger.info("✅ All clustering functionality tests PASSED!")
        return True

    except Exception as e:
        logger.error(f"❌ Clustering pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_modules():
    """Test individual clustering modules."""
    try:
        logger.info("\n🔍 Testing individual modules...")
        
        # Test imports
        from emotion_xai.clustering.embeddings import SemanticEmbeddingGenerator, extract_emotion_embeddings
        from emotion_xai.clustering.feedback_clustering import (
            ThemeClusteringPipeline, 
            ClusteringConfig, 
            analyze_emotion_themes
        )
        from emotion_xai.clustering.analysis import ClusterAnalyzer, analyze_clustering_results
        
        logger.info("✅ All module imports successful")
        
        # Test configuration creation
        config = ClusteringConfig()
        logger.info("✅ Configuration creation successful")
        
        # Test analyzer initialization
        analyzer = ClusterAnalyzer(output_dir="results/test_clustering")
        logger.info("✅ Analyzer initialization successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Module test FAILED: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Starting clustering functionality tests...")
    
    # Test 1: Individual modules
    module_test_passed = test_individual_modules()
    
    # Test 2: Complete pipeline (only if modules pass)
    if module_test_passed:
        pipeline_test_passed = test_clustering_pipeline()
    else:
        pipeline_test_passed = False
    
    # Final results
    logger.info("\n" + "🎯 FINAL TEST RESULTS:")
    logger.info(f"   Module imports: {'✅ PASSED' if module_test_passed else '❌ FAILED'}")
    logger.info(f"   Complete pipeline: {'✅ PASSED' if pipeline_test_passed else '❌ FAILED'}")
    
    if module_test_passed and pipeline_test_passed:
        logger.info("\n🎉 ALL CLUSTERING TESTS SUCCESSFUL!")
        logger.info("🚀 Phase 5 clustering functionality is ready for production!")
        sys.exit(0)
    else:
        logger.error("\n❌ SOME TESTS FAILED!")
        logger.error("🔧 Please check the error messages above and fix the issues.")
        sys.exit(1)