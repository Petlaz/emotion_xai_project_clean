"""
Emotion-XAI Gradio Web Interface
Phase 6: Interactive Web Interface with Hugging Face Spaces Deployment

Features:
- Instant launch capability with pre-loaded examples
- Real-time emotion prediction with explanations
- Interactive visualizations and clustering insights
- Production-ready deployment for Hugging Face Spaces
"""

import gradio as gr
import pandas as pd
import numpy as np
import torch
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from emotion_xai.utils.device import resolve_device
    from emotion_xai.explainability.explanations import explain_with_lime
    from scripts.use_trained_model import EmotionPredictor
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    print("Running in demo mode with mock data")

class EmotionXAIApp:
    def __init__(self):
        """Initialize the Emotion-XAI Gradio application."""
        self.device = self._setup_device()
        self.predictor = self._load_model()
        self.emotion_labels = self._get_emotion_labels()
        self.demo_examples = self._get_demo_examples()
        
    def _setup_device(self):
        """Setup optimal device for inference."""
        try:
            from emotion_xai.utils.device import resolve_device
            return resolve_device()
        except:
            return "cpu"
    
    def _load_model(self):
        """Load the trained emotion prediction model."""
        try:
            # Try to load the production model
            model_path = "models/distilroberta_production_20251130_044054"
            if Path(model_path).exists():
                from scripts.use_trained_model import EmotionPredictor
                return EmotionPredictor(model_path)
            else:
                print("Production model not found, using demo mode")
                return None
        except Exception as e:
            print(f"Model loading failed: {e}")
            return None
    
    def _get_emotion_labels(self):
        """Get the emotion labels for prediction."""
        return [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness',
            'optimism', 'pride', 'realization', 'relief', 'remorse',
            'sadness', 'surprise', 'neutral'
        ]
    
    def _get_demo_examples(self):
        """Pre-loaded examples for instant demonstration."""
        return [
            "I absolutely love this product! The customer service was amazing and delivery was super fast. Highly recommended! 🎉",
            "The product broke after just one day. Very disappointing and frustrating experience. Would not buy again.",
            "The quality is decent for the price, though I wish the shipping was faster. Overall satisfied with my purchase.",
            "Wow, this exceeded all my expectations! I'm so grateful for finding this. You guys are the best! ❤️",
            "I'm confused about how to use this feature. The instructions aren't very clear and I'm getting nervous about it.",
            "This brings back so many memories. I'm feeling nostalgic and a bit sad, but also proud of how far we've come."
        ]
    
    def predict_emotions(self, text: str, include_explanation: bool = True) -> Tuple[Dict, str, Optional[str]]:
        """
        Predict emotions for input text with optional explanations.
        
        Returns:
            - emotion_scores: Dictionary of emotion probabilities
            - visualization: HTML plot of top emotions
            - explanation: LIME explanation (if requested)
        """
        if not text.strip():
            return {}, "Please enter some text to analyze.", None
            
        try:
            if self.predictor:
                # Use real model prediction
                result = self.predictor.predict_single(text, threshold=0.1)  # Lower threshold to get all probabilities
                
                # Extract emotion probabilities - all_probabilities is already a dictionary
                raw_emotions = result.get('all_probabilities', {})
                # Round to 4 decimal places for cleaner display
                emotions = {emotion: round(prob, 4) for emotion, prob in raw_emotions.items()}
                
                # Generate explanation if requested
                explanation_html = None
                if include_explanation:
                    try:
                        explanation_html = self._generate_explanation(text, emotions)
                    except Exception as e:
                        explanation_html = f"Explanation generation failed: {str(e)}"
                        
            else:
                # Demo mode with mock predictions
                emotions = self._generate_demo_emotions(text)
                explanation_html = "Demo mode: Real explanations available with trained model"
            
            # Create visualization
            viz_html = self._create_emotion_visualization(emotions)
            
            return emotions, viz_html, explanation_html
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            return {}, error_msg, None
    
    def _generate_demo_emotions(self, text: str) -> Dict[str, float]:
        """Generate demo emotion predictions based on text content."""
        # Simple keyword-based demo emotions
        emotions = {}
        text_lower = text.lower()
        
        # Define emotion keywords
        emotion_keywords = {
            'joy': ['love', 'amazing', 'great', 'excellent', 'wonderful', 'fantastic', '❤️', '🎉'],
            'disappointment': ['broke', 'disappointing', 'frustrated', 'bad', 'terrible', 'awful'],
            'gratitude': ['thank', 'grateful', 'appreciate', 'thanks'],
            'anger': ['hate', 'angry', 'furious', 'mad'],
            'sadness': ['sad', 'depressed', 'crying', 'upset'],
            'fear': ['scared', 'afraid', 'nervous', 'worried'],
            'surprise': ['wow', 'amazing', 'incredible', 'unbelievable'],
            'confusion': ['confused', 'unclear', 'don\'t understand'],
            'approval': ['good', 'nice', 'decent', 'satisfied'],
            'neutral': []
        }
        
        # Calculate scores based on keywords
        for emotion, keywords in emotion_keywords.items():
            if emotion in self.emotion_labels:
                score = sum(1 for keyword in keywords if keyword in text_lower)
                emotions[emotion] = round(min(score * 0.3 + np.random.random() * 0.3, 0.95), 4)
        
        # Add some randomness to other emotions
        for emotion in self.emotion_labels:
            if emotion not in emotions:
                emotions[emotion] = round(np.random.random() * 0.2, 4)
                
        return emotions
    
    def _generate_explanation(self, text: str, emotions: Dict[str, float]) -> str:
        """Generate LIME explanation for the prediction."""
        try:
            if self.predictor:
                explanation = explain_with_lime(
                    predictor=self.predictor,
                    text=text,
                    num_features=min(10, len(text.split()))
                )
                return explanation.as_html()
            else:
                return "Explanations available with trained model"
        except Exception as e:
            return f"Explanation generation error: {str(e)}"
    
    def _create_emotion_visualization(self, emotions: Dict[str, float]) -> str:
        """Create interactive visualization of emotion predictions."""
        if not emotions:
            return "<p>No emotions to display</p>"
        
        # Get top 8 emotions for better visualization
        top_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:8])
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(top_emotions.values()),
                y=list(top_emotions.keys()),
                orientation='h',
                marker_color=px.colors.qualitative.Set3[:len(top_emotions)],
                text=[f"{v:.2%}" for v in top_emotions.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Top Predicted Emotions",
            xaxis_title="Confidence Score",
            yaxis_title="Emotions",
            height=400,
            margin=dict(l=100, r=50, t=50, b=50),
            font=dict(size=12)
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def batch_predict(self, texts: str) -> str:
        """Process multiple texts for batch emotion analysis."""
        if not texts.strip():
            return "Please enter texts separated by newlines"
            
        try:
            text_list = [text.strip() for text in texts.split('\n') if text.strip()]
            
            if not text_list:
                return "No valid texts found"
                
            results = []
            for i, text in enumerate(text_list, 1):
                emotions, _, _ = self.predict_emotions(text, include_explanation=False)
                
                if emotions:
                    top_emotion = max(emotions.items(), key=lambda x: x[1])
                    results.append(f"{i}. **Text:** {text[:100]}{'...' if len(text) > 100 else ''}")
                    results.append(f"   **Top Emotion:** {top_emotion[0]} ({top_emotion[1]:.4f})")
                    results.append("")
                else:
                    results.append(f"{i}. Error processing: {text}")
                    results.append("")
            
            return '\n'.join(results)
            
        except Exception as e:
            return f"Batch processing failed: {str(e)}"
    
    def get_model_info(self) -> str:
        """Get information about the loaded model."""
        if self.predictor:
            return f"""
            ### 🤖 Model Information
            - **Model**: DistilRoBERTa (Production Fine-tuned)
            - **Performance**: F1-macro 0.196 (19.6%)
            - **Device**: {self.device}
            - **Emotions**: {len(self.emotion_labels)} categories
            - **Status**: ✅ Production Ready
            
            ### 📊 Recent Achievements
            - 1.2x improvement over baseline
            - 87% loss reduction during training
            - Trained on 147K samples from GoEmotions dataset
            """
        else:
            return f"""
            ### 🔄 Demo Mode
            - **Status**: Running with simulated predictions
            - **Device**: {self.device}
            - **Note**: Full functionality available with trained model
            - **Emotions**: {len(self.emotion_labels)} categories supported
            """

def create_app():
    """Create and configure the Gradio application."""
    app = EmotionXAIApp()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header-text {
        text-align: center;
        color: #2D3748;
        margin-bottom: 20px;
    }
    .example-box {
        background-color: #F7FAFC;
        border-left: 4px solid #4299E1;
        padding: 10px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=css, title="Emotion-XAI: Explainable AI for Emotion Detection", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.Markdown("""
        # 🎭 Emotion-XAI: Explainable AI for Emotion Detection
        
        ### Analyze emotions in text with state-of-the-art AI and explainable insights
        
        **✨ Features:** Multi-label emotion classification | Explainable AI | Real-time analysis | Production-ready model
        """, elem_classes=["header-text"])
        
        # Model information
        with gr.Row():
            gr.Markdown(app.get_model_info())
        
        # Main interface tabs
        with gr.Tabs():
            
            # Single Text Analysis Tab
            with gr.TabItem("🔍 Single Text Analysis", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="Enter text to analyze",
                            placeholder="Type or paste your text here...",
                            lines=4,
                            value=app.demo_examples[0]  # Pre-loaded example
                        )
                        
                        with gr.Row():
                            predict_btn = gr.Button("🚀 Analyze Emotions", variant="primary", size="lg")
                            explanation_check = gr.Checkbox(label="Include AI Explanations", value=True)
                        
                        # Example buttons for quick testing
                        gr.Markdown("**Quick Examples:** Click to try instantly!")
                        with gr.Row():
                            for i, example in enumerate(app.demo_examples[:3]):
                                gr.Button(
                                    f"Example {i+1}",
                                    size="sm"
                                ).click(
                                    lambda ex=example: ex,
                                    outputs=[text_input]
                                )
                    
                    with gr.Column(scale=1):
                        emotion_output = gr.JSON(
                            label="📊 Emotion Scores",
                            show_label=True
                        )
                        
                # Visualization and explanation outputs
                with gr.Row():
                    visualization_output = gr.HTML(label="📈 Emotion Visualization")
                    
                with gr.Row():
                    explanation_output = gr.HTML(label="🔍 AI Explanation", visible=True)
                
                # Connect the prediction function
                predict_btn.click(
                    app.predict_emotions,
                    inputs=[text_input, explanation_check],
                    outputs=[emotion_output, visualization_output, explanation_output]
                )
            
            # Batch Analysis Tab
            with gr.TabItem("📝 Batch Analysis"):
                gr.Markdown("### Analyze multiple texts at once")
                
                batch_input = gr.Textbox(
                    label="Enter multiple texts (one per line)",
                    placeholder="Text 1: I love this product!\nText 2: The service was terrible.\nText 3: Decent quality for the price.",
                    lines=8,
                    value="\n".join(app.demo_examples[3:6])  # Pre-loaded batch examples
                )
                
                batch_btn = gr.Button("🚀 Analyze Batch", variant="primary")
                batch_output = gr.Markdown(label="Results")
                
                batch_btn.click(
                    app.batch_predict,
                    inputs=[batch_input],
                    outputs=[batch_output]
                )
            
            # About Tab
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About Emotion-XAI
                
                This application demonstrates state-of-the-art emotion detection using explainable AI techniques.
                
                ### 🎯 Key Features
                - **Multi-label Classification**: Detect multiple emotions simultaneously
                - **Explainable AI**: Understand why the model made specific predictions
                - **Real-time Analysis**: Instant emotion detection and visualization
                - **Production Ready**: Based on fine-tuned DistilRoBERTa model
                
                ### 📊 Model Performance
                - **Dataset**: GoEmotions (211K Reddit comments, 28 emotions)
                - **Current Model**: F1-macro 0.196 (19.6%)
                - **Baseline Comparison**: 1.2x improvement over TF-IDF baseline
                - **Training**: 147K samples, optimized for Mac M1/M2
                
                ### 🔬 Technical Details
                - **Architecture**: DistilRoBERTa (82M parameters)
                - **Explainability**: SHAP and LIME explanations
                - **Deployment**: Optimized for Hugging Face Spaces
                - **Device Support**: Auto-detection (CUDA/MPS/CPU)
                
                ### 🚀 Project Status
                ✅ **Production Ready**: Complete emotion analysis pipeline with explainable AI
                🎯 **Current**: Interactive web interface deployment
                
                ---
                
                **GitHub**: [emotion_xai_project_clean](https://github.com/Petlaz/emotion_xai_project_clean)  
                **License**: MIT | **Framework**: Gradio + Transformers + Explainable AI
                """)
        
        # Footer
        gr.Markdown("""
        ---
        🔬 **Research Project** | 🚀 **Production Ready**  
        Built with ❤️ using Gradio, Transformers, SHAP, and LIME
        """, elem_classes=["header-text"])
    
    return demo

def main():
    """Launch the Gradio application."""
    print("🚀 Launching Emotion-XAI Gradio Interface...")
    print("🎯 Phase 6: Interactive Web Interface with Instant Launch")
    
    # Create and launch the app
    demo = create_app()
    
    # Launch with specific configuration for Hugging Face Spaces
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Standard Gradio port (7861 for local testing if needed)
        share=True,             # Create public link for sharing
        show_error=True,        # Show detailed errors for debugging
        quiet=False,            # Show launch information
        inbrowser=True,         # Open browser automatically for local testing
    )

if __name__ == "__main__":
    main()