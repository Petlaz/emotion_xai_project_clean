#!/bin/bash
# Production Transformer Training Runner
# Usage: ./run_production_training.sh [test|full]

set -e  # Exit on any error

echo "🚀 Emotion-XAI Transformer Training"
echo "==================================="

cd "$(dirname "$0")"

# Determine configuration
MODE=${1:-test}
case $MODE in
    "test")
        CONFIG="configs/test_training.json"
        echo "🧪 TEST mode: 5K samples, ~10 minutes"
        ;;
    "full"|"production")
        CONFIG="configs/production_training.json"
        echo "🏭 PRODUCTION mode: 147K samples, ~60 minutes"
        ;;
    *)
        echo "❌ Usage: $0 [test|full]"
        exit 1
        ;;
esac

# Check prerequisites
[ ! -d "data/processed" ] && { echo "❌ Run data processing first"; exit 1; }

mkdir -p results/production_training

echo "🔥 Starting training..."

# Run the training script
python scripts/train_transformer_production.py --config "$CONFIG"

echo ""
echo "✅ Training completed!"
echo "📊 Check results/ directory for detailed metrics"
echo "🤖 Check models/ directory for trained model"