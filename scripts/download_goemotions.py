#!/usr/bin/env python3
"""
GoEmotions Dataset Download Script

This script downloads the GoEmotions dataset and prepares it for use in the project.
GoEmotions is a dataset of 58k carefully curated Reddit comments labeled for 27 emotion categories.
"""

import os
import sys
from pathlib import Path
import requests
import pandas as pd
from typing import Optional

def download_goemotions(data_dir: Path = Path("data/raw")) -> bool:
    """
    Download GoEmotions dataset from the official repository.
    
    Args:
        data_dir: Directory to save the dataset files
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # GoEmotions dataset URLs (updated official repository)
    urls = {
        "full_1": "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
        "full_2": "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv", 
        "full_3": "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv"
    }
    
    print("🚀 Downloading GoEmotions dataset...")
    
    try:
        for part, url in urls.items():
            file_path = data_dir / f"goemotions_{part}.csv"
            
            if file_path.exists():
                print(f"✅ {part}.csv already exists, skipping...")
                continue
                
            print(f"📥 Downloading {part}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded {part} ({file_path.stat().st_size // 1024} KB)")
            
        # Combine parts into single file for convenience
        combine_splits(data_dir)
        
        print("🎉 GoEmotions dataset download completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def combine_splits(data_dir: Path) -> None:
    """Combine downloaded parts into single CSV file."""
    try:
        print("🔄 Combining dataset parts...")
        
        dfs = []
        for part in ["full_1", "full_2", "full_3"]:
            file_path = data_dir / f"goemotions_{part}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                print(f"  📁 Part {part}: {len(df)} samples")
                dfs.append(df)
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            output_path = data_dir / "goemotions.csv"
            combined_df.to_csv(output_path, index=False)
            
            print(f"✅ Combined dataset saved to {output_path}")
            print(f"📊 Total samples: {len(combined_df)}")
            print(f"📋 Columns: {list(combined_df.columns)}")
            
            # Create a simple processed version with text and emotions
            if 'text' in combined_df.columns:
                print("🔄 Creating processed version...")
                # Keep main columns for our project
                main_cols = ['text', 'id', 'author', 'subreddit']
                emotion_cols = [col for col in combined_df.columns if col not in 
                               ['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 
                                'rater_id', 'example_very_unclear']]
                
                processed_df = combined_df[main_cols + emotion_cols].copy()
                processed_path = data_dir / "goemotions_processed.csv"
                processed_df.to_csv(processed_path, index=False)
                print(f"✅ Processed dataset saved to {processed_path}")
            
    except Exception as e:
        print(f"⚠️ Warning: Could not combine parts: {e}")

def verify_dataset(data_dir: Path = Path("data/raw")) -> bool:
    """Verify the downloaded dataset integrity."""
    try:
        csv_path = data_dir / "goemotions.csv"
        if not csv_path.exists():
            print("❌ Combined dataset file not found")
            return False
            
        df = pd.read_csv(csv_path)
        
        print(f"🔍 Dataset verification:")
        print(f"  📊 Shape: {df.shape}")
        print(f"  📋 Columns: {list(df.columns)}")
        
        # Check for split column (may not exist in raw data)
        if 'split' in df.columns:
            print(f"  🏷️ Splits: {df['split'].value_counts().to_dict()}")
        
        # Check emotion columns
        emotion_cols = [col for col in df.columns if col in [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]]
        print(f"  🎭 Emotion labels found: {len(emotion_cols)}")
        
        # Basic data quality checks
        if len(df) < 50000:  # GoEmotions should have substantial samples
            print("⚠️ Warning: Dataset seems smaller than expected")
            
        if 'text' not in df.columns:
            print("❌ Error: 'text' column not found")
            return False
            
        print("✅ Dataset verification passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
        return False

def main():
    """Main execution function."""
    print("🎯 GoEmotions Dataset Setup")
    print("=" * 40)
    
    # Set up paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "raw"
    
    # Download dataset
    if download_goemotions(data_dir):
        # Verify dataset
        if verify_dataset(data_dir):
            print("\n🚀 Ready to start! Next steps:")
            print("1. Run: jupyter notebook")
            print("2. Open: notebooks/01_data_exploration.ipynb")
            print("3. Start exploring the GoEmotions dataset!")
        else:
            print("\n❌ Dataset verification failed. Please check the files.")
            sys.exit(1)
    else:
        print("\n❌ Dataset download failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()