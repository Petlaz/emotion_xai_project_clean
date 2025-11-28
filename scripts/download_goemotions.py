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
    
    # GoEmotions dataset URLs (official repository)
    urls = {
        "train": "https://storage.googleapis.com/gresearch/goemotions/data/train.tsv",
        "dev": "https://storage.googleapis.com/gresearch/goemotions/data/dev.tsv", 
        "test": "https://storage.googleapis.com/gresearch/goemotions/data/test.tsv"
    }
    
    print("🚀 Downloading GoEmotions dataset...")
    
    try:
        for split, url in urls.items():
            file_path = data_dir / f"goemotions_{split}.tsv"
            
            if file_path.exists():
                print(f"✅ {split}.tsv already exists, skipping...")
                continue
                
            print(f"📥 Downloading {split} split...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded {split} split ({file_path.stat().st_size // 1024} KB)")
            
        # Combine splits into single file for convenience
        combine_splits(data_dir)
        
        print("🎉 GoEmotions dataset download completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def combine_splits(data_dir: Path) -> None:
    """Combine train/dev/test splits into single CSV file."""
    try:
        print("🔄 Combining dataset splits...")
        
        dfs = []
        for split in ["train", "dev", "test"]:
            file_path = data_dir / f"goemotions_{split}.tsv"
            if file_path.exists():
                df = pd.read_csv(file_path, sep='\t')
                df['split'] = split
                dfs.append(df)
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            output_path = data_dir / "goemotions.csv"
            combined_df.to_csv(output_path, index=False)
            
            print(f"✅ Combined dataset saved to {output_path}")
            print(f"📊 Total samples: {len(combined_df)}")
            print(f"📋 Columns: {list(combined_df.columns)}")
            
    except Exception as e:
        print(f"⚠️ Warning: Could not combine splits: {e}")

def verify_dataset(data_dir: Path = Path("data/raw")) -> bool:
    """Verify the downloaded dataset integrity."""
    try:
        csv_path = data_dir / "goemotions.csv"
        if not csv_path.exists():
            print("❌ Combined dataset file not found")
            return False
            
        df = pd.read_csv(csv_path)
        
        print("🔍 Dataset verification:")
        print(f"  📊 Shape: {df.shape}")
        print(f"  📋 Columns: {list(df.columns)}")
        print(f"  🏷️ Splits: {df['split'].value_counts().to_dict()}")
        
        # Basic data quality checks
        if len(df) < 50000:  # GoEmotions should have ~58k samples
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