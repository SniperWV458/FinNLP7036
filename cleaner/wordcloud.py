#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:03:50 2026

@author: ying
"""
import pandas as pd
import glob
import os
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# --- 1. Set Path ---
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
all_files = glob.glob(os.path.join(current_dir, "cleaned_*.csv"))

# --- 2. Group Files by Asset Name ---
# This will group: 'cleaned_DIA_12_18.csv' and 'cleaned_DIA_19_24.csv' into 'DIA'
asset_groups = {}
for f in all_files:
    file_name = os.path.basename(f)
    # Extract asset name (the part after 'cleaned_')
    # e.g., cleaned_DIA_Full... -> DIA
    asset_name = file_name.split('_')[1] 
    
    if asset_name not in asset_groups:
        asset_groups[asset_name] = []
    asset_groups[asset_name].append(f)

# --- 3. Stopwords Setup ---
my_stopwords = set(STOPWORDS)
my_stopwords.update(['http', 'https', 'com', 'co', 'amp', 'rt', 'now', 'today', 'stock', 'market', 'will', 'year', 'week'])

# --- 4. Process Each Asset Group ---
for asset_name, files in asset_groups.items():
    print(f"Merging and processing: {asset_name} ({len(files)} files)")
    
    # Combined data for this specific asset
    combined_data = []
    for f in files:
        temp_df = pd.read_csv(f)
        combined_data.append(temp_df)
    
    df_merged = pd.concat(combined_data, axis=0, ignore_index=True)
    
    # --- 5. Clean Text ---
    full_text = " ".join(df_merged['Text'].astype(str))
    # Remove $tags and [STOCKTWITS]
    full_text = re.sub(r'\$\w+', '', full_text)
    full_text = full_text.replace('[STOCKTWITS]', '')

    # --- 6. Generate WordCloud ---
    wc = WordCloud(
        width=1000, 
        height=500, 
        background_color='white',
        stopwords=my_stopwords,
        max_words=80,
        colormap='coolwarm'
    ).generate(full_text)

    # --- 7. Save Result ---
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{asset_name} Combined Sentiment (2012-2024)", fontsize=20)
    
    save_path = os.path.join(current_dir, f"Combined_Cloud_{asset_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Success: {save_path}")

print("\nAll assets merged and clouds generated!")