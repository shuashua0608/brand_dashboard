#!/usr/bin/env python3

import pandas as pd
import os
import glob
from datetime import datetime

def get_timestamp_range(file_path):
    """Get min and max timestamp from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        if 'timestamp' not in df.columns or len(df) == 0:
            return None, None, 0
        
        # Convert to datetime and get min/max
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        count = len(df)
        
        return min_time, max_time, count
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, 0

def analyze_folder_timestamps(base_dir):
    """Analyze timestamp ranges for all brands in all analysis folders"""
    
    folders = ['ccr_analysis', 'promo_analysis', 'sentiment_analysis']
    results = {}
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue
            
        print(f"\n=== {folder.upper()} ===")
        results[folder] = {}
        
        # Get all brand directories
        brand_dirs = [d for d in os.listdir(folder_path) 
                     if os.path.isdir(os.path.join(folder_path, d))]
        
        for brand in sorted(brand_dirs):
            brand_path = os.path.join(folder_path, brand)
            
            # Determine which file to analyze
            if folder in ['ccr_analysis', 'promo_analysis']:
                # Use *_all.csv files
                pattern = f"{brand_path}/{brand}_*_all.csv"
            else:  # sentiment_analysis
                # Use *_analysis.csv files
                pattern = f"{brand_path}/{brand}_*_analysis.csv"
            
            files = glob.glob(pattern)
            
            if files:
                file_path = files[0]  # Take the first matching file
                min_time, max_time, count = get_timestamp_range(file_path)
                
                if min_time and max_time:
                    results[folder][brand] = {
                        'min_time': min_time,
                        'max_time': max_time,
                        'count': count,
                        'file': os.path.basename(file_path)
                    }
                    
                    print(f"{brand:20} | {min_time.strftime('%Y-%m-%d %H:%M')} to {max_time.strftime('%Y-%m-%d %H:%M')} | {count:5} comments | {os.path.basename(file_path)}")
                else:
                    print(f"{brand:20} | No valid timestamps | {os.path.basename(file_path) if files else 'No file found'}")
            else:
                print(f"{brand:20} | No matching file found")
    
    return results

if __name__ == "__main__":
    base_dir = "processed_data_7_14"
    results = analyze_folder_timestamps(base_dir)
    
    # Summary across all folders
    print(f"\n=== OVERALL SUMMARY ===")
    all_min_times = []
    all_max_times = []
    total_comments = 0
    
    for folder, brands in results.items():
        for brand, data in brands.items():
            if data['min_time'] and data['max_time']:
                all_min_times.append(data['min_time'])
                all_max_times.append(data['max_time'])
                total_comments += data['count']
    
    if all_min_times and all_max_times:
        overall_min = min(all_min_times)
        overall_max = max(all_max_times)
        print(f"Overall time range: {overall_min.strftime('%Y-%m-%d %H:%M')} to {overall_max.strftime('%Y-%m-%d %H:%M')}")
        print(f"Total comments analyzed: {total_comments:,}")
        
        # Calculate duration
        duration = overall_max - overall_min
        print(f"Duration: {duration.days} days, {duration.seconds//3600} hours")