import pandas as pd
import re
import os
import json
import emoji
import glob
import hashlib
from typing import Dict, List, Tuple, Any
from collections import Counter
import numpy as np
from tqdm import tqdm

class DataPreprocessor:
    """Class for preprocessing brand data by extracting metadata and cleaning text"""
    
    def __init__(self):
        """Initialize the data preprocessor"""
        # Compile regex patterns
        self.patterns = {
            'hashtags': re.compile(r'#\w+'),
            'mentions': re.compile(r'@\w+'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        }
    
    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract hashtags, mentions, and emojis"""
        if not text or pd.isna(text):
            return {'hashtags': '', 'mentions': '', 'emojis': ''}
            
        hashtags = [h.replace('#', '') for h in self.patterns['hashtags'].findall(str(text))]
        mentions = [m.replace('@', '') for m in self.patterns['mentions'].findall(str(text))]
        
        # Extract emojis using the emoji package
        emojis = emoji.distinct_emoji_list(str(text))
        return {
            'hashtags': '|'.join(hashtags),
            'mentions': '|'.join(mentions), 
            'emojis': ''.join(set(emojis))  # Remove duplicates
        }
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing hashtags, mentions, emojis, URLs"""
        if not text or pd.isna(text):
            return ""
            
        # Convert to string if not already
        text = str(text)
        
        # Remove hashtags, mentions, URLs
        cleaned = text
        cleaned = self.patterns['hashtags'].sub('', cleaned)
        cleaned = self.patterns['mentions'].sub('', cleaned)
        cleaned = self.patterns['urls'].sub('', cleaned)
        
        # Remove emojis using the emoji package
        cleaned = emoji.replace_emoji(cleaned, replace='')
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a dataframe to add cleaned_transcript, hashtags, mentions, emojis columns"""
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Ensure transcript column exists
        if 'transcript' not in processed_df.columns:
            print("Warning: 'transcript' column not found in dataframe")
            return processed_df
        
        # Extract metadata and clean text
        print(f"Preprocessing {len(processed_df)} rows...")
        
        # Initialize new columns
        processed_df['cleaned_transcript'] = ''
        processed_df['hashtags'] = ''
        processed_df['mentions'] = ''
        processed_df['emojis'] = ''
        
        # Process each row
        for idx, row in tqdm(processed_df.iterrows(), total=len(processed_df), desc="Preprocessing"):
            transcript = row.get('transcript', '')
            
            # Extract metadata
            metadata = self.extract_metadata(transcript)
            
            # Clean text
            cleaned_text = self.clean_text(transcript)
            
            # Update row
            processed_df.at[idx, 'cleaned_transcript'] = cleaned_text
            processed_df.at[idx, 'hashtags'] = metadata.get('hashtags', '')
            processed_df.at[idx, 'mentions'] = metadata.get('mentions', '')
            processed_df.at[idx, 'emojis'] = metadata.get('emojis', '')
        
        return processed_df
    
    def preprocess_file(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Process a single file and save the result"""
        # Load data
        try:
            df = pd.read_csv(input_file)
            print(f"Loaded {len(df)} rows from {input_file}")
        except Exception as e:
            print(f"Error loading file {input_file}: {e}")
            return None
        
        # Process dataframe
        processed_df = self.preprocess_dataframe(df)
        
        # Save result if output_file is provided
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to CSV
            processed_df.to_csv(output_file, index=False)
            print(f"Saved preprocessed data to {output_file}")
        
        return processed_df