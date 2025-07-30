import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
import hashlib
import json
import time
import sys
import os
sys.path.append('..')
from bedrock_api_call import BedrockClaudeDistiller
from load_brand_info import load_brand_info

class PromoDetectionAgent:
    def __init__(self, model_id, region, similarity_threshold=0.85):
        """Initialize the promotional content detection agent"""
        self.distiller = BedrockClaudeDistiller(model_id=model_id, region=region)
        self.similarity_threshold = similarity_threshold
        self.cluster_stats = {
            'total_clusters': 0,
            'comments_in_clusters': 0,
            'llm_calls_saved': 0
        }
    
    def extract_content_hash(self, text: str) -> str:
        """Create hash for duplicate detection with NaN handling"""
        if pd.isna(text) or text is None:
            return hashlib.md5(b'empty_content').hexdigest()
        
        try:
            cleaned = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            return hashlib.md5(cleaned.encode()).hexdigest()
        except Exception:
            return hashlib.md5(b'error_content').hexdigest()
    
    def detect_affiliate_links(self, text: str) -> bool:
        """Detect common affiliate link patterns in text"""
        if pd.isna(text) or text is None:
            return False

        # Check for common URL shorteners used for affiliate links
        if re.search(r'(amzn\.to|bit\.ly|tinyurl\.com|rstyle\.me|liketoknow\.it)/\w+', text, re.IGNORECASE):
            return True
            
        # Check for explicit promotional hashtags
        if re.search(r'#(ad|sponsored|affiliate|partner)\b', text, re.IGNORECASE):
            return True
            
        # Check for common promotional phrases
        if re.search(r'(affiliate link|use my code|use code|my link in bio)', text, re.IGNORECASE):
            return True
                
        return False
    

    def analyze_comment(self, comment_data, brand_name: str, brand_filename: str = 'brand_info_new.json') -> Dict[str, any]:
        """Analyze a comment for promotional content with direct promotional terms in prompt"""
        transcript = comment_data.get('transcript', '')
        hashtags = comment_data.get('hashtags', '')
        mentions = comment_data.get('mentions', '')
        platform = comment_data.get('social_media_channel', 'Unknown')
        
        # Get brand info
        brand_info = load_brand_info(brand_name, brand_filename)
        brand_category = brand_info.get('category', '') if brand_info else ''
        brand_description = brand_info.get('description', '') if brand_info else ''
        
        prompt =f"""Analyze this comment from {platform} to determine if it is promotional content related to {brand_name}:

Brand: {brand_name}
Category: {brand_category}
Description: {brand_description}
Comment: {transcript}

=== PROMOTIONAL INDICATORS ===
Must have AT LEAST ONE to classify as promotional:
- AFFILIATE LINKS: amzn.to, bit.ly, rstyle.me, liketoknow.it, linktr.ee, "link in bio"
- DISCLOSURE TAGS: #ad, #sponsored, #partnership, #gifted, #pr, "paid partnership"
- DISCOUNT CODES: "use my code", "20% off with", promo codes, exclusive offers
- BRAND VOICE: Speaking as the brand ("we", "our product", "our team")
- SALES CALLS: "buy now", "shop here", "get yours", "available at"
- CAMPAIGNS: Giveaways, contests, brand hashtag campaigns

=== NON-PROMOTIONAL INDICATORS ===
These suggest genuine user content:
- Personal experience with specific details
- Balanced reviews with pros/cons
- Answering user questions helpfully
- Casual product mentions in conversation
- Price discussions without sales intent
- Complaints or negative experiences

=== CLASSIFICATION TYPES ===
- Brand_Official: Official brand account or representative
- Influencer: Content creator with commercial intent
- Affiliate: Contains affiliate links/marketing
- UGC_Campaign: Brand campaign participation
- Spam: Unsolicited excessive promotion

Respond with:
IS_PROMOTIONAL: Yes/No (Default to No unless definitive indicators present)
PROMO_TYPE: [Brand_Official, Influencer, Affiliate, UGC_Campaign, Spam] (only if IS_PROMOTIONAL is Yes)
HAS_AFFILIATE_LINKS: Yes/No
PROMOTIONAL_INTENSITY: [High, Medium, Low] (only if IS_PROMOTIONAL is Yes)
REASONING: Brief explanation of key indicators or why it's genuine user content
KEYWORDS: Promotional phrases, hashtags, or terms identified
"""
        try:
            response = self.distiller.call_claude_api(prompt)
            text_response = response['content'][0]['text']
            
            # Parse response
            is_promo = False
            promo_match = re.search(r'IS_PROMOTIONAL:\s*(Yes|No)', text_response, re.IGNORECASE)
            if promo_match and promo_match.group(1).lower() == 'yes':
                is_promo = True

            # Get promotional intensity (replaces confidence)
            promotional_intensity = "Medium"  # Default
            intensity_match = re.search(r'PROMOTIONAL_INTENSITY:\s*(High|Medium|Low)', text_response, re.IGNORECASE)
            if intensity_match:
                promotional_intensity = intensity_match.group(1)
            
            # Get the LLM's assessment of affiliate links
            has_affiliate_links = False
            affiliate_match = re.search(r'HAS_AFFILIATE_LINKS:\s*(Yes|No)', text_response, re.IGNORECASE)
            if affiliate_match and affiliate_match.group(1).lower() == 'yes':
                has_affiliate_links = True
    
            # Double-check for affiliate links with our function as a backup
            if not has_affiliate_links and self.detect_affiliate_links(transcript):
                has_affiliate_links = True
                # If we found affiliate links but LLM didn't, mark as promotional and high intensity
                if not is_promo:
                    is_promo = True
                    promotional_intensity = "High"

            # Handle promotional type
            promo_type = None
            if is_promo:
                type_match = re.search(r'PROMO_TYPE:\s*(Brand_Official|Influencer|Affiliate|UGC_Campaign|Spam)', text_response, re.IGNORECASE)
                if type_match:
                    promo_type = type_match.group(1)
                
                # If we have affiliate links but no type detected, set as Affiliate
                if has_affiliate_links and not promo_type:
                    promo_type = "Affiliate"
                
                # Default fallback if promotional but no type detected
                if not promo_type:
                    promo_type = "Influencer"  # Most common fallback

                # Extract reasoning
            reasoning = ""
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=KEYWORDS:|RESULT_PROPAGATED:|\$)', text_response, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()[:200]

            # Extract keywords
            keywords = []
            keywords_match = re.search(r'KEYWORDS:\s*(.+?)(?=RESULT_PROPAGATED:|\$)', text_response, re.DOTALL)
            if keywords_match:
                keywords_text = keywords_match.group(1).strip()
                if keywords_text and keywords_text.lower() not in ['none', 'n/a', '']:
                    # Split by common delimiters and clean up
                    keywords = [k.strip() for k in re.split(r'[,;|\n]', keywords_text) if k.strip()][:5]
                
            # Fallback keyword extraction from reasoning if no keywords found
            if not keywords and reasoning:
                # Extract hashtags and URLs from reasoning
                url_matches = re.findall(r'https?://\S+|#\w+|@\w+', reasoning)
                if url_matches:
                    keywords = url_matches[:5]
        
            return {
            'is_promotional': is_promo,
            'promo_type': promo_type,
            'promotional_intensity': promotional_intensity,  # Changed from 'confidence'
            'has_affiliate_links': has_affiliate_links,
            'reasoning': reasoning,
            'keywords': keywords
        }
        except Exception as e:
            return {
                'is_promotional': False,
                'promo_type': 'None',
                'promotional_intensity': 'Low',  # Changed from 'confidence'
                'has_affiliate_links': False,
                'reasoning': f"Error: {str(e)[:100]}",
                'keywords': []
            }
    
    def find_duplicate_clusters(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Find clusters of similar/duplicate promotional content"""
        clusters = {}
        
        # Exact duplicates by hash
        hash_groups = df.groupby('content_hash')['fid'].apply(list).to_dict()
        exact_clusters = {f"exact_{k}": v for k, v in hash_groups.items() if len(v) > 1}
        
        # Similar content clusters using cleaned text
        if len(df) > 1:
            try:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(df['cleaned_transcript'].fillna(''))
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                processed = set()
                for i in range(len(df)):
                    if i in processed:
                        continue
                    
                    similar_indices = np.where(similarity_matrix[i] >= self.similarity_threshold)[0]
                    if len(similar_indices) > 1:
                        cluster_fids = df.iloc[similar_indices]['fid'].tolist()
                        clusters[f'similar_{i}'] = cluster_fids
                        processed.update(similar_indices)
            except:
                pass
        
        return {**exact_clusters, **clusters}
    
    
    def process_comments(self, df, brand_name, brand_filename='brand_info_new.json', max_workers=8):
        """Process comments with clustering optimization
        
        Uses clustering metadata from prefilter agent to avoid redundant LLM calls
        """
        print(f"Processing {len(df)} comments for promotional content detection...")
        
        # Add content hash for duplicate detection
        df['content_hash'] = df['cleaned_transcript'].apply(self.extract_content_hash)
        
        # Verify clustering metadata is available
        if not all(col in df.columns for col in ['cluster_id', 'is_representative', 'repetition_count']):
            print("ERROR: Clustering metadata not found. This agent requires clustered data.")
            print("Please run prefilter_agent.py first to generate clustering metadata.")
            return pd.DataFrame()
        
        # Reset cluster stats
        self.cluster_stats = {
            'total_clusters': 0,
            'comments_in_clusters': 0,
            'llm_calls_saved': 0
        }
        
        # Calculate clustering statistics
        clustered_comments = df[df['cluster_id'].notna()].shape[0]
        unique_clusters = df['cluster_id'].nunique() - (1 if df['cluster_id'].isna().any() else 0)
        self.cluster_stats['total_clusters'] = unique_clusters
        self.cluster_stats['comments_in_clusters'] = clustered_comments
        
        # Select only representative comments for LLM processing
        # These are either cluster representatives or comments that don't belong to any cluster
        representative_df = df[df['is_representative'] | df['cluster_id'].isna()]
        
        print(f"Found {clustered_comments} comments in {unique_clusters} clusters")
        print(f"Processing {len(representative_df)} representative comments with LLM...")
        
        # Process representative comments with LLM
        representative_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit LLM analysis tasks for representative comments
            future_to_row = {
                executor.submit(self.analyze_comment, 
                    {'transcript': row.get('transcript', ''), 
                     'cleaned_transcript': row.get('cleaned_transcript', ''),
                     'hashtags': row.get('hashtags', ''),
                     'mentions': row.get('mentions', ''),
                     'social_media_channel': row.get('social_media_channel', 'Unknown')}, 
                    brand_name,
                    brand_filename): row 
                for _, row in representative_df.iterrows()
            }
            
            # Collect results as they complete with progress bar
            for future in tqdm(as_completed(future_to_row), total=len(representative_df), desc=f"Promo Analysis ({max_workers} workers)"):
                original_row = future_to_row[future]
                try:
                    api_result = future.result()
                    
                    # Combine with original row data
                    final_result = {
                        'fid': original_row.get('fid', ''),
                        'transcript': original_row.get('transcript', ''),
                        'cleaned_transcript': original_row.get('cleaned_transcript', ''),
                        'timestamp': original_row.get('timestamp', ''),
                        'text_language': original_row.get('text_language', ''),
                        'social_media_channel': original_row.get('social_media_channel', ''),
                        'hashtags': original_row.get('hashtags', ''),
                        'mentions': original_row.get('mentions', ''),
                        'emojis': original_row.get('emojis', ''),
                        'content_hash': original_row.get('content_hash', ''),
                        'cluster_id': original_row.get('cluster_id'),
                        'is_representative': original_row.get('is_representative', False),
                        'repetition_count': original_row.get('repetition_count', 1),
                        **api_result,
                        'result_propagated': False
                    }
                    representative_results.append(final_result)
                    
                except Exception as e:
                    # Handle individual failures
                    error_result = {
                        'fid': original_row.get('fid', ''),
                        'transcript': original_row.get('transcript', ''),
                        'cleaned_transcript': original_row.get('cleaned_transcript', ''),
                        'timestamp': original_row.get('timestamp', ''),
                        'text_language': original_row.get('text_language', ''),
                        'social_media_channel': original_row.get('social_media_channel', ''),
                        'hashtags': original_row.get('hashtags', ''),
                        'mentions': original_row.get('mentions', ''),
                        'emojis': original_row.get('emojis', ''),
                        'content_hash': original_row.get('content_hash', ''),
                        'cluster_id': original_row.get('cluster_id'),
                        'is_representative': original_row.get('is_representative', False),
                        'repetition_count': original_row.get('repetition_count', 1),
                        'is_promotional': False,
                        'promo_type': 'Error',
                        'promotional_intensity': 'Low',
                        'reasoning': f"Error: {str(e)[:50]}",
                        'keywords': [],
                        'result_propagated': False
                    }
                    representative_results.append(error_result)
        
        # Create a DataFrame with the LLM results for representative comments
        representative_results_df = pd.DataFrame(representative_results)
        
        # Create a mapping from cluster_id to LLM results
        cluster_results = {}
        for _, row in representative_results_df[representative_results_df['cluster_id'].notna()].iterrows():
            cluster_id = row['cluster_id']
            # Extract only the columns that were added by the analyze_comment method
            llm_cols = ['is_promotional', 'promo_type', 'promotional_intensity', 'has_affiliate_links', 'reasoning', 'keywords']
            cluster_results[cluster_id] = {col: row[col] for col in llm_cols}
        
        # Apply the results to all comments
        final_results = []
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            cluster_id = row_dict.get('cluster_id')
            
            if pd.notna(cluster_id) and not row_dict.get('is_representative', False):
                # This is a non-representative comment in a cluster
                # Use the results from the cluster's representative
                cluster_result = cluster_results.get(cluster_id, {
                    'is_promotional': False,
                    'promo_type': 'None',
                    'promotional_intensity': 'Low',
                    'has_affiliate_links': False,
                    'reasoning': 'No representative result found',
                    'keywords': []
                })
                
                final_results.append({
                    **row_dict,
                    **cluster_result,
                    'result_propagated': True
                })
                
                # Count as a saved LLM call
                self.cluster_stats['llm_calls_saved'] += 1
            else:
                # This is either a representative comment or a comment not in any cluster
                # Find the corresponding row in representative_results_df
                matching_rows = representative_results_df[representative_results_df['fid'] == row_dict.get('fid', '')]
                if not matching_rows.empty:
                    final_results.append(matching_rows.iloc[0].to_dict())
                else:
                    # This shouldn't happen, but just in case
                    final_results.append({
                        **row_dict,
                        'is_promotional': False,
                        'promo_type': 'None',
                        'promotional_intensity': 'Low',
                        'has_affiliate_links': False,
                        'reasoning': 'Result not found',
                        'keywords': [],
                        'result_propagated': False
                    })
        
        # Create the final dataframe
        final_df = pd.DataFrame(final_results)
        
        # Print clustering efficiency summary
        print(f"Clustering Efficiency: {self.cluster_stats['llm_calls_saved']} LLM calls saved")
        print(f"Processed {len(final_df)} comments with {len(representative_df)} LLM calls")
        
        return final_df
    
    # _process_all_comments method removed as we always use clustered data
    
def process_single_brand(file_path=None, model_id=None, region=None, brand_name=None, brand_filename='brand_info_new.json', max_workers=8, sample=None,  output_dir="processed_data"):
    """Process a single brand for promotional content detection"""
    import os
    import time
    
    # If brand name is provided but no file path, try to get file path from brand_info.json
    if brand_name and not file_path:
        brand_info = load_brand_info(brand_name, brand_filename)
        if brand_info and 'file_path' in brand_info:
            file_path = brand_info['file_path']
            print(f"Using file path from brand_info.json: {file_path}")
    
    # Extract brand name from file path if not provided
    if not brand_name and file_path:
        # Try to extract from directory name first
        brand_name = os.path.basename(os.path.dirname(file_path))
        if not brand_name or brand_name == '':
            # If that fails, try from filename
            filename = os.path.basename(file_path)
            if '_' in filename:
                brand_name = filename.split('_')[0]
            else:
                brand_name = filename.replace('.csv', '')
    
    # Ensure we have both brand name and file path
    if not brand_name or not file_path:
        print("ERROR: Both brand name and file path are required")
        return 0, 0
    
    # Ensure brand_name is a string before calling upper()
    brand_display = brand_name.upper() if isinstance(brand_name, str) else str(brand_name).upper()
    print(f"\n=== PROMOTIONAL DETECTION: {brand_display} ===")
    
    # Verify file exists and is readable
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return 0, 0
    
    try:
        # Load relevant comments
        df = pd.read_csv(file_path)
        if len(df) == 0:
            print(f"ERROR: File is empty: {file_path}")
            return 0, 0
    except Exception as e:
        print(f"ERROR: Cannot read file {file_path}: {e}")
        return 0, 0
    
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
        print(f"Analyzing {len(df)} sample comments...")
    else:
        print(f"Analyzing {len(df)} relevant comments...")
    
    # Initialize agent
    agent = PromoDetectionAgent(model_id=model_id, region=region)
    
    # Process comments with true parallel processing
    print(f"Concurrent promotional analysis with {max_workers} workers...")
    start_time = time.time()
    results_df = agent.process_comments(df, brand_name, brand_filename, max_workers)
    processing_time = time.time() - start_time
    
    # Filter promotional comments
    promo_comments = results_df[results_df['is_promotional'] == True]
    
    # Find duplicate clusters
    clusters = agent.find_duplicate_clusters(results_df)
    
    # Calculate high intensity promotional content
    high_intensity_promo = results_df[(results_df['is_promotional'] == True) & (results_df['promotional_intensity'] == 'High')]
    
    # Display results
    print(f"Processed: {len(results_df)} | Promotional: {len(promo_comments)} | High Intensity: {len(high_intensity_promo)}")
    print(f"Promo Rate: {len(promo_comments)/len(results_df)*100:.1f}% | High Intensity Rate: {len(high_intensity_promo)/len(results_df)*100:.1f}%")
    print(f"Processing Time: {processing_time:.1f}s")
    print(f"Throughput: {len(results_df)/processing_time:.1f} comments/second")
    
    # Display clustering efficiency
    propagated_count = results_df['result_propagated'].sum() if 'result_propagated' in results_df.columns else 0
    if propagated_count > 0:
        efficiency_percentage = (propagated_count / len(results_df)) * 100
        print(f"Clustering Efficiency: {propagated_count} LLM calls saved ({efficiency_percentage:.1f}%)")
    
    # Save results in brand-specific folder
    promo_dir = f"{output_dir}/promo_analysis"
    brand_dir = f"{promo_dir}/{brand_name}"
    os.makedirs(brand_dir, exist_ok=True)
    
    all_file = f"{brand_dir}/{brand_name}_promo_all.csv"
    promo_file = f"{brand_dir}/{brand_name}_promo_detected.csv"
    summary_file = f"{brand_dir}/{brand_name}_promo_summary.json"
    
    results_df.to_csv(all_file, index=False)
    promo_comments.to_csv(promo_file, index=False)
    
    # Platform distribution
    platform_distribution = {}
    if 'social_media_channel' in promo_comments.columns:
        platform_distribution = promo_comments['social_media_channel'].value_counts().to_dict()
    
    # Promo type distribution
    promo_types = {}
    if 'promo_type' in promo_comments.columns:
        # Only count actual promo types (not None)
        promo_types = promo_comments[promo_comments['promo_type'] != 'None']['promo_type'].value_counts().to_dict()
    
    # Calculate clustering efficiency
    propagated_count = results_df['result_propagated'].sum() if 'result_propagated' in results_df.columns else 0
    efficiency_percentage = (propagated_count / len(results_df)) * 100 if len(results_df) > 0 else 0
    
    # Verify that we have the same number of comments as the input
    original_count = len(df)
    processed_count = len(results_df)
    if original_count != processed_count:
        print(f"WARNING: Comment count mismatch! Original: {original_count}, Processed: {processed_count}")
        print("This suggests that some comments may have been lost during processing.")
    else:
        print(f"✓ All {original_count} comments preserved in the output")
    
    # Calculate high intensity promotional content
    high_intensity_promo_final = results_df[(results_df['is_promotional'] == True) & (results_df['promotional_intensity'] == 'High')]
    
    # Save summary in requested format
    summary = {
        'brand': brand_name,
        'total_processed': len(results_df),
        'promotional_count': len(promo_comments),
        'high_intensity_promo_count': len(high_intensity_promo),
        'promotional_rate': len(promo_comments)/len(results_df) if len(results_df) > 0 else 0,
        'high_intensity_promo_rate': len(high_intensity_promo)/len(results_df) if len(results_df) > 0 else 0,
        'duplicate_clusters': len(clusters),
        'total_duplicates': sum(len(fids) for fids in clusters.values()),
        'repeated_content_rate': sum(len(fids) for fids in clusters.values()) / len(results_df) * 100 if len(results_df) > 0 else 0,
        'processing_time': processing_time,
        'throughput': len(results_df)/processing_time if processing_time > 0 else 0,
        'platform_distribution': platform_distribution,
        'promo_types': promo_types,
        'promotional_intensity_distribution': results_df[results_df['is_promotional'] == True]['promotional_intensity'].value_counts().to_dict() if 'promotional_intensity' in results_df.columns else {},
        'processing_method': 'clustered',
        'risk_indicators': {
            'high_promo_rate': len(high_intensity_promo) / len(results_df) > 0.2 if len(results_df) > 0 else False,
            'high_duplicate_rate': sum(len(fids) for fids in clusters.values()) / len(results_df) > 0.2 if len(results_df) > 0 else False,
            'spam_detected': 'Spam' in promo_comments['promo_type'].values if len(promo_comments) > 0 else False
        },
        'clustering_efficiency': {
            'total_clusters': agent.cluster_stats.get('total_clusters', 0),
            'comments_in_clusters': agent.cluster_stats.get('comments_in_clusters', 0),
            'llm_calls_saved': agent.cluster_stats.get('llm_calls_saved', 0),
            'efficiency_percentage': efficiency_percentage
        },
        'comment_preservation': {
            'original_count': original_count,
            'processed_count': processed_count,
            'all_preserved': original_count == processed_count
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Results saved: {all_file}, {promo_file}, {summary_file}")
    
    # Show sample promotional comments
    if len(high_intensity_promo) > 0:
        print(f"\nSample High Intensity Promotional Content:")
        for _, row in high_intensity_promo.head(2).iterrows():
            print(f"FID: {row['fid']}")
            print(f"Type: {row['promo_type']} (Promotional Intensity: {row['promotional_intensity']})")
            print(f"Text: {row['cleaned_transcript'][:100]}...")
            print(f"Keywords: {', '.join(row['keywords']) if isinstance(row['keywords'], list) else 'None'}")
            print()
    elif len(promo_comments) > 0:
        print(f"\nSample Promotional Content (No High Intensity Examples):")
        for _, row in promo_comments.head(2).iterrows():
            print(f"FID: {row['fid']}")
            print(f"Type: {row['promo_type']} (Promotional Intensity: {row['promotional_intensity']})")
            print(f"Text: {row['cleaned_transcript'][:100]}...")
            print(f"Keywords: {', '.join(row['keywords']) if isinstance(row['keywords'], list) else 'None'}")
            print()
    
    return len(results_df), len(promo_comments)

if __name__ == "__main__":
    import argparse
    import glob
    import os
    
    parser = argparse.ArgumentParser(description='Promotional Content Detection Agent')
    parser.add_argument('--brand', help='Single brand to analyze')
    parser.add_argument('--brand_filename', default='brand_info_new.json', help='Brand info JSON filename')
    parser.add_argument('--input', help='Single input CSV file')
    #parser.add_argument('--input_dir', default='processed_data', help='Directory with relevant comments')
    parser.add_argument('--output_dir', default='processed_data', help='Output directory for all analysis results')
    parser.add_argument('--process_all', action='store_true', help='Process all brands in input_dir')
    parser.add_argument('--model_id', default='us.anthropic.claude-3-5-haiku-20241022-v1:0', help='Model ID')
    parser.add_argument('--region', default='us-east-2', help='AWS region')
    parser.add_argument('--sample', type=int, help='Sample size per brand')
    parser.add_argument('--max_workers', type=int, default=30, help='Max parallel workers')
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocess all brands (ignore existing results)')
    
    args = parser.parse_args()
    
    if args.process_all:
        # Process all relevant comment files in brand folders
        csv_files = glob.glob(os.path.join(args.output_dir, 'prefilter/*/*_comments_relevant.csv'))
        print(f"Found {len(csv_files)} relevant comment files to analyze")
        
        # Check which brands are already processed
        to_process = []
        skipped = []
        
        for file_path in sorted(csv_files):
            brand_name = os.path.basename(os.path.dirname(file_path))
            summary_file = f"{args.output_dir}/promo_analysis/{brand_name}/{brand_name}_promo_summary.json"
            
            if os.path.exists(summary_file) and not args.force_reprocess:
                skipped.append(brand_name)
                print(f"Skipping {brand_name} (already processed)")
            else:
                to_process.append(file_path)
                if os.path.exists(summary_file) and args.force_reprocess:
                    print(f"Force reprocessing {brand_name}")
        
        print(f"\nProcessing Status:")
        print(f"  • To process: {len(to_process)} brands")
        print(f"  • Already done: {len(skipped)} brands")
        
        if len(skipped) > 0:
            print(f"  • Skipped: {', '.join(skipped)}")
        
        if args.force_reprocess:
            print(f"  • Force reprocess mode: ON")
        
        if len(to_process) == 0:
            print("\n✅ All brands already processed!")
            print("Use --force_reprocess to reprocess all brands")
            exit(0)
        
        total_analyzed = 0
        total_promos = 0
        
        for file_path in to_process:
            try:
                analyzed, promos = process_single_brand(
                    file_path=file_path,
                    model_id=args.model_id,
                    region=args.region,
                    brand_name=None,  # Extract from file path
                    brand_filename=args.brand_filename,
                    max_workers=args.max_workers,
                    sample=args.sample,
                    output_dir=args.output_dir
                )
                total_analyzed += analyzed
                total_promos += promos
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"\n=== OVERALL PROMOTIONAL SUMMARY ===")
        print(f"Total Analyzed: {total_analyzed}")
        print(f"Total Promotional: {total_promos}")
        print(f"Overall Promo Rate: {total_promos/total_analyzed*100:.1f}%" if total_analyzed > 0 else "0%")
        os.makedirs(f'{args.output_dir}/promo_analysis', exist_ok=True)
        with open(f'{args.output_dir}/promo_analysis/overall_summary.json', 'w') as f:
            json.dump({
                'total_brands': len(csv_files),
                'total_analyzed': total_analyzed,
                'total_promotional': total_promos,
                'overall_promo_rate': total_promos/total_analyzed if total_analyzed > 0 else 0,
                'sample_size': args.sample if args.sample else 'all',
                'processing_method': 'clustered'
            }, f, indent=2)
        
        print(f"Overall summary saved: {args.output_dir}/promo_analysis/overall_summary.json")
        
    elif args.brand and args.input:
        # Process single brand
        process_single_brand(
            file_path=args.input,
            model_id=args.model_id,
            region=args.region,
            brand_name=args.brand,
            brand_filename=args.brand_filename,
            max_workers=args.max_workers,
            sample=args.sample,
            output_dir=args.output_dir
        )
        
    else:
        print("Usage: Either use --process_all or provide --brand and --input")
        print("Examples:")
        print("  python promo_detection_agent.py --process_all --max_workers 8")
        print("  python promo_detection_agent.py --brand rhode --input processed_filter/rhode/rhode_comments_relevant.csv --max_workers 8")