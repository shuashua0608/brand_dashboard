import pandas as pd
import time
from typing import Dict, List, Optional
import sys
import os
sys.path.append('..')
from bedrock_api_call import BedrockClaudeDistiller
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import re
from datetime import datetime
from load_brand_info import load_brand_info
from data_preprocesser import DataPreprocessor

class PrefilterAgent:
    def __init__(self, model_id, region, brand_name: str, brand_filename, min_cluster_size: int = 50, 
                 start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Initialize the prefilter agent
        
        Args:
            model_id: LLM model ID
            region: AWS region
            brand_name: Name of the brand to analyze
            min_cluster_size: Minimum number of repetitions to consider a cluster
            start_date: Start date for filtering (YYYY-MM-DD format)
            end_date: End date for filtering (YYYY-MM-DD format)
        """
        self.brand_name = brand_name.lower()
        self.distiller = BedrockClaudeDistiller(model_id=model_id,
            region=region)
        
        # Load brand info
        self.brand_info = load_brand_info(brand_name, brand_filename)
        if self.brand_info:
            self.brand_category = self.brand_info.get('category', '')
            self.brand_description = self.brand_info.get('description', '')
        else:
            self.brand_category = ''
            self.brand_description = ''
            print(f"WARNING: No brand info found for {brand_name} or brand is in skip list")
        
        # Initialize data preprocessor
        self.preprocessor = DataPreprocessor()
        # Compile brand regex pattern
        self.brand_pattern = re.compile(rf'\b{re.escape(brand_name)}\b', re.IGNORECASE)
        # Clustering parameters
        self.min_cluster_size = min_cluster_size
        
        # Date filtering parameters
        self.start_date = start_date
        self.end_date = end_date
        if start_date:
            try:
                self.start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                print(f"Warning: Invalid start_date format '{start_date}'. Expected YYYY-MM-DD. Date filtering disabled.")
                self.start_date = None
                self.start_date_obj = None
        else:
            self.start_date_obj = None
            
        if end_date:
            try:
                self.end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                print(f"Warning: Invalid end_date format '{end_date}'. Expected YYYY-MM-DD. Date filtering disabled.")
                self.end_date = None
                self.end_date_obj = None
        else:
            self.end_date_obj = None
    
    def filter_by_date_range(self, df: pd.DataFrame, timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """Filter DataFrame by date range
        
        Args:
            df: DataFrame to filter
            timestamp_column: Column name containing timestamps
            
        Returns:
            Filtered DataFrame
        """
        if not self.start_date and not self.end_date:
            print("No date filtering applied - processing all data")
            return df
        
        if timestamp_column not in df.columns:
            print(f"Warning: Timestamp column '{timestamp_column}' not found. Available columns: {list(df.columns)}")
            print("Date filtering skipped - processing all data")
            return df
        
        original_count = len(df)
        filtered_df = df.copy()
        
        # Convert timestamp column to datetime if it's not already
        try:
            if not pd.api.types.is_datetime64_any_dtype(filtered_df[timestamp_column]):
                filtered_df[timestamp_column] = pd.to_datetime(filtered_df[timestamp_column], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not convert timestamp column to datetime: {e}")
            print("Date filtering skipped - processing all data")
            return df
        
        # Apply date filters
        if self.start_date_obj:
            before_start_filter = len(filtered_df)
            filtered_df = filtered_df[filtered_df[timestamp_column] >= self.start_date_obj]
            after_start_filter = len(filtered_df)
            print(f"Start date filter ({self.start_date}): {before_start_filter} -> {after_start_filter} comments")
        
        if self.end_date_obj:
            before_end_filter = len(filtered_df)
            filtered_df = filtered_df[filtered_df[timestamp_column] <= self.end_date_obj]
            after_end_filter = len(filtered_df)
            print(f"End date filter ({self.end_date}): {before_end_filter} -> {after_end_filter} comments")
        
        filtered_count = len(filtered_df)
        print(f"Date filtering summary: {original_count} -> {filtered_count} comments ({filtered_count/original_count*100:.1f}% retained)")
        
        return filtered_df
    
    def get_llm_relevance(self, original_text: str, platform: str) -> Dict[str, str]:
        """Get LLM assessment of relevance using original text"""
        prompt = f"""
      This is a PRE-FILTER to identify brand-relevant content. Be INCLUSIVE and keep content meaningfully related to the brand.

        Brand: {self.brand_name}
        Category: {self.brand_category}
        Description: {self.brand_description}
        Comment: {original_text}
        Platform: {platform}

        KEEP if the comment meets AT LEAST ONE of these criteria:
        - Contains brand-related opinions, reviews or experiences
        - Discusses products, features, or similar product/brands comparison related to {self.brand_name}
        - Mentions the brand in context of its category ({self.brand_category})

        FILTER OUT if:
        - {self.brand_name} appears randomly with no real connection
        - Completely unrelated content with coincidental mention
        - Misspellings referring to different words
        - Long irrelevant story where brand name just happens to appear

        Respond with:
        RELEVANT: Yes/No
        REASONING: 1-2 sentences explaining why this content is relevant or not relevant to {self.brand_name}"""
        
        try:
            response = self.distiller.call_claude_api(prompt)
            text_response = response['content'][0]['text']
            
            # Parse response
            relevant = False
            if 'RELEVANT:' in text_response:
                relevant_part = text_response.split('RELEVANT:')[1].split('\n')[0].strip().lower()
                relevant = 'yes' in relevant_part
            reasoning = text_response.split('REASONING:')[1].strip() if 'REASONING:' in text_response else "LLM analysis"
            
            return {
                'relevant': relevant,
                'reasoning': reasoning,  # Limit reasoning length
                'confidence': 'Medium'
            }
        except Exception as e:
            return {
                'relevant': False,
                'reasoning': f"Error: {str(e)[:50]}",
                'confidence': 'Low'
            }
    
    def analyze_comment(self, comment_data):
        """Process a single comment with metadata extraction and LLM relevance check"""
        transcript = comment_data.get('transcript', '')
        platform = comment_data.get('social_media_channel', 'Unknown')
        start_time = time.time()
        
        # Quick check for empty or very short content
        if not transcript or len(transcript.strip()) < 5:
            return {
                'relevant': False,
                'reasoning': 'Content too short or empty',
                'confidence': 'High',
                'processing_time': time.time() - start_time
            }
        
        # Use LLM for relevance analysis
        llm_result = self.get_llm_relevance(transcript, platform)
        llm_result['processing_time'] = time.time() - start_time
        return llm_result
    
    def detect_clusters(self, df: pd.DataFrame, text_column: str = 'cleaned_transcript') -> pd.DataFrame:
        """Detect clusters of similar comments
        
        Args:
            df: DataFrame containing comments
            text_column: Column name containing the text to analyze
            
        Returns:
            DataFrame with added cluster information
        """
        print(f"Detecting comment clusters (min size: {self.min_cluster_size})...")
        
        # Count occurrences of each comment
        value_counts = df[text_column].value_counts()
        
        # Find repeated comments meeting the minimum cluster size
        repeated_comments = value_counts[value_counts >= self.min_cluster_size]
        
        # Create cluster IDs for repeated comments
        cluster_mapping = {}
        for i, (comment, count) in enumerate(repeated_comments.items()):
            cluster_id = f"cluster_{i+1}"
            cluster_mapping[comment] = {
                'cluster_id': cluster_id,
                'repetition_count': count
            }
        
        # Add cluster information to the dataframe
        result_df = df.copy()
        
        # Initialize cluster columns
        result_df['cluster_id'] = None
        result_df['repetition_count'] = 1
        result_df['is_representative'] = False
        
        # Track which clusters we've seen
        seen_clusters = set()
        
        # Add cluster information
        for idx, row in result_df.iterrows():
            text = row[text_column]
            if text in cluster_mapping:
                cluster_info = cluster_mapping[text]
                cluster_id = cluster_info['cluster_id']
                result_df.at[idx, 'cluster_id'] = cluster_id
                result_df.at[idx, 'repetition_count'] = cluster_info['repetition_count']
                
                # Mark the first occurrence of each cluster as representative
                if cluster_id not in seen_clusters:
                    result_df.at[idx, 'is_representative'] = True
                    seen_clusters.add(cluster_id)
        
        # Generate clustering summary
        total_comments = len(df)
        clustered_comments = result_df[result_df['cluster_id'].notna()].shape[0]
        unique_clusters = len(repeated_comments)
        efficiency_gain = clustered_comments - unique_clusters
        
        print(f"\n=== COMMENT CLUSTERING SUMMARY ===")
        print(f"Total comments: {total_comments}")
        print(f"Comments in clusters: {clustered_comments} ({clustered_comments/total_comments*100:.1f}%)")
        print(f"Unique clusters: {unique_clusters}")
        print(f"Efficiency gain: {efficiency_gain} fewer LLM calls ({efficiency_gain/total_comments*100:.1f}%)")
        
        return result_df
    
    def process_comments(self, df: pd.DataFrame, max_workers: int = 30) -> pd.DataFrame:
        """Process comments with true parallel processing and clustering"""
        print(f"Initial dataset: {len(df)} comments")
        
        # Apply date filtering first
        df = self.filter_by_date_range(df)
        
        print(f"Processing {len(df)} comments with {max_workers} workers...")
        all_results = []
        
        # Ensure each row has a unique identifier for checkpoint tracking
        if 'fid' not in df.columns:
            df['fid'] = df.index.astype(str)
        
        # Preprocess all comments to extract metadata and clean text
        preprocessed_data = []
        for _, row in df.iterrows():
            transcript = row.get('transcript', '')
            metadata = self.preprocessor.extract_metadata(transcript)
            cleaned_text = self.preprocessor.clean_text(transcript)
            
            preprocessed_data.append({
                'fid': row.get('fid', ''),
                'transcript': transcript,
                'cleaned_transcript': cleaned_text,
                'timestamp': row.get('timestamp', ''),
                'text_language': row.get('text_language', ''),
                'social_media_channel': row.get('SocialMediaChannel', ''),
                'hashtags': metadata.get('hashtags', ''),
                'mentions': metadata.get('mentions', ''),
                'emojis': metadata.get('emojis', '')
            })
        
        # Create DataFrame with preprocessed data
        preprocessed_df = pd.DataFrame(preprocessed_data)
        # Detect clusters
        clustered_df = self.detect_clusters(preprocessed_df, 'cleaned_transcript')
        # Select only representative comments for LLM processing
        # These are either cluster representatives or comments that don't belong to any cluster
        representative_df = clustered_df[clustered_df['is_representative'] | clustered_df['cluster_id'].isna()]
        print(f"Processing {len(representative_df)} representative comments with LLM...")
        # Process representative comments with LLM
        representative_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit LLM analysis tasks for representative comments
            future_to_row = {
                executor.submit(self.analyze_comment, 
                    {'transcript': row.get('transcript', ''), 
                     'social_media_channel': row.get('social_media_channel', 'Unknown')}): row 
                for _, row in representative_df.iterrows()
            }
            
            # Collect results as they complete with progress bar
            for future in tqdm(as_completed(future_to_row), total=len(representative_df), desc=f"LLM Analysis ({max_workers} workers)"):
                row = future_to_row[future]
                try:
                    api_result = future.result()
                    representative_results.append({**row.to_dict(), **api_result})
                except Exception as e:
                    representative_results.append({
                        **row.to_dict(),
                        'relevant': False,
                        'reasoning': f"Error: {str(e)[:50]}",
                        'confidence': 'Low',
                        'processing_time': 0.0
                    })
        
        # Create a DataFrame with the LLM results for representative comments
        representative_results_df = pd.DataFrame(representative_results)
        
        # Create a mapping from cluster_id to LLM results
        cluster_results = {}
        for _, row in representative_results_df[representative_results_df['cluster_id'].notna()].iterrows():
            cluster_id = row['cluster_id']
            # Extract only the columns that were added by the LLM analysis
            llm_cols = ['relevant', 'reasoning', 'confidence', 'processing_time']
            cluster_results[cluster_id] = {col: row[col] for col in llm_cols}
        
        # Apply the results to all comments
        final_results = []
        
        for _, row in clustered_df.iterrows():
            row_dict = row.to_dict()
            cluster_id = row_dict.get('cluster_id')
            
            if pd.notna(cluster_id) and not row_dict.get('is_representative', False):
                # This is a non-representative comment in a cluster
                # Use the results from the cluster's representative
                cluster_result = cluster_results.get(cluster_id, {
                    'relevant': False,
                    'reasoning': 'No representative result found',
                    'confidence': 'Low',
                    'processing_time': 0.0
                })
                
                final_results.append({**row_dict, **cluster_result, 'result_propagated': True})
            else:
                # This is either a representative comment or a comment not in any cluster
                # Find the corresponding row in representative_results_df
                matching_rows = representative_results_df[representative_results_df['fid'] == row_dict.get('fid', '')]
                if not matching_rows.empty:
                    final_results.append({**matching_rows.iloc[0].to_dict(), 'result_propagated': False})
                else:
                    # This shouldn't happen, but just in case
                    final_results.append({
                        **row_dict,
                        'relevant': False,
                        'reasoning': 'Result not found',
                        'confidence': 'Low',
                        'processing_time': 0.0,
                        'result_propagated': False
                    })
        
        # Create the final dataframe
        final_df = pd.DataFrame(final_results)
        
        # Print summary
        propagated_count = final_df['result_propagated'].sum() if 'result_propagated' in final_df.columns else 0
        print(f"Propagated results to {propagated_count} comments")
        print(f"Total comments processed: {len(final_df)}")
        
        return final_df

def process_single_brand(brand_name, brand_filename, model_id, region, max_workers=8, sample=None, output_dir="processed_data", min_cluster_size=50, start_date=None, end_date=None):
    """Process a single brand dataset - prefilter only
    
    Args:
        brand_name: Name of the brand to analyze
        model_id: LLM model ID
        region: AWS region
        max_workers: Maximum number of parallel workers
        sample: Sample size (if None, process all comments)
        output_dir: Output directory for results
        min_cluster_size: Minimum number of repetitions to consider a cluster
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
    """
    import os
    import json
    from datetime import datetime
    
    # Load brand info from JSON
    brand_info = load_brand_info(brand_name, brand_filename)
    if not brand_info:
        print(f"ERROR: Brand '{brand_name}' not found in brand info JSON or is in skip list")
        return 0, 0
    
    # First check if preprocessed data exists
    preprocessed_file = f"{output_dir}/cleaned_brand/{brand_name}.csv"
    if os.path.exists(preprocessed_file):
        print(f"Found preprocessed data: {preprocessed_file}")
        file_path = preprocessed_file
    else:
        # Check for file_path or filepath in brand_info
        file_path = brand_info.get('file_path') or brand_info.get('filepath')
        if not file_path:
            print(f"ERROR: Brand '{brand_name}' missing file_path/filepath in brand info JSON")
            return 0, 0
        
        # Handle relative paths - if file doesn't exist directly, try to resolve relative to current directory
        if not os.path.exists(file_path) and not os.path.isabs(file_path):
            # Try current directory
            current_dir_path = os.path.join(os.getcwd(), file_path)
            if os.path.exists(current_dir_path):
                file_path = current_dir_path
                print(f"Using resolved path: {file_path}")
    
    # Get brand info for display
    print(f"Brand: {brand_name} ({brand_info.get('category', '')})")
    print(f"Description: {brand_info.get('description', '')[:50]}...")
    print(f"File path: {file_path}")
    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return 0, 0

    print(f"\n=== Processing {brand_name.upper()} ===")
    print(f"Clustering threshold: {min_cluster_size} comments")
    if start_date or end_date:
        print(f"Date range: {start_date or 'earliest'} to {end_date or 'latest'}")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Sample if specified
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
        print(f"Processing {len(df)} sample comments for brand '{brand_name}'...")
    else:
        print(f"Processing {len(df)} comments for brand '{brand_name}'...")
    
    # Initialize agent with clustering and date filtering
    agent = PrefilterAgent(model_id=model_id, region=region, brand_name=brand_name, brand_filename=brand_filename, 
                          min_cluster_size=min_cluster_size, start_date=start_date, end_date=end_date)
    
    # Process comments with LLM analysis and clustering
    results_df = agent.process_comments(df, max_workers=max_workers)
    
    # Save results
    if output_dir:
        brand_dir = f"{output_dir}/prefilter/{brand_name}"
    else:
        brand_dir = f"processed_data/prefilter/{brand_name}"
    
    os.makedirs(brand_dir, exist_ok=True)
    print(f"Saving results to {brand_dir}")
    
    # Save all results
    results_df.to_csv(f"{brand_dir}/{brand_name}_comments_all.csv", index=False)
    # Save relevant comments only
    relevant_df = results_df[results_df['relevant'] == True]
    relevant_df.to_csv(f"{brand_dir}/{brand_name}_comments_relevant.csv", index=False)
    
    # Calculate clustering statistics
    total_comments = len(results_df)
    clustered_comments = results_df[results_df['cluster_id'].notna()].shape[0]
    unique_clusters = results_df['cluster_id'].nunique() - (1 if results_df['cluster_id'].isna().any() else 0)
    propagated_results = results_df['result_propagated'].sum() if 'result_propagated' in results_df.columns else 0
    
    # Summary with clustering and date filtering information
    summary = {
        'brand': brand_name,
        'original_comments': len(df),  # Before any processing
        'total_comments': len(results_df),  # After date filtering
        'relevant_comments': len(relevant_df),
        'relevance_rate': len(relevant_df) / len(results_df) * 100 if len(results_df) > 0 else 0,
        'avg_processing_time': results_df['processing_time'].mean(),
        'date_filtering': {
            'start_date': start_date,
            'end_date': end_date,
            'comments_after_date_filter': len(results_df)
        },
        'clustering_info': {
            'total_comments': total_comments,
            'clustered_comments': str(clustered_comments),
            'unique_clusters': unique_clusters,
            'propagated_results': str(propagated_results),
            'llm_calls_saved': str(propagated_results)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{brand_dir}/{brand_name}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n=== {brand_name.upper()} SUMMARY ===")
    print(f"Total: {len(df)} | Relevant: {len(relevant_df)} ({summary['relevance_rate']:.1f}%)")
    print(f"Clustering: {clustered_comments} comments in {unique_clusters} clusters")
    print(f"LLM calls saved: {propagated_results} ({propagated_results/total_comments*100:.1f}% efficiency)")
    print(f"Saved results to {brand_dir}")
    
    return len(results_df), len(relevant_df)

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Prefilter agent for brand comment analysis')
    parser.add_argument('--brand', help='Single brand name to filter for')
    parser.add_argument('--brand_filename', default='brand_info_new.json')
    parser.add_argument('--output_dir', default='processed_data', help='Output directory for all analysis results')
    parser.add_argument('--process_all', action='store_true', help='Process all brands in brand_info_new.json')
    parser.add_argument('--model_id', default='us.anthropic.claude-3-5-haiku-20241022-v1:0', help='Model ID for LLM')
    parser.add_argument('--region', default='us-east-2', help='AWS region')
    parser.add_argument('--sample', type=int, default=None, help='Sample size per brand (default: process all)')
    parser.add_argument('--max_workers', type=int, default=30, help='Max parallel workers for LLM calls')
    parser.add_argument('--min_cluster_size', type=int, default=50, help='Minimum number of repetitions to consider a cluster')
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocess all brands (ignore existing results)')
    parser.add_argument('--start_date', type=str, help='Start date for filtering (YYYY-MM-DD format)')
    parser.add_argument('--end_date', type=str, help='End date for filtering (YYYY-MM-DD format)')
    
    args = parser.parse_args()
    
    if args.process_all: 
        # Load all brands from brand_info_new.json
        brand_data = load_brand_info(brand_filename=args.brand_filename)
        if not brand_data or 'brands' not in brand_data:
            print("ERROR: Could not load brands from brand_info_new.json")
            exit(1)
            
        # Get skip brands list
        skip_brands = brand_data.get('skip_brands', [])
        
        # Filter out brands in skip list
        brands = []
        for brand in brand_data['brands']:
            if brand['name'] not in skip_brands:
                brands.append(brand['name'])
            else:
                print(f"INFO: Skipping brand '{brand['name']}' (in skip_brand_name.txt)")
                
        print(f"Found {len(brands)} brands to process in brand_info_new.json")
        
        # Check which brands are already processed
        to_process = []
        skipped = []
        
        for brand_name in sorted(brands):
            summary_file = f"{args.output_dir}/prefilter/{brand_name}/{brand_name}_summary.json"
            
            if os.path.exists(summary_file) and not args.force_reprocess:
                skipped.append(brand_name)
                print(f"⏭️  Skipping {brand_name} (already processed)")
            else:
                to_process.append(brand_name)
                if os.path.exists(summary_file) and args.force_reprocess:
                    print(f"🔄  Force reprocessing {brand_name}")
        
        print(f"\n Processing Status:")
        print(f"  • To process: {len(to_process)} brands")
        print(f"  • Already done: {len(skipped)} brands")
        
        if len(skipped) > 0:
            print(f"  • Skipped: {', '.join(skipped)}")
        
        if args.force_reprocess:
            print(f"  • Force reprocess mode: ON")
        
        if len(to_process) == 0:
            print("\n All brands already processed!")
            print("Use --force_reprocess to reprocess all brands")
            exit(0)
        
        total_processed = 0
        total_relevant = 0
        
        for brand_name in to_process:
            try:
                processed, relevant = process_single_brand(
                    brand_name, args.brand_filename, args.model_id, args.region, 
                    args.max_workers, args.sample,
                    args.output_dir, args.min_cluster_size, args.start_date, args.end_date
                )
                total_processed += processed
                total_relevant += relevant
            except Exception as e:
                print(f"Error processing {brand_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n=== OVERALL SUMMARY ===")
        print(f"Total processed: {total_processed}")
        print(f"Total relevant: {total_relevant}")
        print(f"Overall relevance rate: {total_relevant/total_processed*100:.1f}%" if total_processed > 0 else "0%")
        
        # Create summary for this run
        current_run_summary = {
            'processed_brands': to_process,
            'total_processed': total_processed,
            'total_relevant': total_relevant,
            'overall_relevance_rate': total_relevant / total_processed if total_processed > 0 else 0,
            'used_llm': True,
            'sample_size': args.sample if args.sample else 'all',
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Load existing overall summary if it exists
        os.makedirs(args.output_dir, exist_ok=True)
        summary_file = f'{args.output_dir}/prefilter_overall_summary.json'
        existing_summary = {}
        
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    existing_summary = json.load(f)
                print(f"Loaded existing summary with {existing_summary.get('total_brands', 0)} brands")
            except json.JSONDecodeError:
                print(f"Warning: Could not parse existing summary file. Creating new one.")
        
        # Get brand-specific summaries
        brand_summaries = {}
        all_brands = set()
        
        # Add existing brand summaries if available
        if 'brand_summaries' in existing_summary:
            brand_summaries = existing_summary['brand_summaries']
            all_brands.update(brand_summaries.keys())
        
        # Add newly processed brands
        for brand_name in to_process:
            brand_summary_file = f"{args.output_dir}/prefilter/{brand_name}/{brand_name}_summary.json"
            if os.path.exists(brand_summary_file):
                try:
                    with open(brand_summary_file, 'r') as f:
                        brand_summary = json.load(f)
                    brand_summaries[brand_name] = brand_summary
                    all_brands.add(brand_name)
                except:
                    print(f"Warning: Could not load summary for {brand_name}")
        
        # Calculate overall totals from all brand summaries
        total_all_processed = 0
        total_all_relevant = 0
        
        for brand_name, summary in brand_summaries.items():
            total_all_processed += summary.get('total_comments', 0)
            total_all_relevant += summary.get('relevant_comments', 0)
        
        # Create updated overall summary
        overall_summary = {
            'total_brands': len(all_brands),
            'processed_brands': list(all_brands),
            'total_processed': total_all_processed,
            'total_relevant': total_all_relevant,
            'overall_relevance_rate': total_all_relevant / total_all_processed if total_all_processed > 0 else 0,
            'used_llm': True,
            'last_run': {
                'timestamp': current_run_summary['timestamp'],
                'processed_brands': current_run_summary['processed_brands'],
                'sample_size': current_run_summary['sample_size']
            },
            'brand_summaries': brand_summaries
        }
        
        # Save updated overall summary
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)
            
            # Save the file with explicit encoding
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(overall_summary, f, indent=2)
                
            print(f"Successfully wrote summary to {summary_file}")
        except Exception as e:
            print(f"Error saving overall summary: {str(e)}")
            # Try an alternative location
            alt_summary_file = 'prefilter_overall_summary.json'
            print(f"Trying to save to alternative location: {alt_summary_file}")
            with open(alt_summary_file, 'w', encoding='utf-8') as f:
                json.dump(overall_summary, f, indent=2)
        
        print(f"Overall summary saved: {summary_file}")
        print(f"Total brands in summary: {len(all_brands)}")
        print(f"Total comments processed across all brands: {total_all_processed}")
        print(f"Total relevant comments across all brands: {total_all_relevant}")
        print(f"Overall relevance rate: {overall_summary['overall_relevance_rate']*100:.1f}%")
        
    elif args.brand:
        # Process single brand
        process_single_brand(
            args.brand, args.brand_filename, args.model_id, args.region, 
            args.max_workers, args.sample,
            args.output_dir, args.min_cluster_size, args.start_date, args.end_date
        )
    
    else:
        print("Usage: Either use --process_all or provide --brand")
        print("Examples:")
        print("  python prefilter_agent.py --process_all")
        print("  python prefilter_agent.py --brand Lattafa")
        print("  python prefilter_agent.py --brand Lattafa --start_date 2024-01-01 --end_date 2024-12-31")
        print("  python prefilter_agent.py --process_all --start_date 2024-06-01")