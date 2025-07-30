import sys
import os
sys.path.append('..')
import json
import pandas as pd
import re
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from bedrock_api_call import BedrockClaudeDistiller
from load_brand_info import load_brand_info

class SentimentAgent:
    def __init__(self, model_id, region):
        """Initialize the sentiment analysis agent"""
        self.distiller = BedrockClaudeDistiller(model_id=model_id, region=region)
        self.cluster_stats = {
            'total_clusters': 0,
            'comments_in_clusters': 0,
            'llm_calls_saved': 0
        }
    
    def analyze_sentiment(self, transcript, brand_name, brand_filename):
        """Simple sentiment analysis focused on customer experience"""
        
        # Get brand info
        brand_info = load_brand_info(brand_name,brand_filename)
        brand_category = brand_info.get('category', '') if brand_info else ''
        brand_description = brand_info.get('description', '') if brand_info else ''
        
        prompt = f"""
        Analyze this comment's sentiment and emotion specifically about {brand_name}:

Brand: {brand_name}
Category: {brand_category}
Description: {brand_description}
Comment: {transcript}

=== SENTIMENT CLASSIFICATION ===
Focus only on sentiment toward the brand, not general comment tone:
- Positive: Favorable, praising, recommending the brand
- Negative: Critical, disappointed, warning against the brand
- Neutral: Factual, indifferent, or no clear brand opinion
- Mixed: Contains both positive and negative brand sentiments

=== EMOTION DETECTION ===
Choose the PRIMARY emotion expressed toward the brand:
- Joy: happiness, delight, excitement, satisfaction, love
- Anger: frustration, irritation, rage, annoyance, outrage
- Sadness: disappointment, regret, sorrow, melancholy, grief
- Fear: worry, anxiety, concern, nervousness, panic
- Surprise: amazement, shock, astonishment, wonder, confusion
- Disgust: revulsion, distaste, contempt, repulsion, aversion
- Trust: confidence, faith, reliability, security, comfort
- Anticipation: excitement, hope, expectation, eagerness, curiosity
- Neutral: no strong emotion expressed

Respond with:
SENTIMENT: [Positive, Negative, Neutral, Mixed]
EMOTION: [Joy, Anger, Sadness, Fear, Surprise, Disgust, Trust, Anticipation, Neutral]
REASONING: Brief explanation focusing on specific words/phrases that indicate the sentiment and emotion toward the brand"""
        
        try:
            response = self.distiller.call_claude_api(prompt)
            content = response['content'][0]['text']
            
            sentiment = "Neutral"
            sentiment_match = re.search(r'SENTIMENT:\s*(Positive|Negative|Neutral|Mixed)', content, re.IGNORECASE)
            if sentiment_match:
                sentiment = sentiment_match.group(1)
            
            emotion = "Neutral"
            emotion_match = re.search(r'EMOTION:\s*(Joy|Anger|Sadness|Fear|Surprise|Disgust|Trust|Anticipation|Neutral)', content, re.IGNORECASE)
            if emotion_match:
                emotion = emotion_match.group(1)
            
            reasoning = ""
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', content, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()[:100]
            
            return {
                'sentiment': sentiment,
                'emotion': emotion,
                'reasoning': reasoning
            }
                
        except Exception as e:
            return {
                'sentiment': 'Neutral',
                'emotion': 'Neutral',
                'reasoning': f"Error: {str(e)[:50]}"
            }
    
    def process_comments(self, df, brand_name, brand_filename, max_workers=8):
        """Process comments with clustering optimization
        
        Uses clustering metadata from prefilter agent to avoid redundant LLM calls
        """
        print(f"Processing {len(df)} comments for sentiment analysis...")
        
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
                executor.submit(self.analyze_sentiment, row.get('transcript', ''), brand_name, brand_filename): row 
                for _, row in representative_df.iterrows()
            }
            
            # Collect results as they complete with progress bar
            for future in tqdm(as_completed(future_to_row), total=len(representative_df), desc=f"Sentiment Analysis ({max_workers} workers)"):
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
                        'cluster_id': original_row.get('cluster_id'),
                        'is_representative': original_row.get('is_representative', False),
                        'repetition_count': original_row.get('repetition_count', 1),
                        'sentiment': 'Neutral',
                        'emotion': 'Neutral',
                        'reasoning': f"Error: {str(e)[:50]}",
                        'result_propagated': False
                    }
                    representative_results.append(error_result)
        
        # Create a DataFrame with the LLM results for representative comments
        representative_results_df = pd.DataFrame(representative_results)
        
        # Create a mapping from cluster_id to LLM results
        cluster_results = {}
        for _, row in representative_results_df[representative_results_df['cluster_id'].notna()].iterrows():
            cluster_id = row['cluster_id']
            # Extract only the columns that were added by the analyze_sentiment method
            llm_cols = ['sentiment', 'emotion', 'reasoning']
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
                    'sentiment': 'Neutral',
                    'emotion': 'Neutral',
                    'reasoning': 'No representative result found'
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
                        'sentiment': 'Neutral',
                        'emotion': 'Neutral',
                        'reasoning': 'Result not found',
                        'result_propagated': False
                    })
        
        # Create the final dataframe
        final_df = pd.DataFrame(final_results)
        
        # Print clustering efficiency summary
        print(f"Clustering Efficiency: {self.cluster_stats['llm_calls_saved']} LLM calls saved")
        print(f"Processed {len(final_df)} comments with {len(representative_df)} LLM calls")
        
        return final_df
    
    # _process_all_comments method removed as we always use clustered data

def calculate_sentiment_statistics(df):
    """Calculate sentiment distribution statistics"""
    sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
    emotion_dist = df['emotion'].value_counts()
    
    # Risk indicators
    negative_emotions = ['Anger', 'Sadness', 'Fear', 'Disgust']
    positive_emotions = ['Joy', 'Trust', 'Anticipation']
    negative_emotion_count = df[df['emotion'].isin(negative_emotions)].shape[0]
    positive_emotion_count = df[df['emotion'].isin(positive_emotions)].shape[0]
    
    return {
        'total_comments': len(df),
        'sentiment_distribution': {
            'positive_pct': round(sentiment_dist.get('Positive', 0), 1),
            'negative_pct': round(sentiment_dist.get('Negative', 0), 1),
            'neutral_pct': round(sentiment_dist.get('Neutral', 0), 1),
            'mixed_pct': round(sentiment_dist.get('Mixed', 0), 1)
        },
        'top_emotions': [
            {'emotion': emotion, 'count': count} 
            for emotion, count in emotion_dist.head(5).items()
        ],
        'negative_emotions_count': negative_emotion_count,
        'negative_emotions_pct': round(negative_emotion_count / len(df) * 100, 1) if len(df) > 0 else 0,
        'positive_emotions_count': positive_emotion_count,
        'positive_emotions_pct': round(positive_emotion_count / len(df) * 100, 1) if len(df) > 0 else 0
    }

def process_brand_sentiment(input_file=None, model_id=None, region=None, brand_name=None, brand_filename='brand_info_new.json', max_workers=8, output_dir="processed_data"):
    """Process sentiment analysis for a single brand"""
    # If only brand name is provided, try to get file path from brand_info.json
    if brand_name and not input_file:
        brand_info = load_brand_info(brand_name,brand_filename)
        if brand_info and 'file_path' in brand_info:
            input_file = brand_info['file_path']
            print(f"Using file path from brand_info.json: {input_file}")
        else:
            # Try to find prefilter results
            prefilter_file = f"{output_dir}/prefilter/{brand_name}/{brand_name}_comments_relevant.csv"
            if os.path.exists(prefilter_file):
                input_file = prefilter_file
                print(f"Using prefilter results: {input_file}")
    
    # Extract brand name from file path if not provided
    if not brand_name and input_file:
        # Try to extract from directory name first
        dir_name = os.path.basename(os.path.dirname(input_file))
        if dir_name and dir_name != '':
            brand_name = dir_name
        else:
            # Fall back to filename
            filename = os.path.basename(input_file)
            if '_' in filename:
                brand_name = filename.split('_')[0]
            else:
                brand_name = filename.replace('.csv', '')
    
    # Ensure we have both brand name and file path
    if not brand_name or not input_file:
        print(f"ERROR: Both brand name and file path are required. Got brand_name={brand_name}, input_file={input_file}")
        return None, None
    
    # Verify file exists
    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        return None, None
    
    try:
        # Load data
        df = pd.read_csv(input_file)
        if len(df) == 0:
            print(f"ERROR: File is empty: {input_file}")
            return None, None
    except Exception as e:
        print(f"ERROR: Cannot read file {input_file}: {e}")
        return None, None
    
    # Ensure brand_name is a string before calling upper()
    brand_display = brand_name.upper() if isinstance(brand_name, str) else str(brand_name).upper()
    print(f"\n=== SENTIMENT ANALYSIS: {brand_display} ===")
    print(f"Processing {len(df)} comments...")
    
    # Initialize and run sentiment analysis
    agent = SentimentAgent(model_id=model_id, region=region)
    results_df = agent.process_comments(df, brand_name, brand_filename, max_workers)
    
    # Calculate statistics
    stats = calculate_sentiment_statistics(results_df)
    
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
    
    # Add clustering efficiency to stats
    stats['clustering_efficiency'] = {
        'total_clusters': agent.cluster_stats.get('total_clusters', 0),
        'comments_in_clusters': agent.cluster_stats.get('comments_in_clusters', 0),
        'llm_calls_saved': agent.cluster_stats.get('llm_calls_saved', 0),
        'efficiency_percentage': efficiency_percentage
    }
    stats['processing_method'] = 'clustered'
    stats['comment_preservation'] = {
        'original_count': original_count,
        'processed_count': processed_count,
        'all_preserved': original_count == processed_count
    }
    
    # Save results
    sentiment_dir = f"{output_dir}/sentiment_analysis"
    brand_dir = f"{sentiment_dir}/{brand_name}"
    os.makedirs(brand_dir, exist_ok=True)
    
    results_df.to_csv(f"{brand_dir}/{brand_name}_sentiment_analysis.csv", index=False)
    
    with open(f"{brand_dir}/{brand_name}_sentiment_summary.json", 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Display summary
    print(f"Sentiment: Negative {stats['sentiment_distribution']['negative_pct']:.1f}% | Positive {stats['sentiment_distribution']['positive_pct']:.1f}%")
    print(f"Negative Emotions: {stats['negative_emotions_count']} ({stats['negative_emotions_pct']:.1f}%)")
    
    # Display clustering efficiency
    if 'clustering_efficiency' in stats:
        efficiency = stats['clustering_efficiency']
        if efficiency['llm_calls_saved'] > 0:
            print(f"Clustering Efficiency: {efficiency['llm_calls_saved']} LLM calls saved ({efficiency['efficiency_percentage']:.1f}%)")
    
    if stats['negative_emotions_pct'] > 20:
        print("⚠️  High negative emotion rate detected")
    
    return results_df, stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentiment Analysis Agent')
    parser.add_argument('--brand', help='Brand name to analyze')
    parser.add_argument('--brand_filename', default='brand_info_new.json')
    parser.add_argument('--input', help='Input CSV file (relevant comments)')
    parser.add_argument('--output_dir', default='processed_data', help='Output directory for all analysis results')
    parser.add_argument('--process_all', action='store_true', help='Process all brands from brand_info.json')
    parser.add_argument('--model_id', default='us.anthropic.claude-3-5-haiku-20241022-v1:0', help='Model ID')
    parser.add_argument('--region', default='us-east-2', help='AWS region')
    parser.add_argument('--max_workers', type=int, default=30, help='Max parallel workers')
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocess all brands (ignore existing results)')
    
    args = parser.parse_args()
    
    if args.process_all:
        # Load all brands from brand_info.json
        brand_data = load_brand_info(brand_filename=args.brand_filename)
        if not brand_data or 'brands' not in brand_data:
            print("ERROR: Could not load brands from brand_info.json")
            exit(1)
            
        brands = [brand['name'] for brand in brand_data['brands']]
        print(f"Found {len(brands)} brands in brand_info.json")
        
        # Check which brands are already processed
        to_process = []
        skipped = []
        
        for brand_name in sorted(brands):
            summary_file = f"{args.output_dir}/sentiment_analysis/{brand_name}/{brand_name}_sentiment_summary.json"
            
            if os.path.exists(summary_file) and not args.force_reprocess:
                skipped.append(brand_name)
                print(f"⏭️  Skipping {brand_name} (already processed)")
            else:
                to_process.append(brand_name)
                if os.path.exists(summary_file) and args.force_reprocess:
                    print(f"🔄  Force reprocessing {brand_name}")
        
        print(f"\n📊 Processing Status:")
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
        
        for brand_name in to_process:
            try:
                # Try to find prefilter results first
                prefilter_file = f"{args.output_dir}/prefilter/{brand_name}/{brand_name}_comments_relevant.csv"
                if os.path.exists(prefilter_file):
                    input_file = prefilter_file
                else:
                    input_file = None  # Let process_brand_sentiment find the file
                    
                results_df, stats = process_brand_sentiment(
                    input_file=input_file,
                    model_id=args.model_id,
                    region=args.region,
                    brand_name=brand_name,
                    brand_filename=args.brand_filename,
                    max_workers=args.max_workers,
                    output_dir=args.output_dir
                )
                
                if results_df is not None:
                    total_analyzed += len(results_df)
            except Exception as e:
                print(f"Error processing {brand_name}: {e}")
        
        print(f"\n=== OVERALL SENTIMENT SUMMARY ===")
        print(f"Total Analyzed: {total_analyzed}")
        
        # Save overall summary
        os.makedirs(f'{args.output_dir}/sentiment_analysis', exist_ok=True)
        with open(f'{args.output_dir}/sentiment_analysis/overall_summary.json', 'w') as f:
            json.dump({
                'total_brands': len(brands),
                'total_analyzed': total_analyzed,
                'processing_method': 'clustered'
            }, f, indent=2)
        
        print(f"Overall summary saved: {args.output_dir}/sentiment_analysis/overall_summary.json")
    
    elif args.brand:
        # Process single brand
        process_brand_sentiment(
            input_file=args.input,  # May be None
            model_id=args.model_id,
            region=args.region,
            brand_name=args.brand,
            brand_filename=args.brand_filename,
            max_workers=args.max_workers,
            output_dir=args.output_dir
        )
    elif args.input:
        # Process file without explicit brand name
        process_brand_sentiment(
            input_file=args.input,
            model_id=args.model_id,
            region=args.region,
            brand_filename=args.brand_filename,
            max_workers=args.max_workers,
            output_dir=args.output_dir
        )
    else:
        print("Usage: --brand BRAND_NAME OR --input file.csv OR --process_all")