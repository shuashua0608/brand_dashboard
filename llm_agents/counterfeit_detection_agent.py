import sys
import os
sys.path.append('..')
import json
import pandas as pd
import time
import re
import glob
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from bedrock_api_call import BedrockClaudeDistiller
from load_brand_info import load_brand_info

class CounterfeitDetectionAgent:
    def __init__(self, model_id, region):
        """Initialize the counterfeit detection agent"""
        self.distiller = BedrockClaudeDistiller(model_id=model_id, region=region)
        self.cluster_stats = {
            'total_clusters': 0,
            'comments_in_clusters': 0,
            'llm_calls_saved': 0
        }
    
    def analyze_comment(self, comment_data, brand_name, brand_filename):
        """Analyze a comment for counterfeit/fake product concerns"""
        transcript = comment_data.get('transcript', '')
        
        # Get brand info
        brand_info = load_brand_info(brand_name, brand_filename)
        brand_category = brand_info.get('category', '') if brand_info else ''
        brand_description = brand_info.get('description', '') if brand_info else ''
        
        prompt = f"""Analyze this comment for counterfeit risks related to {brand_name}.
Brand: {brand_name}
Category: {brand_category}
Description: {brand_description}
Comment: {transcript}

=== COUNTERFEIT TERMS TO LOOK FOR ===
- DIRECT COUNTERFEIT: fake, counterfeit, replica, knockoff, bootleg, imitation, forgery, fraudulent, bogus, phony, sham, dupe, clone, copy, copycat, rip-off, look-alike
- SLANG/INTERNET: sus, sketch, sketchy, not legit, fugazi, not real, not authentic, questionable, dodgy, rep, reps
- RISKY PLATFORMS: DHgate, AliExpress, Wish, Temu, Shein, Taobao, Alibaba, street vendor, flea market, random seller
- SUSPICIOUS SOURCES: no receipt, no box, no packaging, bought from friend, cheap price, too good to be true, got it cheap
- QUALITY/AUTHENTICITY: cheap version, poor quality, doesn't last, off-brand, generic brand, seems fake, doesn't look right, wrong packaging, not like original, packaging looks off, color is wrong
- AMAZON-SPECIFIC: Amazon fake, Amazon counterfeit, Amazon seller issue, not from official store

=== RISK CLASSIFICATION ===
Risk levels (only if COUNTERFEIT_RISK is Yes):
- High: Direct counterfeit accusations, explicit fake product complaints, clear fraudulent activity mentions
- Medium: Authenticity doubts, quality issues suggesting counterfeits, suspicious source mentions, comparison to authentic products
- Low: Vague quality complaints, unclear authenticity concerns, general suspicions without specific evidence

Risk types:
- Amazon_fake_concern: Amazon-related counterfeit issues
- Fake_product: Clear counterfeit/fake product issues (non-Amazon)  
- Quality_authenticity: Quality/authenticity concerns
- Suspicious_source: Risky platforms or questionable sources

Respond with:
COUNTERFEIT_RISK: Yes/No
HAS_CCR_RISK: Yes/No (Yes if Amazon-related counterfeit risk detected)
RISK_LEVEL: [High, Medium, Low] (only if COUNTERFEIT_RISK is Yes)
RISK_TYPE: [Amazon_fake_concern, Fake_product, Quality_authenticity, Suspicious_source] (ONLY if COUNTERFEIT_RISK is Yes)
REASONING: Brief explanation focusing on specific risk indicators found
KEYWORDS: Key counterfeit-related phrases or terms identified
"""
        
#         """Analyze this comment for counterfeit risks related to {brand_name}.

# Brand: {brand_name} | Category: {brand_category}
# Comment: {transcript}

# COUNTERFEIT INDICATORS:
# • Direct: fake, counterfeit, replica, knockoff, bootleg, imitation, fraudulent, bogus, phony, dupe, clone, copy, rip-off
# • Slang: sus, sketch, not legit, fugazi, not real, not authentic, questionable, dodgy, rep, reps
# • Risky platforms: DHgate, AliExpress, Wish, Temu, Shein, Taobao, Alibaba, street vendor, flea market
# • Suspicious sources: no receipt, no box, cheap price, too good to be true, bought from friend
# • Quality issues: poor quality, doesn't last, off-brand, seems fake, wrong packaging, not like original
# • Amazon-specific: Amazon fake, Amazon counterfeit, Amazon seller issue, not from official store

# Detect related phrases, slang, misspellings, and contextual indicators of these concepts.

# CLASSIFICATION:

# Risk Types:
# • Amazon_fake_concern: Amazon + fake/counterfeit mentions
# • Fake_product: Clear counterfeit evidence (non-Amazon)
# • Quality_issue: Quality problems suggesting authenticity concerns
# • Suspicious_source: Risky platforms or questionable sources
# • Authenticity_question: Doubts about product authenticity
# • None: No counterfeit concerns

# Risk Levels:
# • Critical: Amazon_fake_concern (ALWAYS Critical - no exceptions)
# • High: Clear counterfeit evidence (non-Amazon)
# • Medium: Quality/authenticity concerns, suspicious sources
# • Low: Minimal/unclear risk indicators
# • None: No risk

# MANDATORY RULE: Amazon_fake_concern = Critical (always)

# Respond:
# COUNTERFEIT_RISK: Yes/No
# RISK_LEVEL: [Critical, High, Medium, Low, None]
# RISK_TYPE: [Amazon_fake_concern, Fake_product, Quality_issue, Suspicious_source, Authenticity_question, None]
# REASONING: Brief explanation
# KEYWORDS: Key phrases found"""
        
        try:
            response = self.distiller.call_claude_api(prompt)
            content = response['content'][0]['text']
            
            # Parse response
            counterfeit_risk = False
            risk_match = re.search(r'COUNTERFEIT_RISK:\s*(Yes|No)', content, re.IGNORECASE)
            if risk_match:
                counterfeit_risk = risk_match.group(1).lower() == 'yes'
            
            risk_level = "None"
            level_match = re.search(r'RISK_LEVEL:\s*(High|Medium|Low|None)', content, re.IGNORECASE)
            #level_match = re.search(r'RISK_LEVEL:\s*(Critical|High|Medium|Low|None)', content, re.IGNORECASE)
            if level_match:
                risk_level = level_match.group(1)
            
            risk_type = "None"
            type_match = re.search(r'RISK_TYPE:\s*(Amazon_fake_concern|Fake_product|Quality_issue|Suspicious_source|Authenticity_question|None)', content, re.IGNORECASE)
            if type_match:
                risk_type = type_match.group(1)
            
            reasoning = ""
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=KEYWORDS:|$)', content, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()[:200]
            
            keywords = []
            keywords_match = re.search(r'KEYWORDS:\s*(.+?)(?:\n|$)', content, re.DOTALL)
            if keywords_match:
                keywords_text = keywords_match.group(1).strip()
                keywords = [k.strip() for k in re.split(r'[,\n-]', keywords_text) if k.strip()][:5]
            
            # Risk detection logic
            valid_risk_levels = ['high', 'medium', 'low']
            valid_risk_types = ['Amazon_fake_concern', 'Fake_product', 'Quality_issue', 'Suspicious_source', 'Authenticity_question']
            
            has_risk_level = risk_level.lower() in valid_risk_levels
            has_valid_risk_type = risk_type in valid_risk_types
            
            if risk_type == 'Amazon_fake_concern' and risk_level.lower() != 'high':
                print(f"⚠️  VALIDATION ERROR: Amazon_fake_concern classified as {risk_level}, correcting to High")
                risk_level = 'High'

            return {
                'counterfeit_risk': counterfeit_risk,
                'has_ccr_risk': has_risk_level and has_valid_risk_type,
                'risk_level': risk_level,
                'risk_type': risk_type,
                'reasoning': reasoning,
                'keywords': keywords
            }
                
        except Exception as e:
            return {
                'counterfeit_risk': False,
                'has_ccr_risk': False,
                'risk_level': 'None',
                'risk_type': 'None',
                'reasoning': f"Error: {str(e)[:100]}",
                'keywords': []
            }
    

    
    def process_comments(self, df, brand_name, brand_filename, max_workers=8): 
        """Process comments with clustering optimization
        
        Uses clustering metadata from prefilter agent to avoid redundant LLM calls
        """
        print(f"Processing {len(df)} comments with clustering optimization and {max_workers} workers...")
        
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
                    {'transcript': row.get('transcript', ''), 'cleaned_transcript': row.get('cleaned_transcript', '')}, 
                    brand_name, brand_filename): row 
                for _, row in representative_df.iterrows()
            }
            
            # Collect results as they complete with progress bar
            for future in tqdm(as_completed(future_to_row), total=len(representative_df), desc=f"CCR Analysis ({max_workers} workers)"):
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
                        'counterfeit_risk': False,
                        'has_ccr_risk': False,
                        'risk_level': 'None',
                        'risk_type': 'None',
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
            llm_cols = ['counterfeit_risk', 'has_ccr_risk', 'risk_level', 'risk_type', 'reasoning', 'keywords']
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
                    'counterfeit_risk': False,
                    'has_ccr_risk': False,
                    'risk_level': 'None',
                    'risk_type': 'None',
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
                        'counterfeit_risk': False,
                        'has_ccr_risk': False,
                        'risk_level': 'None',
                        'risk_type': 'None',
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

def process_single_brand(file_path=None, model_id=None, region=None, brand_name=None, brand_filename=None, max_workers=8, sample=None, output_dir="processed_data"): 
    """Process a single brand for counterfeit detection"""
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
    print(f"\n=== CCR Analysis: {brand_display} ===")
    
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
    agent = CounterfeitDetectionAgent(model_id=model_id, region=region)
    
    # Process comments with concurrent batching
    print(f"Concurrent CCR analysis...")
    start_time = time.time()
    results_df = agent.process_comments(df, brand_name, brand_filename, max_workers)
    processing_time = time.time() - start_time
    
    # Filter counterfeit risk comments using new Yes/No indicator
    ccr_risks = results_df[results_df['counterfeit_risk'] == True]
    
    # Display results
    print(f"Processed: {len(results_df)} | CCR Risks: {len(ccr_risks)}")
    print(f"Risk Rate: {len(ccr_risks)/len(results_df)*100:.1f}%")
    print(f"Processing Time: {processing_time:.1f}s")
    print(f"Throughput: {len(results_df)/processing_time:.1f} comments/second")
    
    # Display risk level distribution
    if len(ccr_risks) > 0:
        risk_levels = ccr_risks[ccr_risks['risk_level'] != 'None']['risk_level'].value_counts()
        if not risk_levels.empty:
            print(f"Risk Level Distribution: {dict(risk_levels)}")
        
        risk_types = ccr_risks[ccr_risks['risk_type'] != 'None']['risk_type'].value_counts()
        if not risk_types.empty:
            print(f"Risk Type Distribution: {dict(risk_types)}")
    
    # Display clustering efficiency
    propagated_count = results_df['result_propagated'].sum() if 'result_propagated' in results_df.columns else 0
    if propagated_count > 0:
        efficiency_percentage = (propagated_count / len(results_df)) * 100
        print(f"Clustering Efficiency: {propagated_count} LLM calls saved ({efficiency_percentage:.1f}%)")
    
    # Save results in brand-specific folder
    ccr_dir = f"{output_dir}/ccr_analysis"
    brand_dir = f"{ccr_dir}/{brand_name}"
    os.makedirs(brand_dir, exist_ok=True)
    
    all_file = f"{brand_dir}/{brand_name}_ccr_all.csv"
    risk_file = f"{brand_dir}/{brand_name}_ccr_risk.csv"
    summary_file = f"{brand_dir}/{brand_name}_ccr_summary.json"
    
    results_df.to_csv(all_file, index=False)
    ccr_risks.to_csv(risk_file, index=False)
    
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
    
    # Calculate risk level distribution
    risk_level_distribution = {}
    if len(ccr_risks) > 0:
        # Filter out 'None' risk levels and count distribution
        valid_risk_levels = ccr_risks[ccr_risks['risk_level'] != 'None']['risk_level'].value_counts().to_dict()
        risk_level_distribution = valid_risk_levels
        if risk_level_distribution:
            print(f"Risk Level Distribution: {risk_level_distribution}")
    else:
        print("No CCR risks found - risk level distribution will be empty")
    
    # Save summary in requested format
    summary = {
        'brand': brand_name,
        'total_processed': len(results_df),
        'total_ccr_risks': len(ccr_risks),
        'ccr_risk_rate': len(ccr_risks)/len(results_df) if len(results_df) > 0 else 0,
        'processing_method': 'clustered',
        'processing_time': processing_time,
        'throughput': len(results_df)/processing_time if processing_time > 0 else 0,
        'sample_size': sample if sample else 'all',
        'risk_breakdown': ccr_risks[ccr_risks['risk_type'] != 'None']['risk_type'].value_counts().to_dict() if len(ccr_risks) > 0 else {},
        'risk_level_distribution': risk_level_distribution,
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
        json.dump(summary, f, indent=2)
    
    print(f"Results saved: {all_file}, {risk_file}, {summary_file}")
    
    # Show sample high-risk comments
    if len(ccr_risks) > 0:
        print(f"\nSample CCR Risks:")
        for _, row in ccr_risks.head(2).iterrows():
            print(f"FID: {row['fid']}")
            print(f"Risk: {row['risk_level']} - {row['risk_type']}")
            print(f"Text: {row['cleaned_transcript'][:100]}...")
            print(f"Keywords: {', '.join(row['keywords']) if row['keywords'] else 'None'}")
            print()
    
    return len(results_df), len(ccr_risks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CCR Detection Agent for Brand Monitoring')
    parser.add_argument('--brand', help='Single brand to analyze')
    parser.add_argument('--brand_filename', default='brand_info_new.json')
    parser.add_argument('--input', help='Single input CSV file')
    #parser.add_argument('--input_dir', default='processed_data', help='Directory with relevant comments')
    parser.add_argument('--output_dir', default='processed_data', help='Output directory for all analysis results')
    parser.add_argument('--process_all', action='store_true', help='Process all brands in input_dir')
    parser.add_argument('--model_id', default='us.anthropic.claude-3-5-haiku-20241022-v1:0', help='Model ID')
    parser.add_argument('--region', default='us-east-2', help='AWS region')
    parser.add_argument('--sample', type=int, help='Sample size per brand')
    parser.add_argument('--max_workers', type=int, default=30, help='Max parallel workers (reduced for connection limits)')
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
            summary_file = f"{args.output_dir}/ccr_analysis/{brand_name}/{brand_name}_ccr_summary.json"
            
            if os.path.exists(summary_file) and not args.force_reprocess:
                skipped.append(brand_name)
                print(f"⏭️  Skipping {brand_name} (already processed)")
            else:
                to_process.append(file_path)
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
        
        total_analyzed = 0
        total_risks = 0
        
        for file_path in to_process:
            try:
                analyzed, risks = process_single_brand(
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
                total_risks += risks
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"\n=== OVERALL CCR SUMMARY ===")
        print(f"Total Analyzed: {total_analyzed}")
        print(f"Total CCR Risks: {total_risks}")
        print(f"Overall Risk Rate: {total_risks/total_analyzed*100:.1f}%" if total_analyzed > 0 else "0%")
        
        # Save overall summary
        overall_summary = {
            'total_brands': len(csv_files),
            'total_analyzed': total_analyzed,
            'total_ccr_risks': total_risks,
            'overall_risk_rate': total_risks/total_analyzed if total_analyzed > 0 else 0,
            'sample_size': args.sample if args.sample else 'all',
            'processing_method': 'clustered'
        }
        
        os.makedirs(f'{args.output_dir}/ccr_analysis', exist_ok=True)
        with open(f'{args.output_dir}/ccr_analysis/overall_ccr_summary.json', 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        print(f"Overall summary saved: {args.output_dir}/ccr_analysis/overall_ccr_summary.json")
        
    elif args.brand and args.input:
        # Process single brand
        analyzed, risks = process_single_brand(
            file_path=args.input, 
            model_id=args.model_id, 
            region=args.region, 
            brand_name=args.brand,
            max_workers=args.max_workers, 
            sample=args.sample, 
            output_dir=args.output_dir
        )
        
    else:
        print("Usage: Either use --process_all or provide --brand and --input")
        print("Examples:")
        print("  python counterfeit_detection_agent.py --process_all --max_workers 12")
        print("  python counterfeit_detection_agent.py --brand lattafa --input processed_filter/lattafa/lattafa_comments_relevant.csv ")