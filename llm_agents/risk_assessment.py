import os
import sys
import json
import argparse
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple
sys.path.append('..')
from bedrock_api_call import BedrockClaudeDistiller
from load_brand_info import load_brand_info

class FinalRiskAssessment:
    def __init__(self, model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", region="us-east-2", 
                 brand_filename="brand_info_new.json"):
        """Initialize the final risk assessment agent"""
        self.distiller = BedrockClaudeDistiller(model_id=model_id, region=region)
        self.brand_filename = brand_filename
        
        # Load brand configurations and skip list
        self.brand_configs = self._load_brand_configs()
        self.skip_brands = self._load_skip_brands()
        # Risk scoring weights - optimized based on signal pattern analysis
        self.weights = {
            'comment_intensity': 0.35,   # Increased - strongest differentiator (volume + repetition patterns)
            'ccr_risk': 0.15,           # Reduced - not a strong differentiator between risk levels
            'promotional_risk': 0.25,   # Reduced - promotional rate is similar across all risk levels
            'sentiment_risk': 0.25,     # Maintained - organic negative sentiment has complex patterns
            }
        
        # Risk level thresholds (0-100 scale) - based on signal pattern analysis
        self.risk_thresholds = {
            'critical': 50,  # Critical brands: high volume (>50 daily) + high repetition (>70%)
            'high': 25,      # High brands: medium volume (10-50 daily) + medium repetition (50-70%)
            'medium': 15,    # Medium brands: low-medium volume + moderate repetition
            'low': 0         # Low brands: very low volume (<10 daily) + low repetition (<50%)
        }
    
    def _load_brand_configs(self) -> Dict:
        """Load brand configurations from brand_configs directory"""
        config_path = f"brand_configs/{self.brand_filename}"
        with open(config_path, 'r') as f:
            brand_data = json.load(f)
            # Create a lookup dictionary by brand name
            brand_lookup = {}
            for brand in brand_data.get('brands', []):
                brand_lookup[brand['name']] = brand
            print(f"✓ Loaded {len(brand_lookup)} brand configurations from {config_path}")
            return brand_lookup
    
    def _load_skip_brands(self) -> set:
        """Load brands to skip from skip_brand_name.txt"""
        skip_path = "brand_configs/skip_brand_name.txt"
        try:
            with open(skip_path, 'r') as f:
                skip_brands = {line.strip() for line in f if line.strip()}
            print(f"✓ Loaded {len(skip_brands)} brands to skip from {skip_path}")
            return skip_brands
        except Exception as e:
            print(f"⚠️  Warning: Could not load skip brands from {skip_path}: {e}")
            return set()
    
    def get_brand_info(self, brand_name: str) -> Dict:
        """Get brand information from configurations"""
        if brand_name in self.brand_configs:
            return self.brand_configs[brand_name]
        else:
            # Fallback to load_brand_info function for compatibility
            return load_brand_info(brand_name, self.brand_filename) or {}
    
    def should_skip_brand(self, brand_name: str) -> bool:
        """Check if brand should be skipped"""
        return brand_name in self.skip_brands
    
    def load_brand_summaries(self, brand_name: str, data_dir: str = "processed_data_7_14") -> Dict:
        """Load summary files for a brand from all three agents"""
        summaries = {}
        # Define file paths
        file_paths = {
            'ccr': f"{data_dir}/ccr_analysis/{brand_name}/{brand_name}_ccr_summary.json",
            'promo': f"{data_dir}/promo_analysis/{brand_name}/{brand_name}_promo_summary.json", 
            'sentiment': f"{data_dir}/sentiment_analysis/{brand_name}/{brand_name}_sentiment_summary.json"
        }
        # Load each summary file
        for analysis_type, file_path in file_paths.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        summaries[analysis_type] = json.load(f)
                    print(f"✓ Loaded {analysis_type} summary for {brand_name}")
                except Exception as e:
                    print(f"⚠️  Error loading {analysis_type} summary: {e}")
                    summaries[analysis_type] = None
            else:
                print(f"❌ Missing {analysis_type} summary: {file_path}")
                summaries[analysis_type] = None
        return summaries
    
    def _normalize_score(self, value: float, max_expected: float = 10.0, power: float = 1.0) -> float:
        """Simple normalization function that scales any value to 0-100"""
        if value <= 0:
            return 0.0
        # Use power scaling for different curve shapes (1.0 = linear, >1.0 = exponential)
        normalized = min((value / max_expected) ** power, 1.0)
        return normalized * 100
    
    def calculate_ccr_risk_score(self, ccr_summary: Dict) -> Tuple[float, Dict]:
        """Enhanced CCR risk score calculation with more comprehensive signals"""
        if not ccr_summary:
            return 0.0, {'reason': 'No CCR data available'}
        total_processed = ccr_summary.get('total_processed', 0)
        if total_processed == 0:
            return 0.0, {'reason': 'No comments processed'}
        
        # Base score from CCR risk rate
        ccr_risk_rate = ccr_summary.get('ccr_risk_rate', 0) * 100  # Convert to percentage
        base_score = self._normalize_score(ccr_risk_rate, max_expected=15.0, power=1.3)
        
        # Enhanced severity scoring - include all risk levels
        risk_level_distribution = ccr_summary.get('risk_level_distribution', {})
        high_risks = risk_level_distribution.get('High', 0)
        medium_risks = risk_level_distribution.get('Medium', 0)
        low_risks = risk_level_distribution.get('Low', 0)
        
        # Weighted severity score based on risk level importance
        severity_score = (high_risks * 10 + medium_risks * 5 + low_risks * 2)
        severity_score = min(severity_score, 30)  # Cap at 30 points
        
        # Enhanced Amazon risk scoring - higher weight due to platform-specific concerns
        risk_breakdown = ccr_summary.get('risk_breakdown', {})
        amazon_risks = risk_breakdown.get('Amazon_fake_concern', 0)
        
        # Progressive Amazon risk scoring
        if amazon_risks >= 3:
            amazon_bonus = 25  # High Amazon risk
        elif amazon_risks >= 2:
            amazon_bonus = 15  # Medium Amazon risk
        elif amazon_risks >= 1:
            amazon_bonus = 8   # Low Amazon risk
        else:
            amazon_bonus = 0
        
        # Total CCR risks as additional signal
        total_ccr_risks = ccr_summary.get('total_ccr_risks', 0)
        total_risk_bonus = min(total_ccr_risks * 2, 15)  # Up to 15 points for volume of risks
        
        # Risk diversity bonus - multiple types of CCR risks indicate broader concern
        risk_types_count = len([v for v in risk_breakdown.values() if v > 0])
        diversity_bonus = min(risk_types_count * 3, 10)  # Up to 10 points for risk diversity
        
        final_score = min(base_score + severity_score + amazon_bonus + total_risk_bonus + diversity_bonus, 100)
        
        details = {
            'ccr_risk_rate_pct': ccr_risk_rate,
            'base_score': base_score,
            'high_risks': high_risks,
            'medium_risks': medium_risks,
            'low_risks': low_risks,
            'severity_score': severity_score,
            'amazon_risks': amazon_risks,
            'amazon_bonus': amazon_bonus,
            'total_ccr_risks': total_ccr_risks,
            'total_risk_bonus': total_risk_bonus,
            'risk_types_count': risk_types_count,
            'diversity_bonus': diversity_bonus,
            'final_score': final_score
        }
        
        return final_score, details
    
    def calculate_promotional_risk_score(self, promo_summary: Dict) -> Tuple[float, Dict]:
        """Simplified promotional risk score calculation"""
        if not promo_summary:
            return 0.0, {'reason': 'No promotional data available'}
        
        total_processed = promo_summary.get('total_processed', 0)
        if total_processed == 0:
            return 0.0, {'reason': 'No comments processed'}
        
        # Base score from promotional rate - reduced weight since it's not a good differentiator
        # All risk levels have similar promotional rates (~55-58%)
        promotional_rate = promo_summary.get('promotional_rate', 0) * 100
        base_score = self._normalize_score(promotional_rate, max_expected=90.0, power=1.0) * 0.6  # Reduced impact
        
        # Spam bonus: any spam is concerning
        promo_types = promo_summary.get('promo_types', {})
        spam_count = promo_types.get('Spam', 0)
        spam_rate = (spam_count / total_processed) * 100 if total_processed > 0 else 0
        spam_bonus = self._normalize_score(spam_rate, max_expected=2.0, power=1.5) * 0.3  # Up to 30 points
        
        # Content repetition bonus (key finding from analysis)
        repeated_content_rate = promo_summary.get('repeated_content_rate', 0)
        repetition_bonus = self._normalize_score(repeated_content_rate, max_expected=100.0, power=1.0) * 0.2  # Up to 20 points
        
        # High intensity promotional content
        high_intensity_rate = promo_summary.get('high_intensity_promo_rate', 0) * 100
        intensity_bonus = self._normalize_score(high_intensity_rate, max_expected=60.0, power=1.0) * 0.15  # Up to 15 points
        
        final_score = min(base_score + spam_bonus + repetition_bonus + intensity_bonus, 100)
        
        details = {
            'promotional_rate': promotional_rate,
            'base_score': base_score,
            'spam_count': spam_count,
            'spam_rate': spam_rate,
            'spam_bonus': spam_bonus,
            'repeated_content_rate': repeated_content_rate,
            'repetition_bonus': repetition_bonus,
            'high_intensity_rate': high_intensity_rate,
            'intensity_bonus': intensity_bonus,
            'final_score': final_score
        }
        
        return final_score, details
    
    def calculate_organic_negative_pct(self, brand_name: str, data_dir: str = "processed_data_7_14") -> float:
        """Calculate organic negative sentiment percentage from detailed CSV files"""
        try:
            import pandas as pd
            
            # Load promo analysis CSV (contains is_promotional flag)
            promo_csv_path = f"{data_dir}/promo_analysis/{brand_name}/{brand_name}_promo_all.csv"
            sentiment_csv_path = f"{data_dir}/sentiment_analysis/{brand_name}/{brand_name}_sentiment_analysis.csv"
            
            if not (os.path.exists(promo_csv_path) and os.path.exists(sentiment_csv_path)):
                return None
            
            # Load the CSV files
            promo_df = pd.read_csv(promo_csv_path)
            sentiment_df = pd.read_csv(sentiment_csv_path)
            # Merge on fid to get promotional status and sentiment for each comment
            merged_df = pd.merge(sentiment_df[['fid', 'sentiment']], 
                               promo_df[['fid', 'is_promotional']], 
                               on='fid', how='inner')
            
            # Filter for non-promotional comments
            organic_comments = merged_df[merged_df['is_promotional'] == False]
            
            if len(organic_comments) == 0:
                return None
            # Calculate negative percentage for organic comments
            negative_organic = len(organic_comments[organic_comments['sentiment'] == 'Negative'])
            organic_negative_pct = (negative_organic / len(organic_comments)) * 100
            return organic_negative_pct
            
        except Exception as e:
            print(f"Warning: Could not calculate organic negative % for {brand_name}: {e}")
            return None

    def calculate_sentiment_risk_score(self, sentiment_summary: Dict, brand_name: str = None, data_dir: str = "processed_data_7_14") -> Tuple[float, Dict]:
        """Simplified sentiment risk score calculation"""
        if not sentiment_summary:
            return 0.0, {'reason': 'No sentiment data available'}
        
        sentiment_dist = sentiment_summary.get('sentiment_distribution', {})
        negative_pct = sentiment_dist.get('negative_pct', 0)
        mixed_pct = sentiment_dist.get('mixed_pct', 0)
        
        # Calculate organic negative sentiment if brand name provided
        organic_negative_pct = None
        if brand_name:
            organic_negative_pct = self.calculate_organic_negative_pct(brand_name, data_dir)
        
        # Use organic negative sentiment as primary metric if available
        primary_negative_pct = organic_negative_pct if organic_negative_pct is not None else negative_pct
        # Base score from negative sentiment - adjusted based on actual patterns
        # High: ~9.9%, Critical: ~6.4%, Low: ~1.8% organic negative sentiment
        # Use moderate power scaling since High brands have highest negative sentiment
        base_score = self._normalize_score(primary_negative_pct, max_expected=12.0, power=1.2)
        # Mixed sentiment bonus (indicates conflicted opinions)
        mixed_bonus = self._normalize_score(mixed_pct, max_expected=10.0, power=1.0) * 0.2  # Up to 20 points
        
        # Negative emotions bonus
        negative_emotions_pct = sentiment_summary.get('negative_emotions_pct', 0)
        emotion_bonus = self._normalize_score(negative_emotions_pct, max_expected=20.0, power=1.0) * 0.15  # Up to 15 points
        
        # High-risk emotions bonus (Anger, Disgust, Fear, Sadness)
        high_risk_emotions = ['Anger', 'Disgust', 'Fear', 'Sadness']
        high_risk_emotion_bonus = 0
        top_emotions = sentiment_summary.get('top_emotions', [])
        total_comments = sentiment_summary.get('total_comments', 1)
        
        for emotion_data in top_emotions:
            emotion = emotion_data.get('emotion', '')
            count = emotion_data.get('count', 0)
            if emotion in high_risk_emotions:
                emotion_pct = (count / total_comments) * 100
                high_risk_emotion_bonus += self._normalize_score(emotion_pct, max_expected=5.0, power=1.2) * 0.1  # Up to 10 points per emotion
        
        high_risk_emotion_bonus = min(high_risk_emotion_bonus, 20)  # Cap total bonus
        final_score = min(base_score + mixed_bonus + emotion_bonus + high_risk_emotion_bonus, 100)
        
        details = {
            'negative_pct': negative_pct,
            'organic_negative_pct': organic_negative_pct,
            'primary_negative_pct': primary_negative_pct,
            'base_score': base_score,
            'mixed_pct': mixed_pct,
            'mixed_bonus': mixed_bonus,
            'negative_emotions_pct': negative_emotions_pct,
            'emotion_bonus': emotion_bonus,
            'high_risk_emotion_bonus': high_risk_emotion_bonus,
            'final_score': final_score
        }
        
        return final_score, details
    
    def calculate_actual_daily_average(self, brand_name: str, data_dir: str = "processed_data_7_14") -> Tuple[float, int, int]:
        """Calculate actual daily average from prefilter relevant.csv timestamp data"""
        try:
            import pandas as pd
            from datetime import datetime
            
            # Use prefilter relevant.csv file
            relevant_csv_path = f"{data_dir}/prefilter/{brand_name}/{brand_name}_comments_relevant.csv"
            
            if not os.path.exists(relevant_csv_path):
                return None, 0, 0
            
            # Load relevant comments CSV and extract timestamps
            df = pd.read_csv(relevant_csv_path)
            if 'timestamp' not in df.columns:
                return None, 0, 0
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            # Calculate date range
            min_date = df['date'].min()
            max_date = df['date'].max()
            total_days = (max_date - min_date).days + 1  # +1 to include both start and end dates
            
            # Calculate daily average
            total_comments = len(df)
            daily_avg = total_comments / total_days if total_days > 0 else 0
            
            return daily_avg, total_comments, total_days
            
        except Exception as e:
            print(f"Warning: Could not calculate actual daily average for {brand_name}: {e}")
            return None, 0, 0

    def calculate_comment_intensity_score(self, summaries: Dict, brand_name: str = None, data_dir: str = "processed_data_7_14") -> Tuple[float, Dict]:
        """Simplified comment intensity score calculation with actual daily average"""
        # Get total comments from any available summary
        total_comments = 0
        for summary in summaries.values():
            if summary and 'total_processed' in summary:
                total_comments = max(total_comments, summary['total_processed'])
        
        if total_comments == 0:
            return 0.0, {'reason': 'No comment data available'}
        
        # Calculate actual daily average from timestamp data
        actual_daily_avg, actual_total_comments, actual_days = None, 0, 0
        if brand_name:
            actual_daily_avg, actual_total_comments, actual_days = self.calculate_actual_daily_average(brand_name, data_dir)
        
        # Use actual daily average if available, otherwise estimate
        if actual_daily_avg is not None:
            estimated_daily_avg = actual_daily_avg
            total_days = actual_days
        else:
            # Fallback: estimate based on typical data collection period
            estimated_daily_avg = total_comments / 14  # Conservative estimate
            total_days = 14
        
        # Volume score: normalize daily average based on actual patterns
        # Critical: ~106 daily, High: ~22 daily, Low: ~3.5 daily
        # Use power scaling to emphasize the large differences
        volume_score = self._normalize_score(estimated_daily_avg, max_expected=120.0, power=1.5)
        # Content repetition score (key finding from analysis)
        repeated_content_rate = 0
        if summaries.get('promo') and 'repeated_content_rate' in summaries['promo']:
            repeated_content_rate = summaries['promo']['repeated_content_rate']
        
        # Normalize repetition rate based on actual patterns
        # Critical: ~75%, High: ~62%, Low: ~39%
        # This is a key differentiator, so give it more weight
        repetition_score = self._normalize_score(repeated_content_rate, max_expected=100.0, power=1.2) * 0.6  # Up to 60 points
        
        # Clustering efficiency score (indicates coordinated activity)
        avg_efficiency = 0
        efficiency_count = 0
        
        for summary in summaries.values():
            if summary and 'clustering_efficiency' in summary:
                efficiency = summary['clustering_efficiency'].get('efficiency_percentage', 0)
                avg_efficiency += efficiency
                efficiency_count += 1
        
        if efficiency_count > 0:
            avg_efficiency = avg_efficiency / efficiency_count
        
        # Normalize clustering efficiency (expect max ~50% for highly coordinated content)
        efficiency_score = self._normalize_score(avg_efficiency, max_expected=50.0, power=1.1) * 0.2  # Up to 20 points
        
        final_score = min(volume_score + repetition_score + efficiency_score, 100)
        
        details = {
            'total_comments': total_comments,
            'actual_daily_avg': round(estimated_daily_avg, 1),
            'total_days': total_days,
            'used_actual_timestamps': actual_daily_avg is not None,
            'volume_score': volume_score,
            'repeated_content_rate': repeated_content_rate,
            'repetition_score': repetition_score,
            'avg_clustering_efficiency': avg_efficiency,
            'efficiency_score': efficiency_score,
            'final_score': final_score
        }
        
        return final_score, details
    
    def calculate_integrated_risk_score(self, summaries: Dict, brand_name: str = None, data_dir: str = "processed_data_7_14") -> Dict:
        """Calculate the integrated risk score from all components"""
        # Calculate individual component scores
        ccr_score, ccr_details = self.calculate_ccr_risk_score(summaries.get('ccr'))
        promo_score, promo_details = self.calculate_promotional_risk_score(summaries.get('promo'))
        sentiment_score, sentiment_details = self.calculate_sentiment_risk_score(summaries.get('sentiment'), brand_name, data_dir)
        intensity_score, intensity_details = self.calculate_comment_intensity_score(summaries, brand_name, data_dir)
        
        # Calculate weighted final score
        final_score = (
            ccr_score * self.weights['ccr_risk'] +
            promo_score * self.weights['promotional_risk'] +
            sentiment_score * self.weights['sentiment_risk'] +
            intensity_score * self.weights['comment_intensity']
        )
        
        # Determine risk level
        if final_score >= self.risk_thresholds['critical']:
            risk_level = 'CRITICAL'
        elif final_score >= self.risk_thresholds['high']:
            risk_level = 'HIGH'
        elif final_score >= self.risk_thresholds['medium']:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'final_risk_score': round(final_score, 2),
            'risk_level': risk_level,
            'component_scores': {
                'ccr_risk': round(ccr_score, 2),
                'promotional_risk': round(promo_score, 2),
                'sentiment_risk': round(sentiment_score, 2),
                'comment_intensity': round(intensity_score, 2)
            },
            'component_details': {
                'ccr': ccr_details,
                'promotional': promo_details,
                'sentiment': sentiment_details,
                'intensity': intensity_details
            },
            'weights_used': self.weights
        }