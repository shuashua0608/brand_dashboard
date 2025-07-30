import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import glob
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Agentic Brand Risk Workflow",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for interactive workflow styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .agent-active {
        border-color: #4CAF50;
        background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%);
        transform: scale(1.02);
    }
    .agent-completed {
        border-color: #2196F3;
        background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%);
    }
    .agent-pending {
        border-color: #ccc;
        background: #f9f9f9;
        opacity: 0.7;
    }
    .risk-critical {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
        color: #c62828;
    }
    .risk-high {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 5px solid #ff9800;
        color: #e65100;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fffde7 0%, #fff9c4 100%);
        border-left: 5px solid #ffc107;
        color: #f57f17;
    }
    .risk-low {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    .workflow-step {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 10px;
        background: #f8f9fa;
    }
    .step-number {
        background: #007bff;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 1rem;
    }
    .progress-bar {
        height: 20px;
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        transition: width 0.5s ease;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .decision-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #ff8a65;
        margin: 1rem 0;
    }
    .action-recommendation {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #26c6da;
        margin: 1rem 0;
    }
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    .typing-cursor {
        animation: blink 1s infinite;
    }
    .ai-thinking {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-style: italic;
    }
    .risk-score-display {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transform: scale(1.05);
        transition: all 0.3s ease;
    }
    .component-reveal {
        opacity: 0;
        animation: fadeInUp 0.8s ease forwards;
    }
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .sample-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        font-style: italic;
    }
    .sample-metadata {
        background: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .risk-type-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-type-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    .promo-type {
        color: #6f42c1;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .irrelevant-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6c757d;
        margin: 0.5rem 0;
        font-style: italic;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

class AgenticWorkflowDashboard:
    def __init__(self, data_dir="processed_data_7_14"):
        self.data_dir = data_dir
        self.workflow_steps = [
            "Prefilter Agent",
            "Counterfeit Detection Agent", 
            "Promotional Detection Agent",
            "Sentiment Analysis Agent",
            "Final Risk Assessment Agent"
        ]
        
    def load_brand_configs(self):
        """Load brand configurations"""
        try:
            with open('brand_configs/brand_info_new.json', 'r') as f:
                data = json.load(f)
            return {brand['name']: brand for brand in data['brands']}
        except Exception as e:
            st.error(f"Error loading brand configs: {e}")
            return {}
    
    def load_skip_brands(self):
        """Load brands to skip"""
        try:
            with open('brand_configs/skip_brand_name.txt', 'r') as f:
                return {line.strip() for line in f if line.strip()}
        except:
            return set()
    
    def get_available_brands(self):
        """Get brands with analysis data"""
        pattern = f"{self.data_dir}/final_risk_assessment/*_final_risk_assessment.json"
        files = glob.glob(pattern)
        brands = []
        skip_brands = self.load_skip_brands()
        
        for file_path in files:
            brand_name = os.path.basename(file_path).replace('_final_risk_assessment.json', '')
            if brand_name not in skip_brands:
                brands.append(brand_name)
        
        return sorted(brands)
    
    def load_agent_data(self, brand_name, agent_type):
        """Load data for specific agent"""
        file_patterns = {
            'prefilter': f"{self.data_dir}/prefilter/{brand_name}/{brand_name}_summary.json",
            'ccr': f"{self.data_dir}/ccr_analysis/{brand_name}/{brand_name}_ccr_summary.json",
            'promo': f"{self.data_dir}/promo_analysis/{brand_name}/{brand_name}_promo_summary.json",
            'sentiment': f"{self.data_dir}/sentiment_analysis/{brand_name}/{brand_name}_sentiment_summary.json",
            'final_risk': f"{self.data_dir}/final_risk_assessment/{brand_name}_final_risk_assessment.json"
        }
        
        file_path = file_patterns.get(agent_type)
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading {agent_type} data: {e}")
        return None
    
    def load_agent_csv_data(self, brand_name, agent_type):
        """Load CSV data for specific agent to show sample rows"""
        file_patterns = {
            'ccr': f"{self.data_dir}/ccr_analysis/{brand_name}/{brand_name}_ccr_all.csv",
            'promo': f"{self.data_dir}/promo_analysis/{brand_name}/{brand_name}_promo_all.csv",
            'sentiment': f"{self.data_dir}/sentiment_analysis/{brand_name}/{brand_name}_sentiment_analysis.csv"
        }
        
        file_path = file_patterns.get(agent_type)
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                return df
            except Exception as e:
                st.error(f"Error loading {agent_type} CSV data: {e}")
        return None
    
    def load_prefilter_csv_data(self, brand_name):
        """Load prefilter CSV data to show sample irrelevant comments"""
        # The correct file path for prefilter data with all comments
        file_path = f"{self.data_dir}/prefilter/{brand_name}/{brand_name}_comments_all.csv"
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Check if it has the 'relevant' column
                if 'relevant' in df.columns:
                    return df
                else:
                    st.warning(f"File {file_path} exists but doesn't have 'relevant' column")
            except Exception as e:
                st.error(f"Error loading prefilter data: {e}")
        else:
            st.warning(f"Prefilter file not found: {file_path}")
        
        return None
    
    def display_sample_rows(self, df, sample_type, brand_name, simulate=False):
        """Display sample rows with animation if simulate is True"""
        if df is None or len(df) == 0:
            return
        
        if simulate:
            # Show thinking phase
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown(f"""
            <div class="ai-thinking">
                🔍 Analyzing {sample_type} examples from the data...
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.5)
            thinking_placeholder.empty()
        
        st.markdown(f"#### 📋 Sample {sample_type} Examples")
        
        # Display each sample with animation if simulate is True
        for i, (_, row) in enumerate(df.iterrows()):
            if simulate:
                sample_placeholder = st.empty()
                sample_placeholder.markdown(f"""
                <div class="ai-thinking">
                    📄 Loading example {i+1}...
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.8)
                sample_placeholder.empty()
            
            # Create expandable section for each sample
            with st.expander(f"Example {i+1}: {sample_type}", expanded=False):
                # Display the cleaned transcript with styling
                st.markdown("**Content:**")
                content = row.get('cleaned_transcript', row.get('transcript', 'No content available'))
                if len(content) > 600:
                    content = content[:600] + "..."
                
                # Use different styling for irrelevant content
                if 'relevant' in row and row.get('relevant') == False:
                    content_class = "irrelevant-content"
                else:
                    content_class = "sample-content"
                
                st.markdown(f"""
                <div class="{content_class}">
                    {content}
                </div>
                """, unsafe_allow_html=True)
                
                # Display relevant metadata based on agent type
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    platform = row.get('social_media_channel', 'Unknown')
                    st.markdown(f"""
                    <div class="sample-metadata">
                        <strong>Platform:</strong><br>
                        {platform}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    timestamp = row.get('timestamp', 'Unknown')
                    if timestamp != 'Unknown':
                        try:
                            # Format timestamp nicely
                            dt = pd.to_datetime(timestamp)
                            formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            formatted_time = timestamp
                    else:
                        formatted_time = timestamp
                    
                    st.markdown(f"""
                    <div class="sample-metadata">
                        <strong>Timestamp:</strong><br>
                        {formatted_time}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Agent-specific information with styling
                    if 'risk_type' in row:  # CCR agent
                        risk_type = row.get('risk_type', 'None')
                        risk_level = row.get('risk_level', 'Unknown')
                        risk_class = 'risk-type-high' if risk_level == 'High' else 'risk-type-medium'
                        
                        st.markdown(f"""
                        <div class="sample-metadata">
                            <strong>Risk Type:</strong><br>
                            <span class="{risk_class}">{risk_type}</span><br>
                            <strong>Level:</strong> {risk_level}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif 'promo_type' in row:  # Promo agent
                        promo_type = row.get('promo_type', 'None')
                        intensity = row.get('promotional_intensity', 'Unknown')
                        
                        st.markdown(f"""
                        <div class="sample-metadata">
                            <strong>Promo Type:</strong><br>
                            <span class="promo-type">{promo_type}</span><br>
                            <strong>Intensity:</strong> {intensity}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif 'sentiment' in row:  # Sentiment agent
                        sentiment = row.get('sentiment', 'Unknown')
                        emotion = row.get('emotion', 'Unknown')
                        sentiment_class = 'sentiment-negative' if sentiment == 'Negative' else 'sentiment-positive'
                        
                        st.markdown(f"""
                        <div class="sample-metadata">
                            <strong>Sentiment:</strong><br>
                            <span class="{sentiment_class}">{sentiment}</span><br>
                            <strong>Emotion:</strong> {emotion}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif 'relevant' in row:  # Prefilter agent
                        relevant = row.get('relevant', 'Unknown')
                        confidence = row.get('confidence', 'Unknown')
                        relevant_class = 'sentiment-negative' if relevant == False else 'sentiment-positive'
                        relevant_text = 'Irrelevant' if relevant == False else 'Relevant'
                        
                        st.markdown(f"""
                        <div class="sample-metadata">
                            <strong>Status:</strong><br>
                            <span class="{relevant_class}">{relevant_text}</span><br>
                            <strong>Confidence:</strong> {confidence}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show reasoning if available
                if 'reasoning' in row and pd.notna(row['reasoning']):
                    st.markdown("**AI Analysis:**")
                    reasoning = str(row['reasoning'])
                    if len(reasoning) > 200:
                        reasoning = reasoning[:200] + "..."
                    st.info(reasoning)
            
            if simulate:
                time.sleep(0.5)  # Pause between samples
    
    def simulate_agent_processing(self, agent_name, duration=2):
        """Simulate agent processing with progress bar"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(duration * 10):
            progress = (i + 1) / (duration * 10)
            progress_bar.progress(progress)
            status_text.text(f"{agent_name} processing... {progress*100:.0f}%")
            time.sleep(0.1)
        
        status_text.text(f"✅ {agent_name} completed!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
    
    def simulate_enhanced_final_assessment(self, agent_name):
        """Enhanced simulation for final risk assessment with detailed steps"""
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 🤖 AI Agent Analysis in Progress...")
            
            # Step 1: Data Integration
            st.markdown("#### 📊 Step 1: Integrating Multi-Agent Signals")
            progress_bar_1 = st.progress(0)
            status_text_1 = st.empty()
            
            integration_steps = [
                "Loading counterfeit detection results...",
                "Processing promotional content analysis...", 
                "Integrating sentiment analysis data...",
                "Calculating component risk scores...",
                "Normalizing risk metrics..."
            ]
            
            for i, step in enumerate(integration_steps):
                progress = (i + 1) / len(integration_steps)
                progress_bar_1.progress(progress)
                status_text_1.text(f"🔄 {step}")
                time.sleep(0.8)
            
            status_text_1.text("✅ Data integration completed!")
            time.sleep(0.5)
            progress_bar_1.empty()
            status_text_1.empty()
            
            # Step 2: Risk Calculation
            st.markdown("#### 🧮 Step 2: Computing Risk Scores")
            progress_bar_2 = st.progress(0)
            status_text_2 = st.empty()
            
            calculation_steps = [
                "Applying risk weighting algorithms...",
                "Cross-validating signal correlations...",
                "Computing composite risk score...",
                "Determining risk level thresholds..."
            ]
            
            for i, step in enumerate(calculation_steps):
                progress = (i + 1) / len(calculation_steps)
                progress_bar_2.progress(progress)
                status_text_2.text(f"⚡ {step}")
                time.sleep(0.7)
            
            status_text_2.text("✅ Risk calculation completed!")
            time.sleep(0.5)
            progress_bar_2.empty()
            status_text_2.empty()
            
            # Step 3: LLM Analysis
            st.markdown("#### 🧠 Step 3: AI Reasoning & Recommendations")
            progress_bar_3 = st.progress(0)
            status_text_3 = st.empty()
            
            llm_steps = [
                "Initializing large language model...",
                "Analyzing risk patterns and context...",
                "Generating key risk factors...",
                "Formulating actionable recommendations...",
                "Validating business impact assessment...",
                "Finalizing comprehensive report..."
            ]
            
            for i, step in enumerate(llm_steps):
                progress = (i + 1) / len(llm_steps)
                progress_bar_3.progress(progress)
                status_text_3.text(f"🤖 {step}")
                time.sleep(0.9)
            
            status_text_3.text("✅ AI analysis completed!")
            time.sleep(0.5)
            progress_bar_3.empty()
            status_text_3.empty()
            
            # AI thinking phase
            st.markdown("""
            <div class="ai-thinking">
                🤖 AI Agent is analyzing patterns and formulating recommendations...
            </div>
            """, unsafe_allow_html=True)
            time.sleep(2)
            
            # Final completion
            st.success("🎯 Final Risk Assessment Agent completed successfully!")
            time.sleep(1)
            
            # Clear the progress container
            progress_container.empty()
    
    def display_animated_risk_results(self, final_risk_score, final_risk_level, component_scores, llm_assessment):
        """Display risk results with animated reveals"""
        
        # Animated score reveal
        st.markdown("### 📊 Risk Score Analysis")
        score_placeholder = st.empty()
        
        # Animate the score counting up
        for i in range(0, int(final_risk_score) + 1, max(1, int(final_risk_score/20))):
            score_placeholder.markdown(f"""
            <div style="text-align: center; font-size: 3rem; font-weight: bold; color: #e74c3c;">
                {i:.1f}/100
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.1)
        
        # Final score with risk level
        time.sleep(0.5)
        risk_class = f"risk-{final_risk_level.lower()}"
        risk_color = {'CRITICAL': '#c62828', 'HIGH': '#e65100', 'MEDIUM': '#f57f17', 'LOW': '#2e7d32'}.get(final_risk_level, '#666')
        
        score_placeholder.markdown(f"""
        <div class="decision-box {risk_class}" style="text-align: center; margin: 2rem 0;">
            <h1 style="color: {risk_color}; font-size: 3rem; margin: 0;">🎯 {final_risk_level} RISK</h1>
            <h2 style="color: {risk_color}; margin: 0.5rem 0;">Score: {final_risk_score:.1f}/100</h2>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)
        
        # Component scores reveal
        if component_scores:
            st.markdown("### 📈 Component Risk Breakdown")
            
            col1, col2, col3, col4 = st.columns(4)
            columns = [col1, col2, col3, col4]
            
            component_names = ['CCR Risk', 'Promotional Risk', 'Sentiment Risk', 'Volume Risk']
            component_keys = ['ccr_risk', 'promotional_risk', 'sentiment_risk', 'volume_risk']
            
            for i, (col, name, key) in enumerate(zip(columns, component_names, component_keys)):
                score = component_scores.get(key, 0)
                
                with col:
                    # Animated component score
                    component_placeholder = st.empty()
                    
                    for j in range(0, int(score) + 1, max(1, int(score/10))):
                        component_placeholder.markdown(f"""
                        <div class="metric-highlight">
                            <h4>{j:.1f}</h4>
                            <p>{name}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.15)
                
                time.sleep(0.3)  # Stagger the reveals
        
        # AI thinking phase before reasoning
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="ai-thinking">
            🤖 AI Agent is formulating detailed analysis and recommendations...
        </div>
        """, unsafe_allow_html=True)
        time.sleep(2)
        thinking_placeholder.empty()
        
        # AI Reasoning reveal
        if llm_assessment:
            self.display_animated_ai_reasoning(llm_assessment)
    
    def display_animated_ai_reasoning(self, llm_assessment):
        """Display AI reasoning with typing effect and pauses"""
        
        st.markdown("### 🧠 AI Agent Analysis Results")
        
        # Analyzing phase
        analyzing_placeholder = st.empty()
        analyzing_placeholder.markdown("""
        <div class="ai-thinking">
            🔍 Analyzing risk patterns and correlations...
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.5)
        analyzing_placeholder.empty()
        
        # Key Risk Factors with typing effect
        key_factors = llm_assessment.get('key_factors', [])
        if key_factors:
            st.markdown("#### 🔑 Key Risk Factors Identified:")
            
            # Show thinking before each factor
            factors_container = st.empty()
            
            displayed_factors = []
            for i, factor in enumerate(key_factors[:5]):
                # Show AI thinking
                factors_container.markdown(f"""
                <div class="ai-thinking">
                    🤖 Identifying risk factor {i+1}...
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.5)
                
                # Clean up factor text
                import re
                cleaned_factor = re.sub(r'^\d+\.\s*', '', factor.strip())
                displayed_factors.append(f"• {cleaned_factor}")
                
                # Update display with new factor
                factors_container.markdown('\n'.join(displayed_factors))
                time.sleep(1.2)
        
        # Pause for processing
        processing_placeholder = st.empty()
        processing_placeholder.markdown("""
        <div class="ai-thinking">
            ⚡ Formulating actionable recommendations...
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.5)
        processing_placeholder.empty()
        
        # Immediate Actions with reveal effect
        immediate_actions = llm_assessment.get('immediate_actions', [])
        if immediate_actions:
            st.markdown("#### ⚡ Recommended Immediate Actions:")
            actions_container = st.empty()
            
            displayed_actions = []
            for i, action in enumerate(immediate_actions[:5]):
                # Show AI thinking
                actions_container.markdown(f"""
                <div class="ai-thinking">
                    💡 Generating recommendation {i+1}...
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.5)
                
                # Clean up action text
                import re
                cleaned_action = re.sub(r'^\d+\.\s*', '', action.strip())
                displayed_actions.append(f"• {cleaned_action}")
                
                # Update display with new action
                actions_container.markdown('\n'.join(displayed_actions))
                time.sleep(1.2)
        
        # Business impact analysis phase
        impact_thinking = st.empty()
        impact_thinking.markdown("""
        <div class="ai-thinking">
            📊 Assessing business impact and strategic implications...
        </div>
        """, unsafe_allow_html=True)
        time.sleep(2)
        impact_thinking.empty()
        
        # Business Impact Assessment with item-by-item reveal
        business_impact = llm_assessment.get('business_impact', '')
        if business_impact:
            st.markdown("#### 💼 Business Impact Assessment:")
            
            # Parse business impact into bullet points if it's a paragraph
            # Try to split by sentences or common separators
            impact_items = self.parse_business_impact_to_items(business_impact)
            
            impact_container = st.empty()
            displayed_impacts = []
            
            for i, impact_item in enumerate(impact_items[:5]):
                # Show AI thinking
                impact_container.markdown(f"""
                <div class="ai-thinking">
                    📈 Analyzing business impact {i+1}...
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.5)
                
                displayed_impacts.append(f"• {impact_item.strip()}")
                
                # Update display with new impact item
                impact_container.markdown(f"""
                <div class="action-recommendation">
                    <h4>💼 Business Impact Assessment</h4>
                    {'<br>'.join(displayed_impacts)}
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1.2)
        
        # AI Reasoning section (moved after business impact)
        reasoning = llm_assessment.get('reasoning', '')
        if reasoning:
            # Reasoning analysis phase
            reasoning_thinking = st.empty()
            reasoning_thinking.markdown("""
            <div class="ai-thinking">
                🧠 Generating detailed AI reasoning and rationale...
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.5)
            reasoning_thinking.empty()
            
            st.markdown("#### 🧠 AI Reasoning & Rationale:")
            st.info(reasoning)
        
        # Final AI completion with confidence indicator
        time.sleep(1)
        
        # Show confidence assessment
        confidence_placeholder = st.empty()
        confidence_placeholder.markdown("""
        <div class="ai-thinking">
            🎯 Validating analysis confidence and finalizing assessment...
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.5)
        confidence_placeholder.empty()
        
        # Final completion message
        st.success("🤖 AI Risk Assessment Analysis Complete! High confidence in recommendations.")
        
        # Add a summary box
        st.markdown("""
        <div class="decision-box">
            <h4>📋 Analysis Summary</h4>
            <p><strong>Assessment Method:</strong> Multi-agent AI pipeline with LLM validation</p>
            <p><strong>Data Integration:</strong> Counterfeit detection, promotional analysis, sentiment analysis</p>
            <p><strong>Confidence Level:</strong> High (validated across multiple risk dimensions)</p>
            <p><strong>Recommendation Status:</strong> Ready for implementation</p>
        </div>
        """, unsafe_allow_html=True)
    
    def parse_business_impact_to_items(self, business_impact):
        """Parse business impact text into individual items for better display"""
        if not business_impact:
            return []
        
        # Try to split by common patterns
        import re
        
        # First, try to split by numbered lists (1., 2., etc.)
        numbered_items = re.split(r'\d+\.\s*', business_impact)
        if len(numbered_items) > 2:  # If we found numbered items
            return [item.strip() for item in numbered_items[1:] if item.strip()]
        
        # Try to split by bullet points or dashes
        bullet_items = re.split(r'[•\-\*]\s*', business_impact)
        if len(bullet_items) > 2:
            return [item.strip() for item in bullet_items[1:] if item.strip()]
        
        # Try to split by sentences ending with periods
        sentences = re.split(r'\.\s+', business_impact)
        if len(sentences) > 1:
            return [sentence.strip() + '.' for sentence in sentences if sentence.strip()]
        
        # Try to split by semicolons or commas for long lists
        if ';' in business_impact:
            return [item.strip() for item in business_impact.split(';') if item.strip()]
        
        # If no clear structure, split by length (for very long paragraphs)
        if len(business_impact) > 200:
            words = business_impact.split()
            chunks = []
            current_chunk = []
            
            for word in words:
                current_chunk.append(word)
                if len(' '.join(current_chunk)) > 80:  # Roughly 80 chars per item
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
        
        # Fallback: return as single item
        return [business_impact]
    
    def filter_by_language(self, df, sample_size=2):
        """Filter dataframe to prioritize English content, fallback to random selection"""
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        # First, try to get English content
        if 'text_language' in df.columns:
            # Filter for English content
            english_content = df[df['text_language'] == 'en']
            
            if len(english_content) >= sample_size:
                # If we have enough English content, sample from it randomly
                return english_content.sample(n=sample_size, random_state=42)
            elif len(english_content) > 0:
                # If we have some English content but not enough, take all English + random others
                remaining_needed = sample_size - len(english_content)
                non_english = df[df['text_language'] != 'en']
                
                if len(non_english) > 0:
                    additional_samples = non_english.sample(n=min(remaining_needed, len(non_english)), random_state=42)
                    return pd.concat([english_content, additional_samples]).reset_index(drop=True)
                else:
                    return english_content.reset_index(drop=True)
        
        # Fallback: no text_language column or no English content found
        # Return random sample
        if len(df) >= sample_size:
            return df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            return df.reset_index(drop=True)
    
    def display_static_risk_results(self, final_risk_score, final_risk_level, component_scores, llm_assessment):
        """Display risk results without animation (for non-simulate mode)"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-highlight">
                <h4>{final_risk_score:.1f}</h4>
                <p>Final Risk Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ccr_score = component_scores.get('ccr_risk', 0)
            st.markdown(f"""
            <div class="metric-highlight">
                <h4>{ccr_score:.1f}</h4>
                <p>CCR Risk Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            promo_score = component_scores.get('promotional_risk', 0)
            st.markdown(f"""
            <div class="metric-highlight">
                <h4>{promo_score:.1f}</h4>
                <p>Promotional Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            sentiment_score = component_scores.get('sentiment_risk', 0)
            st.markdown(f"""
            <div class="metric-highlight">
                <h4>{sentiment_score:.1f}</h4>
                <p>Sentiment Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk level visualization
        risk_class = f"risk-{final_risk_level.lower()}"
        st.markdown(f"""
        <div class="decision-box {risk_class}">
            <h2>🎯 FINAL RISK ASSESSMENT</h2>
            <h3>Risk Level: {final_risk_level}</h3>
            <h4>Risk Score: {final_risk_score:.1f}/100</h4>
        </div>
        """, unsafe_allow_html=True)
    
    def display_prefilter_agent(self, brand_name, simulate=False):
        """Display prefilter agent workflow"""
        st.markdown("""
        <div class="agent-card agent-active">
            <h3>🔍 Step 1: Prefilter Agent</h3>
            <p><strong>Task:</strong> Filter relevant brand mentions and cluster similar comments</p>
        </div>
        """, unsafe_allow_html=True)
        
        if simulate:
            self.simulate_agent_processing("Prefilter Agent", 2)
        
        data = self.load_agent_data(brand_name, 'prefilter')
        if data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{data.get('total_comments', 0)}</h4>
                    <p>Total Comments</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{data.get('relevant_comments', 0)}</h4>
                    <p>Relevant Comments</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                relevance_rate = data.get('relevance_rate', 0)
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{relevance_rate:.1f}%</h4>
                    <p>Relevance Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                clustering_info = data.get('clustering_info', {})
                llm_saved = clustering_info.get('llm_calls_saved', 0)
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{llm_saved}</h4>
                    <p>LLM Calls Saved</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="decision-box">
                <h4>🎯 Prefilter Decision</h4>
                <p><strong>Filtered:</strong> {data.get('relevant_comments', 0)} relevant comments from {data.get('total_comments', 0)} total</p>
                <p><strong>Efficiency:</strong> Saved {llm_saved} LLM calls through clustering</p>
                <p><strong>Next Step:</strong> Pass filtered comments to specialized risk detection agents</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample irrelevant comments that were filtered out
            prefilter_df = self.load_prefilter_csv_data(brand_name)
            if prefilter_df is not None:
                # Filter for irrelevant comments
                irrelevant_comments = prefilter_df[prefilter_df['relevant'] == False]
                
                if len(irrelevant_comments) > 0:
                    # Get unique irrelevant comments (remove duplicates based on content)
                    unique_irrelevant = self.get_unique_comments(irrelevant_comments)
                    
                    # Apply language filter for better readability (prioritize English)
                    sample_irrelevant = self.filter_by_language(unique_irrelevant, 2)
                    if len(sample_irrelevant) > 0:
                        self.display_sample_rows(sample_irrelevant, "Filtered Out (Irrelevant)", brand_name, simulate)
                    else:
                        st.info("Found irrelevant comments but none in English to display")
                else:
                    st.info("No irrelevant comments found (all comments were relevant)")
            else:
                st.warning("No prefilter data available")
    
    def display_ccr_agent(self, brand_name, simulate=False):
        """Display counterfeit detection agent workflow"""
        st.markdown("""
        <div class="agent-card agent-active">
            <h3>⚠️ Step 2: Counterfeit Detection Agent</h3>
            <p><strong>Task:</strong> Identify counterfeit product discussions and authenticity concerns</p>
        </div>
        """, unsafe_allow_html=True)
        
        if simulate:
            self.simulate_agent_processing("Counterfeit Detection Agent", 3)
        
        data = self.load_agent_data(brand_name, 'ccr')
        if data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{data.get('total_processed', 0)}</h4>
                    <p>Comments Analyzed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{data.get('total_ccr_risks', 0)}</h4>
                    <p>CCR Risks Found</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                risk_rate = data.get('ccr_risk_rate', 0) * 100
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{risk_rate:.2f}%</h4>
                    <p>Risk Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                risk_breakdown = data.get('risk_breakdown', {})
                amazon_risks = risk_breakdown.get('Amazon_fake_concern', 0)
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{amazon_risks}</h4>
                    <p>Amazon CCR Issues</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk level distribution
            risk_levels = data.get('risk_level_distribution', {})
            if risk_levels:
                st.markdown("#### Risk Level Distribution")
                risk_df = pd.DataFrame(list(risk_levels.items()), columns=['Level', 'Count'])
                fig = px.bar(risk_df, x='Level', y='Count', 
                           color='Level', 
                           color_discrete_map={'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            <div class="decision-box">
                <h4>🛡️ CCR Agent Decision</h4>
                <p><strong>Risk Assessment:</strong> {data.get('total_ccr_risks', 0)} counterfeit risks detected</p>
                <p><strong>Amazon Issues:</strong> {amazon_risks} Amazon-specific concerns (highest priority)</p>
                <p><strong>Risk Types:</strong> {', '.join(risk_breakdown.keys()) if risk_breakdown else 'None'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample high-risk CCR examples
            if data.get('total_ccr_risks', 0) > 0:
                ccr_df = self.load_agent_csv_data(brand_name, 'ccr')
                if ccr_df is not None:
                    # Filter for high-risk CCR cases
                    high_risk_ccr = ccr_df[
                        (ccr_df['counterfeit_risk'] == True) & 
                        (ccr_df['risk_level'].isin(['High', 'Critical']))
                    ]
                    
                    # Apply language filter
                    sample_ccr = self.filter_by_language(high_risk_ccr, 2)
                    
                    if len(sample_ccr) > 0:
                        self.display_sample_rows(sample_ccr, "High-Risk CCR", brand_name, simulate)
        else:
            st.warning("No CCR data available")
    
    def display_promo_agent(self, brand_name, simulate=False):
        """Display promotional detection agent workflow"""
        st.markdown("""
        <div class="agent-card agent-active">
            <h3>📢 Step 3: Promotional Detection Agent</h3>
            <p><strong>Task:</strong> Identify promotional content, spam, and inauthentic marketing</p>
        </div>
        """, unsafe_allow_html=True)
        
        if simulate:
            self.simulate_agent_processing("Promotional Detection Agent", 3)
        
        data = self.load_agent_data(brand_name, 'promo')
        if data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{data.get('total_processed', 0)}</h4>
                    <p>Comments Analyzed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{data.get('promotional_count', 0)}</h4>
                    <p>Promotional Content</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                promo_rate = data.get('promotional_rate', 0) * 100
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{promo_rate:.1f}%</h4>
                    <p>Promotional Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                high_intensity = data.get('high_intensity_promo_count', 0)
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{high_intensity}</h4>
                    <p>High Intensity Promos</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Promotional types
            promo_types = data.get('promo_types', {})
            if promo_types:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Promotional Content Types")
                    promo_df = pd.DataFrame(list(promo_types.items()), columns=['Type', 'Count'])
                    fig = px.pie(promo_df, values='Count', names='Type', 
                               title="Promotional Content Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    platform_dist = data.get('platform_distribution', {})
                    if platform_dist:
                        st.markdown("#### Platform Distribution")
                        platform_df = pd.DataFrame(list(platform_dist.items()), columns=['Platform', 'Count'])
                        fig = px.bar(platform_df, x='Platform', y='Count', 
                                   color='Count', color_continuous_scale='blues')
                        st.plotly_chart(fig, use_container_width=True)
            
            risk_indicators = data.get('risk_indicators', {})
            spam_detected = risk_indicators.get('spam_detected', False)
            high_duplicate = risk_indicators.get('high_duplicate_rate', False)
            
            st.markdown(f"""
            <div class="decision-box">
                <h4>📊 Promotional Agent Decision</h4>
                <p><strong>Promotional Rate:</strong> {promo_rate:.1f}% of comments are promotional</p>
                <p><strong>Spam Detection:</strong> {'⚠️ Spam detected' if spam_detected else '✅ No spam detected'}</p>
                <p><strong>Duplicate Content:</strong> {'⚠️ High duplicate rate' if high_duplicate else '✅ Normal duplicate rate'}</p>
                <p><strong>Risk Level:</strong> {'HIGH' if spam_detected or high_duplicate else 'MEDIUM' if promo_rate > 50 else 'LOW'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample promotional content examples
            if data.get('promotional_count', 0) > 0:
                promo_df = self.load_agent_csv_data(brand_name, 'promo')
                if promo_df is not None:
                    # Filter for promotional content, prioritize high intensity
                    promotional_content = promo_df[promo_df['is_promotional'] == True]
                    
                    # Try to get high intensity first, then any promotional
                    if 'promotional_intensity' in promotional_content.columns:
                        high_intensity = promotional_content[promotional_content['promotional_intensity'] == 'High']
                        if len(high_intensity) == 0:
                            candidate_promo = promotional_content
                        else:
                            candidate_promo = high_intensity
                    else:
                        candidate_promo = promotional_content
                    
                    # Apply language filter
                    sample_promo = self.filter_by_language(candidate_promo, 2)
                    
                    if len(sample_promo) > 0:
                        self.display_sample_rows(sample_promo, "Promotional Content", brand_name, simulate)
        else:
            st.warning("No promotional data available")
    
    def display_sentiment_agent(self, brand_name, simulate=False):
        """Display sentiment analysis agent workflow"""
        st.markdown("""
        <div class="agent-card agent-active">
            <h3>😊 Step 4: Sentiment Analysis Agent</h3>
            <p><strong>Task:</strong> Analyze brand sentiment and emotional context</p>
        </div>
        """, unsafe_allow_html=True)
        
        if simulate:
            self.simulate_agent_processing("Sentiment Analysis Agent", 2)
        
        data = self.load_agent_data(brand_name, 'sentiment')
        if data:
            col1, col2, col3, col4 = st.columns(4)
            
            sentiment_dist = data.get('sentiment_distribution', {})
            negative_pct = sentiment_dist.get('negative_pct', 0)
            positive_pct = sentiment_dist.get('positive_pct', 0)
            mixed_pct = sentiment_dist.get('mixed_pct', 0)
            neutral_pct = sentiment_dist.get('neutral_pct', 0)
            
            with col1:
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{positive_pct:.1f}%</h4>
                    <p>Positive Sentiment</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{negative_pct:.1f}%</h4>
                    <p>Negative Sentiment</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{mixed_pct:.1f}%</h4>
                    <p>Mixed Sentiment</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                negative_emotions_pct = data.get('negative_emotions_pct', 0)
                st.markdown(f"""
                <div class="metric-highlight">
                    <h4>{negative_emotions_pct:.1f}%</h4>
                    <p>Negative Emotions</p>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment pie chart
                sentiment_data = {
                    'Positive': positive_pct,
                    'Negative': negative_pct,
                    'Mixed': mixed_pct,
                    'Neutral': neutral_pct
                }
                fig = px.pie(values=list(sentiment_data.values()), 
                           names=list(sentiment_data.keys()),
                           title="Sentiment Distribution",
                           color_discrete_sequence=['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top emotions
                top_emotions = data.get('top_emotions', [])
                if top_emotions:
                    emotions_df = pd.DataFrame(top_emotions)
                    fig = px.bar(emotions_df, x='count', y='emotion', 
                               orientation='h', title="Top Emotions",
                               color='count', color_continuous_scale='viridis')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Determine sentiment risk
            sentiment_risk = "HIGH" if negative_pct > 10 or negative_emotions_pct > 15 else "MEDIUM" if negative_pct > 5 else "LOW"
            
            st.markdown(f"""
            <div class="decision-box">
                <h4>💭 Sentiment Agent Decision</h4>
                <p><strong>Overall Sentiment:</strong> {positive_pct:.1f}% positive, {negative_pct:.1f}% negative</p>
                <p><strong>Emotional Risk:</strong> {negative_emotions_pct:.1f}% negative emotions detected</p>
                <p><strong>Sentiment Risk Level:</strong> {sentiment_risk}</p>
                <p><strong>Key Insight:</strong> {'Concerning negative sentiment' if negative_pct > 10 else 'Generally positive brand perception'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample negative sentiment examples
            sentiment_df = self.load_agent_csv_data(brand_name, 'sentiment')
            if sentiment_df is not None:
                # Filter for negative sentiment and negative emotions
                negative_sentiment = sentiment_df[
                    (sentiment_df['sentiment'] == 'Negative') | 
                    (sentiment_df['emotion'].isin(['Anger', 'Sadness', 'Fear', 'Disgust']))
                ]
                
                if len(negative_sentiment) > 0:
                    # Apply language filter
                    sample_negative = self.filter_by_language(negative_sentiment, 2)
                    if len(sample_negative) > 0:
                        self.display_sample_rows(sample_negative, "Negative Sentiment", brand_name, simulate)
                elif negative_pct > 0:  # If no strong negative, show any negative
                    any_negative = sentiment_df[sentiment_df['sentiment'] == 'Negative']
                    sample_negative = self.filter_by_language(any_negative, 2)
                    if len(sample_negative) > 0:
                        self.display_sample_rows(sample_negative, "Negative Sentiment", brand_name, simulate)
        else:
            st.warning("No sentiment data available")
    
    def display_final_risk_agent(self, brand_name, simulate=False):
        """Display final risk assessment agent workflow with enhanced interactivity"""
        st.markdown("""
        <div class="agent-card agent-active">
            <h3>🎯 Step 5: Final Risk Assessment Agent</h3>
            <p><strong>Task:</strong> Integrate all signals and provide comprehensive risk assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        if simulate:
            self.simulate_enhanced_final_assessment("Final Risk Assessment Agent")
        
        data = self.load_agent_data(brand_name, 'final_risk')
        if data:
            # Get calculated risk and LLM assessment
            calculated_risk = data.get('calculated_risk', {})
            llm_assessment = data.get('llm_assessment', {})
            final_risk_level = data.get('final_risk_level', 'UNKNOWN')
            final_risk_score = calculated_risk.get('final_risk_score', 0)
            
            # Component scores
            component_scores = calculated_risk.get('component_scores', {})
            
            # Enhanced animated reveal of results
            if simulate:
                self.display_animated_risk_results(final_risk_score, final_risk_level, component_scores, llm_assessment)
            else:
                self.display_static_risk_results(final_risk_score, final_risk_level, component_scores, llm_assessment)
            
            # Component breakdown chart (only for static display)
            if not simulate and component_scores:
                st.markdown("#### Risk Component Breakdown")
                components_df = pd.DataFrame(list(component_scores.items()), 
                                           columns=['Component', 'Score'])
                fig = px.bar(components_df, x='Component', y='Score',
                           title="Risk Component Scores",
                           color='Score', color_continuous_scale='reds')
                st.plotly_chart(fig, use_container_width=True)
            
            # LLM Assessment Details (only for static display)
            if not simulate and llm_assessment:
                key_factors = llm_assessment.get('key_factors', [])
                reasoning = llm_assessment.get('reasoning', '')
                immediate_actions = llm_assessment.get('immediate_actions', [])
                business_impact = llm_assessment.get('business_impact', '')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if key_factors:
                        st.markdown("#### 🔑 Key Risk Factors")
                        for factor in key_factors[:5]:
                            # Remove any existing numbering and clean up the factor text
                            cleaned_factor = factor.strip()
                            # Remove leading numbers and dots/periods
                            import re
                            cleaned_factor = re.sub(r'^\d+\.\s*', '', cleaned_factor)
                            st.markdown(f"• {cleaned_factor}")
                
                with col2:
                    if immediate_actions:
                        st.markdown("#### ⚡ Immediate Actions")
                        for action in immediate_actions[:5]:
                            # Remove any existing numbering and clean up the action text
                            cleaned_action = action.strip()
                            # Remove leading numbers and dots/periods
                            import re
                            cleaned_action = re.sub(r'^\d+\.\s*', '', cleaned_action)
                            st.markdown(f"• {cleaned_action}")
                
                # Business Impact Assessment (moved before reasoning)
                if business_impact:
                    st.markdown("#### 💼 Business Impact Assessment")
                    impact_items = self.parse_business_impact_to_items(business_impact)
                    
                    if len(impact_items) > 1:
                        # Display as bullet points if we parsed multiple items
                        for item in impact_items[:5]:
                            st.markdown(f"• {item}")
                    else:
                        # Display as a single block if it's just one item
                        st.markdown(f"""
                        <div class="action-recommendation">
                            <p>{business_impact}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if reasoning:
                    st.markdown("#### 🧠 AI Reasoning")
                    st.info(reasoning)
            
            # Data sources
            data_sources = data.get('data_sources', [])
            st.markdown(f"""
            <div class="decision-box">
                <h4>📊 Integration Summary</h4>
                <p><strong>Data Sources:</strong> {', '.join(data_sources)}</p>
                <p><strong>Assessment Method:</strong> Multi-agent signal integration with LLM validation</p>
                <p><strong>Confidence:</strong> {'High' if len(data_sources) >= 3 else 'Medium'}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No final risk assessment data available")

def main():
    dashboard = AgenticWorkflowDashboard()
    
    # Initialize session state for agent execution tracking
    if 'executed_agents' not in st.session_state:
        st.session_state.executed_agents = set()
    if 'current_brand' not in st.session_state:
        st.session_state.current_brand = None
    if 'active_agent' not in st.session_state:
        st.session_state.active_agent = None
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Agentic Brand Risk Workflow</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Interactive demonstration of our AI agent pipeline for brand risk assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Workflow Control")
    
    # Brand selection
    available_brands = dashboard.get_available_brands()
    if not available_brands:
        st.error("No brands with analysis data found in processed_data_7_14/")
        return
    
    selected_brand = st.sidebar.selectbox(
        "Select Brand for Analysis:",
        available_brands,
        index=0
    )
    
    # Reset executed agents if brand changes
    if st.session_state.current_brand != selected_brand:
        st.session_state.executed_agents = set()
        st.session_state.current_brand = selected_brand
        st.session_state.active_agent = None
    
    # Workflow options
    st.sidebar.markdown("## ⚙️ Workflow Options")
    simulate_processing = st.sidebar.checkbox("Simulate Agent Processing", value=True)
    
    # Reset workflow button
    if st.sidebar.button("🔄 Reset Workflow"):
        st.session_state.executed_agents = set()
        st.session_state.active_agent = None
        st.rerun()
    
    # Brand info
    brand_configs = dashboard.load_brand_configs()
    if selected_brand in brand_configs:
        brand_info = brand_configs[selected_brand]
        st.sidebar.markdown("## 📋 Brand Info")
        st.sidebar.markdown(f"**Category:** {brand_info.get('category', 'Unknown')}")
        st.sidebar.markdown(f"**GST_TT Risk:** {brand_info.get('risk_level', 'Unknown')}")
        
        # Add brand description
        description = brand_info.get('description', 'No description available')
        st.sidebar.markdown("**Description:**")
        st.sidebar.markdown(f"*{description}*")
    
    # Workflow status
    st.sidebar.markdown("## 📊 Workflow Status")
    agent_names = ["prefilter", "ccr", "promo", "sentiment", "final_risk"]
    agent_labels = ["🔍 Prefilter", "⚠️ CCR", "📢 Promo", "😊 Sentiment", "🎯 Final"]
    
    for agent, label in zip(agent_names, agent_labels):
        status = "✅" if agent in st.session_state.executed_agents else "⏳"
        st.sidebar.markdown(f"{status} {label}")
    
    # Main workflow display
    if selected_brand:
        st.markdown(f"## 🔄 Analyzing Brand: **{selected_brand}**")
        
        # Agent execution buttons and results
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Step 1: Prefilter Agent
        with col1:
            if st.button("🔍 Run Prefilter Agent", key="prefilter_btn"):
                with st.spinner("Running Prefilter Agent..."):
                    if simulate_processing:
                        time.sleep(2)
                    st.session_state.executed_agents.add("prefilter")
                    st.session_state.active_agent = "prefilter"
                    st.rerun()
        
        # Step 2: CCR Agent
        with col2:
            ccr_disabled = "prefilter" not in st.session_state.executed_agents
            if st.button("⚠️ Run CCR Agent", key="ccr_btn", disabled=ccr_disabled):
                with st.spinner("Running Counterfeit Detection Agent..."):
                    if simulate_processing:
                        time.sleep(3)
                    st.session_state.executed_agents.add("ccr")
                    st.session_state.active_agent = "ccr"
                    st.rerun()
        
        # Step 3: Promo Agent
        with col3:
            promo_disabled = "prefilter" not in st.session_state.executed_agents
            if st.button("📢 Run Promo Agent", key="promo_btn", disabled=promo_disabled):
                with st.spinner("Running Promotional Detection Agent..."):
                    if simulate_processing:
                        time.sleep(3)
                    st.session_state.executed_agents.add("promo")
                    st.session_state.active_agent = "promo"
                    st.rerun()
        
        # Step 4: Sentiment Agent
        with col4:
            sentiment_disabled = "prefilter" not in st.session_state.executed_agents
            if st.button("😊 Run Sentiment Agent", key="sentiment_btn", disabled=sentiment_disabled):
                with st.spinner("Running Sentiment Analysis Agent..."):
                    if simulate_processing:
                        time.sleep(2)
                    st.session_state.executed_agents.add("sentiment")
                    st.session_state.active_agent = "sentiment"
                    st.rerun()
        
        # Step 5: Final Risk Agent
        with col5:
            final_disabled = len(st.session_state.executed_agents) < 4
            if st.button("🎯 Run Final Assessment", key="final_btn", disabled=final_disabled):
                with st.spinner("Running Final Risk Assessment Agent..."):
                    if simulate_processing:
                        time.sleep(4)
                    st.session_state.executed_agents.add("final_risk")
                    st.session_state.active_agent = "final_risk"
                    st.rerun()
        
        st.markdown("---")
        
        # Display results only for the currently active agent
        if st.session_state.active_agent == "prefilter":
            dashboard.display_prefilter_agent(selected_brand, False)
        elif st.session_state.active_agent == "ccr":
            dashboard.display_ccr_agent(selected_brand, False)
        elif st.session_state.active_agent == "promo":
            dashboard.display_promo_agent(selected_brand, False)
        elif st.session_state.active_agent == "sentiment":
            dashboard.display_sentiment_agent(selected_brand, False)
        elif st.session_state.active_agent == "final_risk":
            dashboard.display_final_risk_agent(selected_brand, False)
        else:
            # Show instructions when no agent is active
            st.info("👆 Click on any agent button above to run the analysis and see results")
        
        # Show workflow completion message
        if len(st.session_state.executed_agents) == 5:
            if st.session_state.active_agent == "final_risk":
                st.success("🎉 Workflow completed! All agents have finished their analysis.")
        elif len(st.session_state.executed_agents) > 0:
            remaining = 5 - len(st.session_state.executed_agents)
            st.info(f"⏳ Workflow in progress... {remaining} agents remaining")

if __name__ == "__main__":
    main()