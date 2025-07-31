#!/usr/bin/env python3
"""
Demo version of Agentic Workflow Dashboard
Privacy-safe version with selected brands only
"""

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
    page_title="Agentic Brand Risk Workflow - Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Demo brands (privacy-safe selection)
DEMO_BRANDS = [
    "Amika", "Pop Mart-Labubu", "Rhode", "Colgate", "Dior", 
    "Anua", "Oral-B", "Chanel", "NYX", "Summer Fridays", "GNC", "K18"
]

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
    .demo-notice {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
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

class AgenticWorkflowDashboardDemo:
    def __init__(self, data_dir="demo_data"):
        self.data_dir = data_dir
        self.workflow_steps = [
            "Prefilter Agent",
            "Counterfeit Detection Agent", 
            "Promotional Detection Agent",
            "Sentiment Analysis Agent",
            "Final Risk Assessment Agent"
        ]
        
    def get_available_brands(self):
        """Get demo brands only"""
        return DEMO_BRANDS
    
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

def main():
    dashboard = AgenticWorkflowDashboardDemo()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Agentic Brand Risk Workflow - Demo</h1>', 
                unsafe_allow_html=True)
    
    # Demo notice
    st.markdown("""
    <div class="demo-notice">
        <h3>🎯 Interactive Demo</h3>
        <p>This is a demonstration of our AI-powered multi-agent brand risk assessment workflow.</p>
        <p>Featuring 12 selected brands with real social media analysis data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Demo Control")
    
    # Brand selection
    available_brands = dashboard.get_available_brands()
    
    selected_brand = st.sidebar.selectbox(
        "Select Brand for Analysis:",
        available_brands,
        index=0
    )
    
    # Workflow options
    st.sidebar.markdown("## ⚙️ Demo Options")
    simulate_processing = st.sidebar.checkbox("Simulate Agent Processing", value=True)
    
    # Brand info
    st.sidebar.markdown("## 📋 Selected Brands")
    st.sidebar.markdown("**Demo includes:**")
    for brand in available_brands:
        st.sidebar.markdown(f"• {brand}")
    
    # Main workflow display
    if selected_brand:
        st.markdown(f"## 🔄 Analyzing Brand: **{selected_brand}**")
        
        # Simple demo - show final risk assessment
        data = dashboard.load_agent_data(selected_brand, 'final_risk')
        if data:
            # Get risk information
            final_risk_level = data.get('final_risk_level', 'UNKNOWN')
            calculated_risk = data.get('calculated_risk', {})
            final_risk_score = calculated_risk.get('final_risk_score', 0)
            
            # Display risk assessment
            risk_class = f"risk-{final_risk_level.lower()}"
            st.markdown(f"""
            <div class="decision-box {risk_class}">
                <h2>🎯 FINAL RISK ASSESSMENT</h2>
                <h3>Risk Level: {final_risk_level}</h3>
                <h4>Risk Score: {final_risk_score:.1f}/100</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Show component scores
            component_scores = calculated_risk.get('component_scores', {})
            if component_scores:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    ccr_score = component_scores.get('ccr_risk', 0)
                    st.markdown(f"""
                    <div class="metric-highlight">
                        <h4>{ccr_score:.1f}</h4>
                        <p>CCR Risk</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    promo_score = component_scores.get('promotional_risk', 0)
                    st.markdown(f"""
                    <div class="metric-highlight">
                        <h4>{promo_score:.1f}</h4>
                        <p>Promo Risk</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    sentiment_score = component_scores.get('sentiment_risk', 0)
                    st.markdown(f"""
                    <div class="metric-highlight">
                        <h4>{sentiment_score:.1f}</h4>
                        <p>Sentiment Risk</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    volume_score = component_scores.get('volume_risk', 0)
                    st.markdown(f"""
                    <div class="metric-highlight">
                        <h4>{volume_score:.1f}</h4>
                        <p>Volume Risk</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show LLM assessment if available
            llm_assessment = data.get('llm_assessment', {})
            if llm_assessment:
                key_factors = llm_assessment.get('key_factors', [])
                immediate_actions = llm_assessment.get('immediate_actions', [])
                
                if key_factors or immediate_actions:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if key_factors:
                            st.markdown("#### 🔑 Key Risk Factors")
                            for factor in key_factors[:3]:
                                import re
                                cleaned_factor = re.sub(r'^\d+\.\s*', '', factor.strip())
                                st.markdown(f"• {cleaned_factor}")
                    
                    with col2:
                        if immediate_actions:
                            st.markdown("#### ⚡ Immediate Actions")
                            for action in immediate_actions[:3]:
                                import re
                                cleaned_action = re.sub(r'^\d+\.\s*', '', action.strip())
                                st.markdown(f"• {cleaned_action}")
        else:
            st.warning(f"No risk assessment data available for {selected_brand}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🤖 Powered by Multi-Agent AI Pipeline | 📊 Real Social Media Data Analysis</p>
        <p>This demo showcases AI-driven brand risk assessment capabilities</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()