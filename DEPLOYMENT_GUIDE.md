# 🚀 Demo Website Deployment Guide

## 📋 Repository Status
✅ **Repository**: `shuashua0608/brand_dashboard`  
✅ **Demo Files**: All ready for deployment  
✅ **Privacy**: Only demo data included (12 selected brands)

## 🎯 Deployment Options

### Option 1: Streamlit Cloud (Recommended)
**Easiest and fastest deployment**

1. **Go to**: https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Repository**: `shuashua0608/brand_dashboard`
5. **Branch**: `main`
6. **Main file path**: `agentic_workflow_dashboard_demo.py`
7. **App URL**: Choose your custom URL (e.g., `brand-risk-demo`)
8. **Click "Deploy!"**

**Your demo will be live at**: `https://[your-app-name].streamlit.app`

### Option 2: Railway
**Great for custom domains**

1. **Go to**: https://railway.app/
2. **Sign in** with GitHub
3. **Click "New Project" → "Deploy from GitHub repo"**
4. **Select**: `shuashua0608/brand_dashboard`
5. **Environment Variables**: None needed
6. **Build Command**: `pip install -r requirements_demo.txt`
7. **Start Command**: `streamlit run agentic_workflow_dashboard_demo.py --server.port=$PORT --server.address=0.0.0.0`

### Option 3: Heroku
**Traditional cloud platform**

1. **Install Heroku CLI**
2. **Login**: `heroku login`
3. **Create app**: `heroku create your-brand-demo`
4. **Set buildpack**: `heroku buildpacks:set heroku/python`
5. **Deploy**: `git push heroku main`

**Note**: Use `Procfile_demo` as your Procfile

## 📁 Key Files for Deployment

```
├── agentic_workflow_dashboard_demo.py  # Main demo app
├── requirements_demo.txt               # Dependencies
├── Procfile_demo                      # Heroku/Railway config
├── README_demo.md                     # Demo documentation
└── demo_data/                         # Privacy-safe demo data
    ├── final_risk_assessment/         # Risk scores
    ├── ccr_analysis/                  # Counterfeit data
    ├── promo_analysis/                # Promotional data
    ├── sentiment_analysis/            # Sentiment data
    └── prefilter/                     # Filtered data
```

## 🔧 Configuration Details

### Dependencies (requirements_demo.txt)
```
streamlit>=1.28.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
```

### Start Command
```bash
streamlit run agentic_workflow_dashboard_demo.py --server.port=$PORT --server.address=0.0.0.0
```

## 🎯 Demo Features

✅ **12 Demo Brands**: Amika, Pop Mart-Labubu, Rhode, Colgate, Dior, Anua, Oral-B, Chanel, NYX, Summer Fridays, GNC, K18  
✅ **Interactive Dashboard**: Brand selection and risk visualization  
✅ **AI Risk Assessment**: Multi-agent analysis results  
✅ **Component Scores**: CCR, Promotional, Sentiment, Volume risks  
✅ **Professional UI**: Modern design with animations  

## 🛡️ Privacy & Security

✅ **No sensitive data**: Only processed, anonymized results  
✅ **Selected brands only**: 12 brands instead of full dataset  
✅ **No raw content**: Only analysis summaries  
✅ **No API keys**: Static demo data only  

## 🚀 Quick Start (Streamlit Cloud)

1. **Visit**: https://share.streamlit.io/
2. **New app** → Repository: `shuashua0608/brand_dashboard`
3. **Main file**: `agentic_workflow_dashboard_demo.py`
4. **Deploy** → Your demo is live! 🎉

## 📞 Support

If you encounter any issues:
1. Check the deployment logs
2. Verify all files are in the repository
3. Ensure requirements.txt is correct
4. Test locally first: `streamlit run agentic_workflow_dashboard_demo.py`

---

**🎯 Your AI-powered brand risk demo is ready to impress!**