import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Student Risk Assessment System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* --- Base styles for both themes --- */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card, .profile-section, .stMetric {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .recommendation-item {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .risk-high, .risk-low, .risk-medium {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        margin-right: 18px !important;
        padding: 8px 16px;
        border-radius: 6px;
        transition: all 0.2s ease;
    }

    /* --- NEW Light theme --- */
    @media (prefers-color-scheme: light), [data-theme="light"], .stApp[data-theme="light"] {
        .stApp, .block-container {
            background-color: #f9fbfd !important; /* very light grey-blue */
            color: #212121 !important;
        }
        /* Headings */
        .main-header {
            color: #0d47a1 !important; /* Deep Blue */
        }

        /* Risk cards */
        .risk-high {
            background: #fdecea !important;
            border-left: 4px solid #d32f2f !important;
            color: #b71c1c !important;
        }
        .risk-low {
            background: #edf7ed !important;
            border-left: 4px solid #2e7d32 !important;
            color: #1b5e20 !important;
        }
        .risk-medium {
            background: #fff8e1 !important;
            border-left: 4px solid #ef6c00 !important;
            color: #e65100 !important;
        }

        /* Metrics / profile cards */
        .metric-card, .profile-section, .stMetric {
            background-color: #ffffff !important; /* Pure white cards */
            border: 1px solid #e3f2fd !important; /* Light blue border */
            color: #212121 !important;
            box-shadow: 0 2px 8px rgba(33, 150, 243, 0.08) !important; /* Blue-tinted subtle shadow */
        }

        /* Recommendations */
        .recommendation-item {
            background: #e3f2fd !important; /* Light blue */
            border-left: 3px solid #1976d2 !important;
            color: #0d47a1 !important;
        }

        /* Main background */
        .stApp, .block-container {
            background-color: #f9fbfd !important; /* Very light grey-blue */
            color: #212121 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            color: #212121 !important;
            border-right: 1px solid #e0e0e0 !important;
            box-shadow: 2px 0 6px rgba(0,0,0,0.05);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] button {
            color: #424242 !important;
            background-color: #f5f5f5 !important;
            border: 1px solid #e0e0e0 !important;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #ffffff !important;
            background-color: #1976d2 !important;
            border: 1px solid #1976d2 !important;
        }
    }
    /* --- Dark theme --- */
    @media (prefers-color-scheme: dark), [data-theme="dark"], .stApp[data-theme="dark"] {
        .stApp, .block-container {
            background-color: #121212 !important; /* true dark */
            color: #e0e0e0 !important;
        }
        .main-header {
            color: #90caf9 !important; /* Blue 200 */
        }
        .risk-high {
            background: linear-gradient(135deg, #2d1b1b, #3d2525) !important;
            border-left: 4px solid #ef5350 !important; /* Red 400 */
            color: #ef9a9a !important; /* Red 200 */
        }
        .risk-low {
            background: linear-gradient(135deg, #1b2d1b, #253d25) !important;
            border-left: 4px solid #66bb6a !important; /* Green 400 */
            color: #a5d6a7 !important; /* Green 200 */
        }
        .risk-medium {
            background: linear-gradient(135deg, #332b1b, #463a25) !important;
            border-left: 4px solid #ffa726 !important; /* Orange 400 */
            color: #ffcc80 !important; /* Orange 200 */
        }
        .metric-card, .profile-section, .stMetric {
            background-color: #2d3748 !important; /* Gray 700 */
            border: 1px solid #4a5568 !important;
            color: #e2e8f0 !important; /* Gray 200 */
        }
        .recommendation-item {
            background: linear-gradient(135deg, #1a365d, #2c5282) !important;
            border-left: 3px solid #64b5f6 !important; /* Blue 300 */
            color: #bbdefb !important; /* Blue 100 */
        }
        .stApp, .block-container {
            background-color: #121212 !important; /* true dark bg */
            color: #e0e0e0 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #1e1e1e !important;
            color: #e0e0e0 !important;
            border-right: 1px solid #333333 !important;
        }
        .stTabs [data-baseweb="tab-list"] button {
            color: #b0bec5 !important; /* Grey 400 */
            background-color: transparent !important;
            border: 1px solid #4a5568 !important;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #90caf9 !important; /* Blue 200 */
            background-color: #1e3a5f !important; /* Blue Dark */
            border: 1px solid #64b5f6 !important;
        }
    }

    /* --- Hover and focus (works in both themes) --- */
    .metric-card:hover, .profile-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab-list"] button:focus {
        outline: 2px solid #2196f3;
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_model_components():
    """Load the trained model and all preprocessing components"""
    try:
        # Load the trained model
        model = joblib.load('models/student_risk_model.pkl')
        
        # Load the scaler
        scaler = joblib.load('models/feature_scaler.pkl')
        
        # Load label encoders
        with open('models/label_encoders.pkl', 'rb') as f:
            le_dict = pickle.load(f)
        
        # Load metadata
        with open('models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return model, scaler, le_dict, metadata
        
    except FileNotFoundError as e:
        st.error(f"Model files not found! Please run Risk_Predictor2.ipynb first to train and save the model.")
        st.error(f"Missing file: {str(e)}")
        return None, None, None, None

@st.cache_data
def load_student_data():
    """Load student data"""
    try:
        # Try to load original data first
        df = pd.read_csv('data/StudentPerformanceFactors.csv')
        df = df.dropna()
        
        # Recreate risk labels (same logic as in training)
        def create_comprehensive_risk_label(df):
            # Academic Performance Risk (40% weight)
            exam_low = df['Exam_Score'] < df['Exam_Score'].quantile(0.25)
            prev_low = df['Previous_Scores'] < df['Previous_Scores'].quantile(0.25)
            academic_risk = (exam_low.astype(int) * 0.6 + prev_low.astype(int) * 0.4)
            
            # Engagement Risk (25% weight)
            attendance_low = df['Attendance'] < df['Attendance'].quantile(0.3)
            hours_low = df['Hours_Studied'] < df['Hours_Studied'].quantile(0.3)
            engagement_risk = (attendance_low.astype(int) * 0.6 + hours_low.astype(int) * 0.4)
            
            # Support System Risk (20% weight)
            parental_low = df['Parental_Involvement'].str.lower() == 'low'
            resources_low = df['Access_to_Resources'].str.lower() == 'low'
            support_risk = (parental_low.astype(int) * 0.6 + resources_low.astype(int) * 0.4)
            
            # Personal Factors Risk (15% weight)
            motivation_low = df['Motivation_Level'].str.lower() == 'low'
            sleep_poor = (df['Sleep_Hours'] < 6) | (df['Sleep_Hours'] > 9)
            peer_negative = df['Peer_Influence'].str.lower() == 'negative'
            learning_disability = df['Learning_Disabilities'].str.lower() == 'yes'
            personal_risk = (motivation_low.astype(int) * 0.4 + sleep_poor.astype(int) * 0.2 + 
                            peer_negative.astype(int) * 0.2 + learning_disability.astype(int) * 0.2)
            
            # Combine all risk factors with weights
            composite_risk_score = (academic_risk * 0.40 + engagement_risk * 0.25 + 
                                   support_risk * 0.20 + personal_risk * 0.15)
            
            # Students in top 30% of risk scores are considered "at risk"
            risk_threshold = np.percentile(composite_risk_score, 70)
            at_risk = (composite_risk_score >= risk_threshold).astype(int)
            
            return at_risk, composite_risk_score, risk_threshold
        
        df['AtRisk'], df['RiskScore'], _ = create_comprehensive_risk_label(df)
        return df
        
    except FileNotFoundError:
        # Fallback to sample data
        try:
            df = pd.read_csv('models/sample_data.csv')
            return df
        except FileNotFoundError:
            st.error("No student data found! Please ensure the data files are available.")
            return None

def get_risk_level_info(probability):
    """Get risk level information with colors and descriptions"""
    if probability >= 0.7:
        return {
            'level': 'HIGH RISK',
            'emoji': '🚨',
            'color': '#f44336',
            'bg_class': 'risk-high',
            'description': 'This student requires immediate intervention and support.'
        }
    elif probability >= 0.4:
        return {
            'level': 'MODERATE RISK',
            'emoji': '⚠️',
            'color': '#ff9800',
            'bg_class': 'risk-medium',
            'description': 'This student shows warning signs and would benefit from additional support.'
        }
    elif probability >= 0.2:
        return {
            'level': 'LOW-MODERATE RISK',
            'emoji': '⚠️',
            'color': '#ffc107',
            'bg_class': 'risk-medium',
            'description': 'This student has some risk factors that should be monitored.'
        }
    else:
        return {
            'level': 'LOW RISK',
            'emoji': '✅',
            'color': '#4caf50',
            'bg_class': 'risk-low',
            'description': 'This student is performing well and is likely to succeed.'
        }

def create_risk_gauge(probability):
    """Create a risk probability gauge chart"""
    risk_info = get_risk_level_info(probability)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Probability", 'font': {'size': 20}},
        number = {'suffix': '%', 'font': {'size': 30, 'color': risk_info['color']}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': risk_info['color'], 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#e8f5e8'},
                {'range': [20, 40], 'color': '#fff3e0'},
                {'range': [40, 70], 'color': '#ffeaa7'},
                {'range': [70, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=350, font={'color': "darkblue", 'family': "Arial"})
    return fig

def create_factors_chart(top_factors, student_data):
    """Create horizontal bar chart for influential factors"""
    
    colors = []
    hover_texts = []
    
    for _, row in top_factors.iterrows():
        factor_name = row['factor']
        contribution = row['contribution']
        actual_value = student_data.get(factor_name, 'N/A')
        
        # Color coding based on factor performance
        if factor_name == 'Exam_Score':
            if actual_value < 60:
                color = '#f44336'
                hover_text = f"Low exam score: {actual_value}"
            elif actual_value < 75:
                color = '#ff9800'
                hover_text = f"Moderate exam score: {actual_value}"
            else:
                color = '#4caf50'
                hover_text = f"Good exam score: {actual_value}"
        elif factor_name == 'Attendance':
            if actual_value < 80:
                color = '#f44336'
                hover_text = f"Low attendance: {actual_value}%"
            elif actual_value < 90:
                color = '#ff9800'
                hover_text = f"Moderate attendance: {actual_value}%"
            else:
                color = '#4caf50'
                hover_text = f"Good attendance: {actual_value}%"
        elif factor_name == 'Hours_Studied':
            if actual_value < 15:
                color = '#f44336'
                hover_text = f"Low study hours: {actual_value}/week"
            elif actual_value < 25:
                color = '#ff9800'
                hover_text = f"Moderate study hours: {actual_value}/week"
            else:
                color = '#4caf50'
                hover_text = f"Good study hours: {actual_value}/week"
        else:
            color = '#2196f3'
            hover_text = f"{factor_name}: {actual_value}"
        
        colors.append(color)
        hover_texts.append(hover_text)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_factors['factor'],
        x=top_factors['contribution'],
        orientation='h',
        marker_color=colors,
        text=[f'{val:.3f}' for val in top_factors['contribution']],
        textposition='auto',
        hovertext=hover_texts,
        hovertemplate='%{hovertext}<br>Influence: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="🔍 Top Factors Influencing This Prediction",
        xaxis_title="Influence Score",
        yaxis_title="",
        height=400,
        margin=dict(l=150, r=50, t=50, b=50),
        showlegend=False
    )
    
    return fig

def predict_student_risk(student_id, df, model, scaler, le_dict, feature_names):
    """Predict risk for a student by ID"""
    
    row = df[df['Student_ID'] == student_id]
    if row.empty:
        return None, None, None, None
    
    student_info = row.iloc[0].to_dict()
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['Student_ID', 'AtRisk', 'RiskScore']]
    features = row[feature_cols].copy()
    
    # Encode categorical features
    features_encoded = features.copy()
    for col in le_dict.keys():
        if col in features_encoded.columns:
            try:
                features_encoded[col] = le_dict[col].transform(features[col])
            except ValueError:
                # Handle unknown categories
                features_encoded[col] = 0
    
    # Scale features
    features_scaled = scaler.transform(features_encoded)
    
    # Make prediction
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]
    
    # Get top influential factors
    feature_contributions = model.feature_importances_ * np.abs(features_scaled[0])
    top_factors = pd.DataFrame({
        'factor': feature_names,
        'contribution': feature_contributions
    }).sort_values('contribution', ascending=False).head(5)
    
    return pred, prob[1], student_info, top_factors

def get_recommendations(student_data, risk_probability):
    """Generate personalized recommendations based on risk factors"""
    recommendations = []
    
    if student_data.get('Attendance', 100) < 85:
        recommendations.append({
            'title': '🎯 Improve Attendance',
            'description': 'Aim for 90%+ attendance. Consider addressing barriers to regular school attendance.',
            'priority': 'high'
        })
    
    if student_data.get('Hours_Studied', 30) < 15:
        recommendations.append({
            'title': '📚 Increase Study Time',
            'description': 'Aim for at least 20 hours of focused study per week with a structured schedule.',
            'priority': 'high'
        })
    
    if student_data.get('Exam_Score', 100) < 70:
        recommendations.append({
            'title': '📝 Academic Support',
            'description': 'Consider additional tutoring or study groups to improve exam performance.',
            'priority': 'high'
        })
    
    if student_data.get('Motivation_Level', '').lower() == 'low':
        recommendations.append({
            'title': '💪 Boost Motivation',
            'description': 'Set small, achievable goals and celebrate progress. Consider meeting with a counselor.',
            'priority': 'medium'
        })
    
    sleep_hours = student_data.get('Sleep_Hours', 8)
    if sleep_hours < 7 or sleep_hours > 9:
        recommendations.append({
            'title': '😴 Improve Sleep Schedule',
            'description': 'Maintain 7-9 hours of sleep daily for optimal cognitive function.',
            'priority': 'medium'
        })
    
    if student_data.get('Parental_Involvement', '').lower() == 'low':
        recommendations.append({
            'title': '👨‍👩‍👧‍👦 Family Engagement',
            'description': 'Encourage more family involvement in educational activities and progress tracking.',
            'priority': 'medium'
        })
    
    if student_data.get('Peer_Influence', '').lower() == 'negative':
        recommendations.append({
            'title': '👥 Positive Peer Groups',
            'description': 'Seek out study groups and positive peer influences to support academic goals.',
            'priority': 'medium'
        })
    
    if not recommendations:
        recommendations.append({
            'title': '✅ Maintain Current Performance',
            'description': 'Continue current study habits and engagement levels.',
            'priority': 'low'
        })
        recommendations.append({
            'title': '🎯 Set Higher Goals',
            'description': 'Consider challenging yourself with advanced courses or leadership opportunities.',
            'priority': 'low'
        })
    
    return recommendations

# Main App
def main():
    # Title and header
    st.markdown('<div class="main-header">🎓 Student Risk Assessment System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and data
    model, scaler, le_dict, metadata = load_model_components()
    df = load_student_data()
    
    if model is None or df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("🔍 Student Risk Assessment")
    st.sidebar.markdown("Select a student to analyze their academic risk profile.")
    
    # Student selection
    available_ids = sorted(df['Student_ID'].unique())
    
    # Sample student suggestions
    st.sidebar.markdown("### 💡 Quick Selection")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🎲 Random Student", help="Select a random student"):
            st.session_state.selected_student = np.random.choice(available_ids)
    
    with col2:
        if st.button("🔄 Refresh Data", help="Reload the data"):
            st.cache_data.clear()
            st.rerun()
    
    # Student ID input
    default_student = st.session_state.get('selected_student', available_ids[0])
    student_id = st.sidebar.selectbox(
        "Choose Student ID:",
        options=available_ids,
        index=available_ids.index(default_student) if default_student in available_ids else 0,
        help="Select a student ID to view their comprehensive risk assessment"
    )
    
    # Analysis button
    if st.sidebar.button("📊 Analyze Student", type="primary", help="Generate comprehensive risk assessment"):
        
        # Predict risk
        pred, risk_prob, student_data, top_factors = predict_student_risk(
            student_id, df, model, scaler, le_dict, metadata['feature_names']
        )
        
        if pred is None:
            st.error("Student not found!")
            return
        
        # Get risk level info
        risk_info = get_risk_level_info(risk_prob)
        
        # Main content area
        st.markdown(f"## 📋 Risk Assessment Report for Student ID: **{student_id}**")
        
        # Risk level display
        st.markdown(f"""
        <div class="{risk_info['bg_class']}">
            <h2>{risk_info['emoji']} {risk_info['level']}</h2>
            <p style="font-size: 1.1em; margin-bottom: 0;">{risk_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main dashboard
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Risk gauge
            st.plotly_chart(create_risk_gauge(risk_prob), use_container_width=True)
            
            # Risk probabilities
            st.markdown("### 📈 Probability Breakdown")
            st.metric("Risk Probability", f"{risk_prob:.1%}", delta=None)
            st.metric("Safe Probability", f"{1-risk_prob:.1%}", delta=None)
        
        with col2:
            # Top factors chart
            st.plotly_chart(create_factors_chart(top_factors, student_data), use_container_width=True)
        
        # Student profile
        st.markdown("## 👤 Student Profile")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 📚 Academic")
            exam_score = student_data.get('Exam_Score', 0)
            st.metric("Exam Score", f"{exam_score}/100", 
                     delta=f"{exam_score - student_data.get('Previous_Scores', exam_score):+.0f}")
            st.metric("Previous Scores", f"{student_data.get('Previous_Scores', 0)}")
            st.metric("Study Hours/Week", f"{student_data.get('Hours_Studied', 0)}")
        
        with col2:
            st.markdown("### 📈 Engagement")
            attendance = student_data.get('Attendance', 0)
            st.metric("Attendance", f"{attendance}%")
            st.metric("Motivation", student_data.get('Motivation_Level', 'N/A'))
            st.metric("Sleep Hours", f"{student_data.get('Sleep_Hours', 0)}")
        
        with col3:
            st.markdown("### 👨‍👩‍👧‍👦 Support")
            st.metric("Parental Involvement", student_data.get('Parental_Involvement', 'N/A'))
            st.metric("Access to Resources", student_data.get('Access_to_Resources', 'N/A'))
            st.metric("Tutoring Sessions", f"{student_data.get('Tutoring_Sessions', 0)}")
        
        with col4:
            st.markdown("### 🌟 Environment")
            st.metric("Peer Influence", student_data.get('Peer_Influence', 'N/A'))
            st.metric("School Type", student_data.get('School_Type', 'N/A'))
            st.metric("Learning Disabilities", student_data.get('Learning_Disabilities', 'N/A'))
        
        # Recommendations
        st.markdown("## 💡 Personalized Recommendations")
        recommendations = get_recommendations(student_data, risk_prob)
        
        for i, rec in enumerate(recommendations):
            priority_colors = {'high': '#f44336', 'medium': '#ff9800', 'low': '#4caf50'}
            priority_color = priority_colors.get(rec['priority'], '#2196f3')
            
            st.markdown(f"""
            <div class="recommendation-item" style="border-left-color: {priority_color};">
                <h4 style="margin: 0 0 0.5rem 0; color: {priority_color};">{rec['title']}</h4>
                <p style="margin: 0;">{rec['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("## 📊 Additional Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk factors breakdown
            st.markdown("### ⚠️ Risk Factors Identified")
            risk_factors = []
            
            if student_data.get('Exam_Score', 100) < 60:
                risk_factors.append("Low exam performance")
            if student_data.get('Attendance', 100) < 80:
                risk_factors.append("Poor attendance")
            if student_data.get('Hours_Studied', 30) < 15:
                risk_factors.append("Insufficient study time")
            if student_data.get('Motivation_Level', '').lower() == 'low':
                risk_factors.append("Low motivation")
            if student_data.get('Peer_Influence', '').lower() == 'negative':
                risk_factors.append("Negative peer influence")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
            else:
                st.markdown("✅ No major risk factors identified")
        
        with col2:
            # Positive factors
            st.markdown("### ✅ Positive Factors")
            positive_factors = []
            
            if student_data.get('Exam_Score', 0) >= 75:
                positive_factors.append("Good exam performance")
            if student_data.get('Attendance', 0) >= 90:
                positive_factors.append("Excellent attendance")
            if student_data.get('Hours_Studied', 0) >= 25:
                positive_factors.append("High study commitment")
            if student_data.get('Motivation_Level', '').lower() == 'high':
                positive_factors.append("High motivation")
            if student_data.get('Parental_Involvement', '').lower() == 'high':
                positive_factors.append("Strong parental support")
            
            if positive_factors:
                for factor in positive_factors:
                    st.markdown(f"• {factor}")
            else:
                st.markdown("⚠️ Consider developing more positive factors")
        
        # Model information
        with st.expander("🔧 Model Information", expanded=False):
            st.markdown(f"""
            **Model Performance:**
            - Cross-validation F1 Score: {metadata['model_performance']['cv_f1_mean']:.3f}
            - Test Accuracy: {metadata['model_performance']['test_accuracy']:.3f}
            - Test AUC: {metadata['model_performance']['test_auc']:.3f}
            
            **Features Used:** {len(metadata['feature_names'])} features
            
            **Assessment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)

if __name__ == "__main__":
    main()