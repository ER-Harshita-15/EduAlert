# EduAlert
# 🎓 Student Risk Assessment System

An AI-powered system that predicts student academic risk using machine learning to identify students who may need additional support.

[🎥 Watch Model Usecase Video](https://github.com/user-attachments/assets/3cc55e8f-3d2f-443a-8f23-4286bdcf3cfa)

## Folder Structure 

```
EduAlert/
├── .gitignore
├── app.py
├── demo.html
├── requirements.txt
├── README.md
│
├── App_Flow/
│   ├── About_Student_UI.jpg
│   ├── All_Student_dashboard.jpg
│   ├── App_development_process.jpg
│   ├── App_Features.png
│   └── Single Student Dashboard.jpg
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── models/
│   ├── student_risk_model.pkl
│   ├── feature_scaler.pkl
│   ├── label_encoders.pkl
│   ├── model_metadata.pkl
│   └── sample_data.csv
│
└── src/
    └── Risk_predictor.ipynb
```

## Features

- **Comprehensive Risk Assessment**: Analyzes multiple factors including academic performance, attendance, study habits, and support systems
- **Interactive Dashboard**: Beautiful visualizations with risk gauges, charts, and detailed student profiles
- **Personalized Recommendations**: Actionable advice based on individual risk factors
- **Real-time Predictions**: Instant risk assessment for any student

## How to Use

1. Select a student ID from the dropdown menu
2. Click "📊 Analyze Student" to generate the assessment
3. Review the risk level, influential factors, and recommendations
4. Use insights to provide targeted student support

## Model Information

The system uses a Random Forest classifier trained on student performance data, considering:
- Academic performance (exam scores, previous scores)
- Engagement factors (attendance, study hours)
- Support systems (parental involvement, resources)
- Personal factors (motivation, sleep, peer influence)

- **Algorithm**: Random Forest Classifier
- **Features**: 20 student performance indicators
- **Accuracy**: ~95% on test data
- **Risk Categories**: Low, Moderate, High risk levels

## Deployment

This app is deployed on Streamlit Cloud and automatically updates from the GitHub repository.

## Streamlit app:
  [Student_Risk_prediction_ML_Model](https://edualert01hlk.streamlit.app/)

## demo.html 

This file is containing everthing which we aim to tackle and overcome using this solution 


## Data Privacy

This system uses anonymized student data for demonstration purposes. In a real deployment, ensure compliance with educational data privacy regulations.

---

Built with ❤️ using Streamlit and scikit-learn

