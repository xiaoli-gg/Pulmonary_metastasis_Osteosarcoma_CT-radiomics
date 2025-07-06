import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Medical Image Survival Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        color: #d32f2f;
        border: 2px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_package = joblib.load('ensemble_model.pkl')
        return model_package
    except FileNotFoundError:
        st.error("⚠️ Trained model file not found (ensemble_model.pkl)")
        st.info("Please ensure the model file is in the current directory, or run the training script first to generate the model")
        return None

def predict_survival(model_package, input_data):
    """Perform survival prediction"""
    try:
        models = model_package['models']
        weights = model_package['weights']
        feature_names = model_package['feature_names']
        threshold = model_package['threshold']
        
        # Prepare feature data for each category
        X_trad = input_data[feature_names['traditional']]
        X_deep = input_data[feature_names['deep_learning']]
        X_clinical = input_data[feature_names['clinical']]
        
        # Get prediction probabilities from each model
        proba_trad = models['traditional'].predict_proba(X_trad.values.reshape(1, -1))[0, 1]
        proba_deep = models['deep_learning'].predict_proba(X_deep.values.reshape(1, -1))[0, 1]
        proba_clinical = models['clinical'].predict_proba(X_clinical.values.reshape(1, -1))[0, 1]
        
        # Weighted fusion
        weighted_proba = (weights['traditional'] * proba_trad + 
                         weights['deep_learning'] * proba_deep + 
                         weights['clinical'] * proba_clinical)
        
        # Final prediction
        prediction = 1 if weighted_proba > threshold else 0
        
        return {
            'prediction': prediction,
            'probability': weighted_proba,
            'individual_probabilities': {
                'traditional': proba_trad,
                'deep_learning': proba_deep,
                'clinical': proba_clinical
            },
            'weights': weights
        }
        
    except Exception as e:
        st.error(f"Error occurred during prediction: {str(e)}")
        return None

def main():
    # Main title
    st.markdown('<h1 class="main-header">🏥 Medical Image Survival Predictor</h1>', unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    
    if model_package is None:
        st.stop()
    
    # Display model information
    with st.expander("📊 Model Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔬 Traditional Imaging Features")
            for feature in model_package['feature_names']['traditional']:
                st.write(f"• {feature}")
        
        with col2:
            st.markdown("### 🤖 Deep Learning Features")
            for feature in model_package['feature_names']['deep_learning']:
                st.write(f"• {feature}")
        
        with col3:
            st.markdown("### 👨‍⚕️ Clinical Features")
            for feature in model_package['feature_names']['clinical']:
                st.write(f"• {feature}")
        
        st.markdown("### 📈 Model Performance Metrics")
        metrics = model_package['performance_metrics']
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("AUC", f"{metrics['AUC']:.3f}")
        with col2:
            st.metric("F1 Score", f"{metrics['F1']:.3f}")
        with col3:
            st.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        with col4:
            st.metric("Sensitivity", f"{metrics['Sensitivity']:.3f}")
        with col5:
            st.metric("Specificity", f"{metrics['Specificity']:.3f}")
    
    # Sidebar - Data input
    st.sidebar.markdown("## 📝 Patient Data Input")
    
    # Clinical features input
    st.sidebar.markdown("### 👨‍⚕️ Clinical Features")
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=65, step=1)
    alp = st.sidebar.number_input("ALP", min_value=0.0, value=100.0, step=0.1)
    monocyte = st.sidebar.number_input("Monocyte", min_value=0.0, value=0.5, step=0.01)
    neutrophil = st.sidebar.number_input("Neutrophil", min_value=0.0, value=5.0, step=0.1)
    mlr = st.sidebar.number_input("MLR", min_value=0.0, value=0.3, step=0.01)
    
    # Traditional imaging features input
    st.sidebar.markdown("### 🔬 Traditional Imaging Features")
    rad_features = {}
    feature_names_trad = model_package['feature_names']['traditional']
    
    for i, feature in enumerate(feature_names_trad):
        rad_features[feature] = st.sidebar.number_input(
            f"{feature}", 
            value=0.0, 
            step=0.001, 
            format="%.3f",
            key=f"trad_{i}"
        )
    
    # Deep learning features input
    st.sidebar.markdown("### 🤖 Deep Learning Features")
    deep_features = {}
    feature_names_deep = model_package['feature_names']['deep_learning']
    
    for i, feature in enumerate(feature_names_deep):
        deep_features[feature] = st.sidebar.number_input(
            f"{feature}", 
            value=0.0, 
            step=0.001, 
            format="%.3f",
            key=f"deep_{i}"
        )
    
    # Prediction button
    if st.sidebar.button("🔍 Start Prediction", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'ALP': [alp],
            'Monocyte': [monocyte],
            'Neutrophil': [neutrophil],
            'MLR': [mlr],
            **{k: [v] for k, v in rad_features.items()},
            **{k: [v] for k, v in deep_features.items()}
        })
        
        # Perform prediction
        result = predict_survival(model_package, input_data)
        
        if result is not None:
            # Display prediction results
            st.markdown('<h2 class="section-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Main prediction result
            prediction_text = "High Risk" if result['prediction'] == 1 else "Low Risk"
            risk_class = "high-risk" if result['prediction'] == 1 else "low-risk"
            
            st.markdown(f'''
            <div class="prediction-result {risk_class}">
                🎯 Prediction Result: {prediction_text}<br>
                📊 Risk Probability: {result['probability']:.1%}
            </div>
            ''', unsafe_allow_html=True)
            
            # Detailed results display
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📈 Model Contributions")
                
                # Create contribution chart
                labels = ['Traditional Imaging', 'Deep Learning', 'Clinical Features']
                probabilities = [
                    result['individual_probabilities']['traditional'],
                    result['individual_probabilities']['deep_learning'],
                    result['individual_probabilities']['clinical']
                ]
                weights = [
                    result['weights']['traditional'],
                    result['weights']['deep_learning'],
                    result['weights']['clinical']
                ]
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Model Prediction Probabilities', 'Model Weights'),
                    specs=[[{"type": "bar"}, {"type": "pie"}]]
                )
                
                # Probability bar chart
                fig.add_trace(
                    go.Bar(x=labels, y=probabilities, 
                          marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                          name='Prediction Probability'),
                    row=1, col=1
                )
                
                # Weight pie chart
                fig.add_trace(
                    go.Pie(labels=labels, values=weights,
                          marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                          name='Model Weights'),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    title_text="Model Analysis"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📋 Detailed Data")
                
                # Input data summary
                st.markdown("#### Clinical Features")
                clinical_data = {
                    "Age": f"{age} years",
                    "ALP": f"{alp:.1f}",
                    "Monocyte": f"{monocyte:.2f}",
                    "Neutrophil": f"{neutrophil:.1f}",
                    "MLR": f"{mlr:.2f}"
                }
                
                for key, value in clinical_data.items():
                    st.write(f"• **{key}**: {value}")
                
                st.markdown("#### Prediction Details")
                st.write(f"• **Final Probability**: {result['probability']:.3f}")
                st.write(f"• **Prediction Threshold**: {model_package['threshold']}")
                st.write(f"• **Traditional Imaging Contribution**: {result['individual_probabilities']['traditional']:.3f}")
                st.write(f"• **Deep Learning Contribution**: {result['individual_probabilities']['deep_learning']:.3f}")
                st.write(f"• **Clinical Features Contribution**: {result['individual_probabilities']['clinical']:.3f}")
            
            # Risk interpretation
            st.markdown('<h3 class="section-header">💡 Result Interpretation</h3>', unsafe_allow_html=True)
            
            if result['prediction'] == 1:
                st.warning("""
                **High Risk Patient Recommendations:**
                - 🏥 Recommend enhanced follow-up monitoring
                - 📅 Shorten re-examination intervals
                - 💊 Consider aggressive treatment plans
                - 👨‍⚕️ Discuss treatment strategies in detail with attending physician
                """)
            else:
                st.success("""
                **Low Risk Patient Recommendations:**
                - ✅ Maintain routine follow-up schedule
                - 🏃‍♂️ Maintain healthy lifestyle
                - 📊 Regular monitoring of relevant indicators
                - 😊 Maintain positive attitude
                """)
            
            # Disclaimer
            st.markdown("---")
            st.markdown("""
            **⚠️ Important Disclaimer:**
            This prediction result is for medical reference only and cannot replace professional medical diagnosis.
            Please consult with professional physicians and make medical decisions based on actual clinical conditions.
            """)

if __name__ == "__main__":
    main()
