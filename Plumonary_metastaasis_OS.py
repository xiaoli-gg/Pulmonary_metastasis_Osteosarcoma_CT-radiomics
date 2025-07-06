import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="医学影像生存预测器",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
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
    """加载训练好的模型"""
    try:
        model_package = joblib.load('ensemble_model.pkl')
        return model_package
    except FileNotFoundError:
        st.error("⚠️ 未找到训练好的模型文件 (ensemble_model.pkl)")
        st.info("请确保模型文件在当前目录下，或先运行训练脚本生成模型")
        return None

def predict_survival(model_package, input_data):
    """进行生存预测"""
    try:
        models = model_package['models']
        weights = model_package['weights']
        feature_names = model_package['feature_names']
        threshold = model_package['threshold']
        
        # 准备各类特征数据
        X_trad = input_data[feature_names['traditional']]
        X_deep = input_data[feature_names['deep_learning']]
        X_clinical = input_data[feature_names['clinical']]
        
        # 获取各模型预测概率
        proba_trad = models['traditional'].predict_proba(X_trad.values.reshape(1, -1))[0, 1]
        proba_deep = models['deep_learning'].predict_proba(X_deep.values.reshape(1, -1))[0, 1]
        proba_clinical = models['clinical'].predict_proba(X_clinical.values.reshape(1, -1))[0, 1]
        
        # 加权融合
        weighted_proba = (weights['traditional'] * proba_trad + 
                         weights['deep_learning'] * proba_deep + 
                         weights['clinical'] * proba_clinical)
        
        # 最终预测
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
        st.error(f"预测过程中出现错误: {str(e)}")
        return None

def main():
    # 主标题
    st.markdown('<h1 class="main-header">🏥 医学影像生存预测器</h1>', unsafe_allow_html=True)
    
    # 加载模型
    model_package = load_model()
    
    if model_package is None:
        st.stop()
    
    # 显示模型信息
    with st.expander("📊 模型信息", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔬 传统影像特征")
            for feature in model_package['feature_names']['traditional']:
                st.write(f"• {feature}")
        
        with col2:
            st.markdown("### 🤖 深度学习特征")
            for feature in model_package['feature_names']['deep_learning']:
                st.write(f"• {feature}")
        
        with col3:
            st.markdown("### 👨‍⚕️临床特征")
            for feature in model_package['feature_names']['clinical']:
                st.write(f"• {feature}")
        
        st.markdown("### 📈 模型性能指标")
        metrics = model_package['performance_metrics']
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("AUC", f"{metrics['AUC']:.3f}")
        with col2:
            st.metric("F1 Score", f"{metrics['F1']:.3f}")
        with col3:
            st.metric("准确率", f"{metrics['Accuracy']:.3f}")
        with col4:
            st.metric("敏感性", f"{metrics['Sensitivity']:.3f}")
        with col5:
            st.metric("特异性", f"{metrics['Specificity']:.3f}")
    
    # 侧边栏 - 数据输入
    st.sidebar.markdown("## 📝 患者数据输入")
    
    # 临床特征输入
    st.sidebar.markdown("### 👨‍⚕️ 临床特征")
    age = st.sidebar.number_input("年龄", min_value=0, max_value=120, value=65, step=1)
    alp = st.sidebar.number_input("ALP (碱性磷酸酶)", min_value=0.0, value=100.0, step=0.1)
    monocyte = st.sidebar.number_input("单核细胞", min_value=0.0, value=0.5, step=0.01)
    neutrophil = st.sidebar.number_input("中性粒细胞", min_value=0.0, value=5.0, step=0.1)
    mlr = st.sidebar.number_input("MLR (单核细胞淋巴细胞比值)", min_value=0.0, value=0.3, step=0.01)
    
    # 传统影像特征输入
    st.sidebar.markdown("### 🔬 传统影像特征")
    rad_features = {}
    feature_names_trad = model_package['feature_names']['traditional']
    
    for i, feature in enumerate(feature_names_trad):
        display_name = feature.split('_')[-1] if '_' in feature else feature
        rad_features[feature] = st.sidebar.number_input(
            f"{display_name}", 
            value=0.0, 
            step=0.001, 
            format="%.3f",
            key=f"trad_{i}"
        )
    
    # 深度学习特征输入
    st.sidebar.markdown("### 🤖 深度学习特征")
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
    
    # 预测按钮
    if st.sidebar.button("🔍 开始预测", type="primary"):
        # 准备输入数据
        input_data = pd.DataFrame({
            'Age': [age],
            'ALP': [alp],
            'Monocyte': [monocyte],
            'Neutrophil': [neutrophil],
            'MLR': [mlr],
            **{k: [v] for k, v in rad_features.items()},
            **{k: [v] for k, v in deep_features.items()}
        })
        
        # 进行预测
        result = predict_survival(model_package, input_data)
        
        if result is not None:
            # 显示预测结果
            st.markdown('<h2 class="section-header">📊 预测结果</h2>', unsafe_allow_html=True)
            
            # 主要预测结果
            prediction_text = "高风险" if result['prediction'] == 1 else "低风险"
            risk_class = "high-risk" if result['prediction'] == 1 else "low-risk"
            
            st.markdown(f'''
            <div class="prediction-result {risk_class}">
                🎯 预测结果: {prediction_text}<br>
                📊 风险概率: {result['probability']:.1%}
            </div>
            ''', unsafe_allow_html=True)
            
            # 详细结果展示
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📈 各模型贡献度")
                
                # 创建贡献度图表
                labels = ['传统影像', '深度学习', '临床特征']
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
                    subplot_titles=('各模型预测概率', '模型权重'),
                    specs=[[{"type": "bar"}, {"type": "pie"}]]
                )
                
                # 概率条形图
                fig.add_trace(
                    go.Bar(x=labels, y=probabilities, 
                          marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                          name='预测概率'),
                    row=1, col=1
                )
                
                # 权重饼图
                fig.add_trace(
                    go.Pie(labels=labels, values=weights,
                          marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                          name='模型权重'),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    title_text="模型分析"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📋 详细数据")
                
                # 输入数据摘要
                st.markdown("#### 临床特征")
                clinical_data = {
                    "年龄": f"{age} 岁",
                    "ALP": f"{alp:.1f}",
                    "单核细胞": f"{monocyte:.2f}",
                    "中性粒细胞": f"{neutrophil:.1f}",
                    "MLR": f"{mlr:.2f}"
                }
                
                for key, value in clinical_data.items():
                    st.write(f"• **{key}**: {value}")
                
                st.markdown("#### 预测详情")
                st.write(f"• **最终概率**: {result['probability']:.3f}")
                st.write(f"• **预测阈值**: {model_package['threshold']}")
                st.write(f"• **传统影像贡献**: {result['individual_probabilities']['traditional']:.3f}")
                st.write(f"• **深度学习贡献**: {result['individual_probabilities']['deep_learning']:.3f}")
                st.write(f"• **临床特征贡献**: {result['individual_probabilities']['clinical']:.3f}")
            
            # 风险解释
            st.markdown('<h3 class="section-header">💡 结果解释</h3>', unsafe_allow_html=True)
            
            if result['prediction'] == 1:
                st.warning("""
                **高风险患者建议：**
                - 🏥 建议加强随访监测
                - 📅 缩短复查间隔
                - 💊 考虑积极的治疗方案
                - 👨‍⚕️ 与主治医师详细讨论治疗策略
                """)
            else:
                st.success("""
                **低风险患者建议：**
                - ✅ 维持常规随访计划
                - 🏃‍♂️ 保持健康的生活方式
                - 📊 定期复查相关指标
                - 😊 保持积极的心态
                """)
            
            # 免责声明
            st.markdown("---")
            st.markdown("""
            **⚠️ 重要声明：**
            此预测结果仅供医疗参考，不能替代专业医疗诊断。
            请务必咨询专业医师，结合临床实际情况做出医疗决策。
            """)

if __name__ == "__main__":
    main()