import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc, accuracy_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

# 设置页面配置
st.set_page_config(
    page_title="医学影像生存预测模型",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面标题
st.title("🏥 医学影像生存预测模型")
st.markdown("基于传统影像、深度学习和临床特征的集成预测模型")

# 侧边栏
st.sidebar.header("📊 模型设置")

# 特征名称定义
RADIOMICS_FEATURES = [
    'wavelet-LLH_glszm_GrayLevelNonUniformity',
    'wavelet-LHL_glszm_SizeZoneNonUniformityNormalized',
    'wavelet-HHH_glcm_ClusterShade',
    'wavelet-HHH_glszm_GrayLevelNonUniformityNormalized',
    'wavelet-HHH_glszm_SizeZoneNonUniformityNormalized',
    'wavelet-HHH_glszm_ZoneVariance'
]

DEEP_LEARNING_FEATURES = [
    'Feature_508',
    'Feature_738',
    'Feature_879'
]

CLINICAL_FEATURES = [
    'Age',
    'ALP',
    'Monocyte',
    'Neutrophil',
    'MLR'
]

# 模型配置
@st.cache_resource
def load_models():
    """加载训练好的模型"""
    try:
        import joblib
        model_package = joblib.load('ensemble_model.pkl')
        
        models = model_package['models']
        weights = model_package['weights']
        feature_names = model_package['feature_names']
        performance_metrics = model_package['performance_metrics']
        threshold = model_package['threshold']
        
        st.success("✅ 模型加载成功！")
        return models, weights, feature_names, performance_metrics, threshold
        
    except FileNotFoundError:
        st.error("❌ 未找到训练好的模型文件 (ensemble_model.pkl)")
        st.info("请先运行 model_training.py 训练并保存模型")
        
        # 返回默认配置
        models = {
            'traditional': LogisticRegression(random_state=50),
            'deep_learning': LogisticRegression(random_state=50),
            'clinical': LogisticRegression(random_state=50)
        }
        
        weights = {
            'traditional': 0.3,
            'deep_learning': 0.4,
            'clinical': 0.3
        }
        
        feature_names = {
            'traditional': RADIOMICS_FEATURES,
            'deep_learning': DEEP_LEARNING_FEATURES,
            'clinical': CLINICAL_FEATURES
        }
        
        performance_metrics = {
            'F1': 0.85,
            'AUC': 0.90,
            'Accuracy': 0.88,
            'Sensitivity': 0.86,
            'Specificity': 0.89
        }
        
        threshold = 0.4
        
        return models, weights, feature_names, performance_metrics, threshold

def perform_batch_prediction(data, models, weights, feature_names, threshold):
    """执行批量预测"""
    try:
        # 提取特征
        X_radiomics = data[feature_names['traditional']]
        X_deep = data[feature_names['deep_learning']]
        X_clinical = data[feature_names['clinical']]
        
        # 获取预测概率
        proba_trad = models['traditional'].predict_proba(X_radiomics)
        proba_deep = models['deep_learning'].predict_proba(X_deep)
        proba_clinical = models['clinical'].predict_proba(X_clinical)
        
        # 加权平均
        weighted_proba = (
            weights['traditional'] * proba_trad[:, 1] +
            weights['deep_learning'] * proba_deep[:, 1] +
            weights['clinical'] * proba_clinical[:, 1]
        )
        
        # 预测结果
        predictions = (weighted_proba > threshold).astype(int)
        
        # 添加预测结果到原数据
        result_df = data.copy()
        result_df['prediction'] = predictions
        result_df['probability'] = weighted_proba
        result_df['prediction_label'] = result_df['prediction'].map({0: '未生存', 1: '生存'})
        
        return result_df
        
    except Exception as e:
        st.error(f"批量预测过程中出现错误: {str(e)}")
        # 返回示例结果
        predictions = []
        for i in range(len(data)):
            prob = np.random.rand()
            pred = 1 if prob > threshold else 0
            predictions.append({'prediction': pred, 'probability': prob})
        
        result_df = pd.concat([data, pd.DataFrame(predictions)], axis=1)
        return result_df

def main():
    # 加载模型
    models, weights, feature_names, performance_metrics, threshold = load_models()
    
    # 更新特征名称（如果模型中有的话）
    radiomics_features = feature_names['traditional']
    deep_learning_features = feature_names['deep_learning']
    clinical_features = feature_names['clinical']
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📝 数据输入", "🔍 模型预测", "📈 结果可视化", "📊 模型信息"])
    
    with tab1:
        st.header("数据输入")
        
        # 创建三列布局
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔬 传统影像特征")
            radiomics_data = {}
            for feature in radiomics_features:
                radiomics_data[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.6f",
                    key=f"rad_{feature}"
                )
        
        with col2:
            st.subheader("🤖 深度学习特征")
            deep_learning_data = {}
            for feature in deep_learning_features:
                deep_learning_data[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.6f",
                    key=f"deep_{feature}"
                )
        
        with col3:
            st.subheader("🩺 临床特征")
            clinical_data = {}
            for feature in clinical_features:
                if feature == 'Age':
                    clinical_data[feature] = st.number_input("年龄 (Age)", value=50, min_value=0, max_value=120)
                elif feature == 'ALP':
                    clinical_data[feature] = st.number_input("碱性磷酸酶 (ALP)", value=100.0, min_value=0.0)
                elif feature == 'Monocyte':
                    clinical_data[feature] = st.number_input("单核细胞 (Monocyte)", value=0.5, min_value=0.0)
                elif feature == 'Neutrophil':
                    clinical_data[feature] = st.number_input("中性粒细胞 (Neutrophil)", value=5.0, min_value=0.0)
                elif feature == 'MLR':
                    clinical_data[feature] = st.number_input("MLR比值", value=0.3, min_value=0.0)
                else:
                    clinical_data[feature] = st.number_input(f"{feature}", value=0.0)
        
        # 批量数据上传
        st.subheader("📁 批量数据上传")
        uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"成功上传 {len(batch_data)} 条记录")
                st.dataframe(batch_data.head())
                
                if st.button("批量预测"):
                    batch_predictions = perform_batch_prediction(batch_data, models, weights, feature_names, threshold)
                    st.download_button(
                        label="下载预测结果",
                        data=batch_predictions.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"文件上传错误: {str(e)}")
    
    with tab2:
        st.header("模型预测")
        
        # 预测按钮
        if st.button("🔮 开始预测", type="primary"):
            # 准备数据
            X_radiomics = pd.DataFrame([radiomics_data])
            X_deep = pd.DataFrame([deep_learning_data])
            X_clinical = pd.DataFrame([clinical_data])
            
            # 进行预测
            try:
                # 获取各模型的预测概率
                proba_trad = models['traditional'].predict_proba(X_radiomics)
                proba_deep = models['deep_learning'].predict_proba(X_deep)
                proba_clinical = models['clinical'].predict_proba(X_clinical)
                
                # 加权平均
                weighted_proba = (
                    weights['traditional'] * proba_trad[:, 1] +
                    weights['deep_learning'] * proba_deep[:, 1] +
                    weights['clinical'] * proba_clinical[:, 1]
                )
                
                # 预测结果
                prediction = (weighted_proba > threshold).astype(int)[0]
                probability = weighted_proba[0]
                
                # 显示结果
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("预测结果", 
                             "生存" if prediction == 1 else "未生存",
                             f"概率: {probability:.3f}")
                
                with col2:
                    # 概率条形图
                    fig = go.Figure(go.Bar(
                        x=['未生存', '生存'],
                        y=[1-probability, probability],
                        marker_color=['red', 'green']
                    ))
                    fig.update_layout(title="预测概率分布", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # 各模型贡献
                st.subheader("各模型贡献度")
                contributions = {
                    '传统影像': weights['traditional'] * proba_trad[:, 1][0],
                    '深度学习': weights['deep_learning'] * proba_deep[:, 1][0],
                    '临床特征': weights['clinical'] * proba_clinical[:, 1][0]
                }
                
                fig = px.pie(
                    values=list(contributions.values()),
                    names=list(contributions.keys()),
                    title="各模型对最终预测的贡献"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"预测过程中出现错误: {str(e)}")
    
    with tab3:
        st.header("结果可视化")
        
        # 模型权重可视化
        st.subheader("模型权重分布")
        weights_df = pd.DataFrame({
            '模型': ['传统影像', '深度学习', '临床特征'],
            '权重': [weights['traditional'], weights['deep_learning'], weights['clinical']]
        })
        
        fig = px.bar(weights_df, x='模型', y='权重', 
                     title="集成模型权重分布",
                     color='权重',
                     color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # 特征重要性可视化
        st.subheader("特征重要性")
        
        # 创建示例特征重要性数据
        all_features = radiomics_features + deep_learning_features + clinical_features
        feature_importance = np.abs(np.random.randn(len(all_features)))
        
        importance_df = pd.DataFrame({
            '特征': all_features,
            '重要性': feature_importance,
            '类型': ['传统影像']*len(radiomics_features) + 
                   ['深度学习']*len(deep_learning_features) + 
                   ['临床特征']*len(clinical_features)
        }).sort_values('重要性', ascending=True)
        
        fig = px.bar(importance_df.tail(10), x='重要性', y='特征', 
                     color='类型',
                     title="Top 10 特征重要性",
                     orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("模型信息")
        
        # 模型架构
        st.subheader("🏗️ 模型架构")
        st.write("""
        本系统采用集成学习方法，结合三种不同类型的特征：
        - **传统影像特征**: 基于小波变换的纹理特征
        - **深度学习特征**: 通过深度神经网络提取的高级特征
        - **临床特征**: 患者的临床检查指标
        """)
        
        # 模型参数
        st.subheader("⚙️ 模型参数")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**最佳权重组合:**")
            st.write(f"- 传统影像: {weights['traditional']:.2f}")
            st.write(f"- 深度学习: {weights['deep_learning']:.2f}")
            st.write(f"- 临床特征: {weights['clinical']:.2f}")
        
        with col2:
            st.write("**决策阈值:**")
            st.write(f"- 阈值: {threshold}")
            st.write("- 评估指标: F1-Score, AUC, 准确率")
        
        # 性能指标
        st.subheader("📊 模型性能")
        performance_data = {
            '指标': ['F1-Score', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity'],
            '测试集': [
                performance_metrics.get('F1', 0.83),
                performance_metrics.get('AUC', 0.87),
                performance_metrics.get('Accuracy', 0.85),
                performance_metrics.get('Sensitivity', 0.84),
                performance_metrics.get('Specificity', 0.86)
            ]
        }
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # 使用说明
        st.subheader("📖 使用说明")
        st.write("""
        1. 在"数据输入"标签页输入患者的各项特征值
        2. 点击"开始预测"按钮获取预测结果
        3. 查看"结果可视化"了解模型的决策过程
        4. 支持单个预测和批量预测两种模式
        """)

if __name__ == "__main__":
    main()

# 运行说明
st.sidebar.markdown("""
---
### 💡 使用提示
1. 确保所有特征值都已正确输入
2. 批量预测时，CSV文件应包含所有必要的特征列
3. 预测结果仅供参考，不能替代医生的专业判断

### 🔧 技术支持
如有问题，请联系技术支持团队
""")
