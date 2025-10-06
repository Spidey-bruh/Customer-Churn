# churn_dashboard_v4.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Dashboard Setup
# -------------------------------
st.set_page_config(layout="wide", page_title="Customer Churn Dashboard")
st.markdown("<h1 style='text-align:center; color: darkblue;'>💼 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)

# ---- User Instructions ----
st.markdown("""
**How to use this dashboard:**
1. Upload a CSV file with a 'Churn' column containing 'Yes' or 'No'.
2. Use the sidebar filters to select customer segments.
3. Adjust the prediction threshold to see changes in metrics.
4. Explore tabs for metrics, feature importance, and high-risk customers.
5. Optionally, tune Random Forest hyperparameters in the sidebar and watch metrics update.
6. Download the filtered high-risk customers CSV if needed.
""")

# -------------------------------
# 2. File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded — using example CSV 'Telco-Customer-Churn.csv'")
    df = pd.read_csv("Telco-Customer-Churn.csv")

df_original = df.copy()

# -------------------------------
# 3. Preprocessing
# -------------------------------
def preprocess(df):
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'Churn']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

df_proc = preprocess(df)
X = df_proc.drop('Churn', axis=1)
y = df_proc['Churn']

# -------------------------------
# 4. Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 5. Handle imbalance
# -------------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# -------------------------------
# 6. Sidebar Filters
# -------------------------------
st.sidebar.header("Filter Customers")
filters = {}
for col in ['gender', 'SeniorCitizen', 'Partner', 'Dependents']:
    if col in df_original.columns:
        filters[col] = st.sidebar.multiselect(f"{col}", options=df_original[col].unique(), default=df_original[col].unique())

threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

# -------------------------------
# 7. Hyperparameter Tuning
# -------------------------------
st.sidebar.header("Random Forest Hyperparameters")
n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 200, step=50)
max_depth = st.sidebar.slider("Maximum Depth", 2, 20, 10)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
min_samples_leaf = st.sidebar.slider("Min Samples per Leaf", 1, 5, 1)

rf_model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_res, y_res)
y_proba = rf_model.predict_proba(X_scaled)[:,1]
y_pred = (y_proba >= threshold).astype(int)

# -------------------------------
# 8. Apply Filters
# -------------------------------
df_proc['pred_proba'] = y_proba
df_proc['pred_label'] = y_pred
high_risk = df_proc[df_proc['pred_label']==1].copy()

for col, values in filters.items():
    if col in df_original.columns:
        high_risk = high_risk[df_original[col].isin(values)]

# -------------------------------
# 9. Tabs Layout
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Metrics & Charts", "📈 Feature Importance", "⚠️ High-risk Customers"])

# ---- Metrics & Charts ----
with tab1:
    st.markdown("### Key Metrics")
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc = roc_auc_score(y, y_proba)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("ROC AUC", f"{roc:.2f}")

    col_cm, col_pr = st.columns(2)

    with col_cm:
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', xticklabels=["No","Yes"], yticklabels=["No","Yes"], cbar=False, ax=ax)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.tick_params(labelsize=8)
        st.pyplot(fig)

    with col_pr:
        st.markdown("### Precision-Recall Curve")
        precisions, recalls, _ = precision_recall_curve(y, y_proba)
        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax2.plot(recalls, precisions, marker='.', color='darkgreen')
        ax2.fill_between(recalls, precisions, alpha=0.2, color='green')
        ax2.set_xlabel("Recall", fontsize=9)
        ax2.set_ylabel("Precision", fontsize=9)
        ax2.set_title("Precision-Recall Curve", fontsize=11)
        ax2.tick_params(labelsize=8)
        st.pyplot(fig2)

# ---- Feature Importance ----
with tab2:
    st.markdown("### Top 10 Feature Importance")
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
    feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(10)

    fig3, ax3 = plt.subplots(figsize=(6,4))
    sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis', ax=ax3)
    ax3.set_xlabel("Importance", fontsize=9)
    ax3.set_ylabel("Feature", fontsize=9)
    ax3.set_title("Top 10 Features Affecting Churn", fontsize=11)
    ax3.tick_params(labelsize=8)
    st.pyplot(fig3)

# ---- High-risk Customers ----
with tab3:
    st.markdown("### High-risk Customers (Filtered)")
    st.dataframe(high_risk.head(50).style.applymap(
        lambda x: 'background-color : #ffcccc' if isinstance(x,(int,float)) and x>0.5 else '', subset=['pred_proba']))
    st.download_button(
        "Download Filtered High-risk Customers CSV",
        high_risk.to_csv(index=False),
        file_name="high_risk_customers.csv"
    )
