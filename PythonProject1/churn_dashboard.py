# churn_dashboard.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)

# -------------------------------
# 1. Dashboard Setup
# -------------------------------
st.set_page_config(layout="wide", page_title="Customer Churn Dashboard")
st.title("Customer Churn Prediction Dashboard")
st.write("Upload a CSV with a 'Churn' column (Yes/No) to see predictions and visualizations.")

# -------------------------------
# 2. File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded — using example CSV 'Telco-Customer-Churn.csv'")
    df = pd.read_csv("Telco-Customer-Churn.csv")

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
# 4. Load Model & Scaler
# -------------------------------
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")
X_scaled = scaler.transform(X)
y_proba = model.predict_proba(X_scaled)[:,1]

# -------------------------------
# 5. Threshold Slider (for Recall)
# -------------------------------
threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
y_pred = (y_proba >= threshold).astype(int)

# -------------------------------
# 6. Metrics
# -------------------------------
st.subheader("Metrics")
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

# -------------------------------
# 7. Confusion Matrix
# -------------------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# -------------------------------
# 8. Precision-Recall Curve
# -------------------------------
st.subheader("Precision-Recall Curve")
precisions, recalls, _ = precision_recall_curve(y, y_proba)
fig2, ax2 = plt.subplots()
ax2.plot(recalls, precisions, marker='.')
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve")
st.pyplot(fig2)

# -------------------------------
# 9. Feature Importance
# -------------------------------
st.subheader("Top Feature Importance")
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
})
feat_imp['abs_coef'] = feat_imp['coefficient'].abs()
feat_imp = feat_imp.sort_values(by='abs_coef', ascending=False).head(10)

fig3, ax3 = plt.subplots(figsize=(8,5))
sns.barplot(x='abs_coef', y='feature', data=feat_imp, palette='viridis', ax=ax3)
ax3.set_xlabel("Coefficient magnitude")
ax3.set_ylabel("Feature")
ax3.set_title("Top 10 Features Affecting Churn")
st.pyplot(fig3)

# -------------------------------
# 10. High-risk Customers Table
# -------------------------------
df_proc['pred_proba'] = y_proba
df_proc['pred_label'] = y_pred
st.subheader("High-risk Customers (predicted churn = 1)")
st.dataframe(df_proc[df_proc['pred_label']==1].sort_values('pred_proba', ascending=False).head(50))

st.download_button(
    "Download High-risk Customers CSV",
    df_proc[df_proc['pred_label']==1].to_csv(index=False),
    file_name="high_risk_customers.csv"
)
