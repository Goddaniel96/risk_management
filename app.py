import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Load real dataset ==========
@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv")
    df['DebtPerMonth'] = df['Credit amount'] / df['Duration']
    df['IsYoung'] = (df['Age'] < 30).astype(int)
    df['HasSaving'] = df['Saving accounts'].notnull().astype(int)
    df['Risk'] = df['Risk'].map({'Low': 0, 'Medium': 1, 'High': 2}) if df['Risk'].dtype == 'object' else df['Risk']
    return df

df = load_data()

# ========== Section 1: Risk Prediction Dashboard ==========
@st.cache_data
def train_model():
    model = XGBClassifier(max_depth=4, n_estimators=30, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    X = df[['Age', 'Job', 'Credit amount', 'Duration', 'DebtPerMonth', 'IsYoung', 'HasSaving']]
    y = df['Risk']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

st.title("📊 Risk Management Dashboard")
section = st.sidebar.radio("เลือกหน้า Dashboard:", ["🔮 ทำนายความเสี่ยงรายบุคคล", "📈 วิเคราะห์ภาพรวมตาม Purpose"])

if section == "🔮 ทำนายความเสี่ยงรายบุคคล":
    st.header("🔮 Risk Level Prediction")
    age = st.slider("อายุ (ปี)", 18, 70, 30)
    job = st.selectbox("ระดับอาชีพ (0-3)", sorted(df['Job'].unique()))
    credit = st.number_input("วงเงินกู้ (บาท)", min_value=1000, max_value=20000, value=8000)
    duration = st.number_input("ระยะเวลาผ่อน (เดือน)", min_value=6, max_value=60, value=24)
    is_young = 1 if age < 30 else 0
    has_saving = st.selectbox("มีบัญชีเงินฝากหรือไม่", ["ไม่มี", "มี"])
    has_saving = 1 if has_saving == "มี" else 0

    X_input = pd.DataFrame([{
        'Age': age,
        'Job': job,
        'Credit amount': credit,
        'Duration': duration,
        'DebtPerMonth': credit / duration,
        'IsYoung': is_young,
        'HasSaving': has_saving,
    }])

    X_input_scaled = scaler.transform(X_input)
    pred = model.predict(X_input_scaled)[0]
    pred_proba = model.predict_proba(X_input_scaled)[0]

    risk_label = ['Low', 'Medium', 'High']
    st.subheader("🔎 ผลการประเมินความเสี่ยง")
    st.metric(label="ระดับความเสี่ยง", value=risk_label[pred])
    st.progress(float(pred_proba[pred]))
    st.write("ความมั่นใจของโมเดล:")
    st.write({risk_label[i]: f"{proba:.2f}" for i, proba in enumerate(pred_proba)})

# ========== Section 2: Purpose Risk Overview ==========
elif section == "📈 วิเคราะห์ภาพรวมตาม Purpose":
    st.header("📈 วิเคราะห์ความเสี่ยงตามวัตถุประสงค์การขอกู้")
    risk_levels = ['Low', 'Medium', 'High']
    selected_purpose = st.selectbox("เลือกวัตถุประสงค์ในการขอกู้:", sorted(df['Purpose'].dropna().unique()))
    filtered_df = df[df['Purpose'] == selected_purpose]

    st.subheader(f"ระดับความเสี่ยงของลูกค้าที่ขอกู้เพื่อ '{selected_purpose}'")
    purpose_risk_counts = filtered_df['Risk'].map({0: 'Low', 1: 'Medium', 2: 'High'}).value_counts().reindex(risk_levels, fill_value=0)
    st.bar_chart(purpose_risk_counts)

    st.subheader("🔤 WordCloud ของวัตถุประสงค์ทั้งหมด")
    purpose_text = ' '.join(df['Purpose'].dropna().astype(str))
    wc = WordCloud(width=800, height=300, background_color='white').generate(purpose_text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    st.subheader("🔍 ความถี่ของการใช้แต่ละ Purpose")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='Purpose', order=df['Purpose'].value_counts().index, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("🧠 วิเคราะห์เชิงข้อความ")
    st.markdown("""
    - วัตถุประสงค์ที่พบว่าเสี่ยงสูงบ่อย: **business**, **retraining**
    - วัตถุประสงค์ทั่วไปที่มีความเสี่ยงต่ำ: **radio/TV**, **furniture**, **vacation**
    - การกู้เพื่อ 'education' หรือ 'repairs' มักอยู่ในระดับกลาง ขึ้นกับวงเงิน
    """)
