import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Section 1: Risk Prediction Dashboard ==========
@st.cache_data
def load_model():
    model = XGBClassifier(max_depth=4, n_estimators=30, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    df = pd.DataFrame({
        'Age': np.random.randint(18, 70, 500),
        'Job': np.random.randint(0, 4, 500),
        'Credit amount': np.random.randint(1000, 20000, 500),
        'Duration': np.random.randint(6, 48, 500),
    })
    df['DebtPerMonth'] = df['Credit amount'] / df['Duration']
    df['IsYoung'] = (df['Age'] < 30).astype(int)
    df['HasSaving'] = np.random.randint(0, 2, 500)
    def create_risk(row):
        if row['Credit amount'] > 10000 and row['Duration'] > 24:
            return 2
        elif row['Credit amount'] > 5000:
            return 1
        else:
            return 0
    y = df.apply(create_risk, axis=1)
    X = df[['Age', 'Job', 'Credit amount', 'Duration', 'DebtPerMonth', 'IsYoung', 'HasSaving']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = load_model()

st.title("📊 Risk Management Dashboard")
section = st.sidebar.radio("เลือกหน้า Dashboard:", ["🔮 ทำนายความเสี่ยงรายบุคคล", "📈 วิเคราะห์ภาพรวมตาม Purpose"])

if section == "🔮 ทำนายความเสี่ยงรายบุคคล":
    st.header("🔮 Risk Level Prediction")
    age = st.slider("อายุ (ปี)", 18, 70, 30)
    job = st.selectbox("ระดับอาชีพ (0-3)", [0, 1, 2, 3])
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
    purposes = ['car', 'radio/TV', 'furniture', 'education', 'business', 'vacation', 'repairs', 'retraining']
    risk_levels = ['Low', 'Medium', 'High']
    n = 300
    df = pd.DataFrame({
        'Purpose': np.random.choice(purposes, n),
        'Credit amount': np.random.randint(1000, 20000, n),
        'Duration': np.random.randint(6, 48, n),
    })
    def simulate_risk(row):
        if row['Purpose'] in ['business', 'retraining'] or (row['Credit amount'] > 12000 and row['Duration'] > 24):
            return 'High'
        elif row['Credit amount'] > 7000:
            return 'Medium'
        else:
            return 'Low'
    df['Risk'] = df.apply(simulate_risk, axis=1)
    selected_purpose = st.selectbox("เลือกวัตถุประสงค์ในการขอกู้:", sorted(df['Purpose'].unique()))
    filtered_df = df[df['Purpose'] == selected_purpose]
    st.subheader(f"ระดับความเสี่ยงของลูกค้าที่ขอกู้เพื่อ '{selected_purpose}'")
    risk_counts = filtered_df['Risk'].value_counts().reindex(risk_levels, fill_value=0)
    st.bar_chart(risk_counts)

    st.subheader("🔤 WordCloud ของวัตถุประสงค์ทั้งหมด")
    purpose_text = ' '.join(df['Purpose'])
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
