# risk_management
# 🛡️ Risk Management Dashboard with ML + NLP  
**ประเมินความเสี่ยงลูกค้า & วิเคราะห์พฤติกรรมการกู้เงินด้วย Machine Learning และ Natural Language Processing**

---

## 📌 Overview
ระบบ Dashboard แบบ Interactive ที่ผสานพลังของ  
🔎 **Machine Learning (XGBoost / Random Forest)**  
💬 **NLP (Text Analysis + WordCloud)**  
📊 **Visual Analytics**  
เพื่อช่วยวิเคราะห์ความเสี่ยงของลูกค้าและวัตถุประสงค์ในการขอกู้เงิน

---

## 🚀 Features

### 🔮 ทำนายระดับความเสี่ยงรายบุคคล
- กรอกข้อมูลลูกค้า → ได้ผลระดับความเสี่ยง (Low / Medium / High)
- แสดง Confidence Score + Probabilities
- Feature Engineering อัจฉริยะ (เช่น DebtPerMonth, IsYoung)

### 📈 วิเคราะห์ภาพรวมตามวัตถุประสงค์ (Purpose)
- Bar Chart เปรียบเทียบระดับความเสี่ยงของแต่ละ Purpose
- WordCloud แสดงคำยอดนิยม
- วิเคราะห์เชิงข้อความระบุแนวโน้มพฤติกรรมเสี่ยง

---

## 🧠 Technologies Used
- **Frontend**: Streamlit
- **ML**: XGBoost, Random Forest (scikit-learn)
- **NLP**: WordCloud, Text Aggregation
- **Data**: Pandas, NumPy, Matplotlib, Seaborn

---

## 🛠️ Getting Started

1. ติดตั้งไลบรารีที่จำเป็น
```bash
pip install -r requirements.txt
