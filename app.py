import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="產品評論情緒分析器", page_icon="📊", layout="centered")

st.title("📊 產品評論情緒分析系統")
st.write("請在下方輸入產品評論或使用心得，系統會模擬並輸出 1 至 5 等級的評分。")

# 使用 st.cache_resource 避免每次點擊按鈕都重新載入模型
@st.cache_resource
def load_model():
    # 使用中文電商微調之情緒分析模型
    return pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")

try:
    with st.spinner("正在載入情緒分析模型，請稍候..."):
        sentiment_pipeline = load_model()
except Exception as e:
    st.error(f"模型載入失敗: {e}")

def analyze_sentiment(text):
    # 分析輸入文字情緒，並將其信心分數精密映射至 1 到 5 顆星。
    results = sentiment_pipeline(text)[0]
    label = results['label']
    score = results['score']

    # 將正面/負面情緒機率對接到 1 到 5 星
    if 'LABEL_1' in label or 'positive' in label.lower():
        rating = round(3 + score * 2)
    else:
        rating = round(3 - score * 2)

    rating = max(1, min(5, rating))
    return rating

# 建立使用者文字輸入框
user_input = st.text_area("👉 請輸入您的評論內容：", placeholder="例如：這款產品真的很好用，出貨速度也很快！", height=150)

# 按鈕與分析觸發
if st.button("🚀 開始進行分析", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ 請先輸入文字內容再進行分析！")
    else:
        with st.spinner("計算中..."):
            star_rating = analyze_sentiment(user_input)

        # 呈現分析結果
        st.success("🎉 分析完成！")

        # 依星等設計不同顏色與排版
        stars = "⭐" * star_rating + "☆" * (5 - star_rating)
        st.markdown(f"### 預測評等: **{star_rating} 星**")
        st.markdown(f"### 視覺化星等: <span style='color: #FFD700; font-size: 24px;'>{stars}</span>", unsafe_allow_html=True)

        # 評語解釋
        explanations = {
            1: "🔴 極度負面評論：使用者對產品極為不滿。",
            2: "🟠 偏向負面評論：產品有明顯缺點或服務不佳。",
            3: "🟡 中立/普通評論：產品表現平平，無功無過。",
            4: "🟢 偏向正面評論：整體體驗良好，有些許小瑕疵但可接受。",
            5: "🔵 極度正面評論：使用者非常滿意，高度推薦。"
        }
        st.info(explanations[star_rating])