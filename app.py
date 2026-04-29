import streamlit as st
from predict import predict_sentiment

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.stTextArea textarea {
    border-radius: 12px;
    font-size: 18px;
}
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: 600;
}
.result-box {
    padding: 20px;
    border-radius: 14px;
    margin-top: 20px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>💬 Sentiment Analysis Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Analyze emotional tone in real time using a fine-tuned BERT model.</p>", unsafe_allow_html=True)

text = st.text_area("Enter text for sentiment analysis", height=180)

if st.button("Analyze Sentiment"):
    if text.strip():
        result = predict_sentiment(text)

        if "Positive" in result:
            color = "#14532d"
        else:
            color = "#7f1d1d"

        st.markdown(
            f"<div class='result-box' style='background:{color};'>{result}</div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter some text.")