import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go

st.set_page_config(page_title="가짜 뉴스 탐지기", page_icon="📰", layout="centered")

@st.cache_resource
def load_assets():
    model = load_model('fake_news_lstm_model.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

try:
    model, tokenizer = load_assets()
    max_len = 500 # AI가 공부한 500단어 기준
except Exception as e:
    st.error("모델이나 토크나이저 파일을 찾을 수 없습니다.")
    st.stop()

st.title("📰 AI 기반 가짜 뉴스 탐지 대시보드")
st.markdown("딥러닝(LSTM) 모델을 활용하여 입력된 **영문 뉴스 기사**의 진위 여부를 판별합니다.")

user_input = st.text_area("검증할 뉴스 기사 본문을 입력하세요 (영문 기사만 판별 가능합니다):", height=200, 
                          placeholder="여기에 영어로 된 뉴스 기사를 복사해서 붙여넣으세요...")

if st.button("분석 시작", type="primary"):
    if len(user_input.strip()) < 10:
        st.warning("분석을 위해 충분한 길이의 텍스트를 입력해주세요.")
    else:
        with st.spinner("AI가 기사 문맥을 분석 중입니다..."):
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=max_len)
            prediction = model.predict(padded)[0][0]
            
            # --- 원본 데이터셋 기준 완벽 고정 (1=Fake, 0=Real) ---
            fake_prob = prediction * 100
            real_prob = (1 - prediction) * 100
            is_fake = prediction >= 0.5 
            
            st.divider()
            st.subheader("💡 AI 분석 결과")
            
            if is_fake:
                st.error(f"이 기사는 **가짜 뉴스(Fake News)**일 확률이 높습니다! (확률: {fake_prob:.1f}%)")
                color = "red"
                value = fake_prob
                title = "가짜 뉴스 확률"
            else:
                st.success(f"이 기사는 **진짜 뉴스(Real News)**일 확률이 높습니다! (확률: {real_prob:.1f}%)")
                color = "green"
                value = real_prob
                title = "진짜 뉴스 확률"
                
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': title, 'font': {'size': 24}},
                number = {'suffix': "%"},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "whitesmoke"}],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.info("※ 본 대시보드는 LSTM 알고리즘 기반의 확률적 예측이므로 100%의 정확도를 보장하지는 않습니다.")