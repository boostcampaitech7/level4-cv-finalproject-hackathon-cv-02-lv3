import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
from Search.search import search  # 🔥 search 함수 import 추가

# 페이지 설정
st.set_page_config(
    page_title="LuckyVicky AI Solution",
    page_icon="🌟",
    layout="wide"
)

# 타이틀
st.title("🖥️ AI 솔루션 최종 결과")
st.write("전체 데이터를 활용한 최적화 결과입니다!")
st.write("최적화에는 약 30분 정도 소요될 수 있습니다. 최종 결과를 확인하고 데이터를 다운로드하세요.")
st.divider()

if st.session_state.train_score:
    # 세션 데이터 불러오기
    model = st.session_state.model
    search_x = st.session_state.search_x
    search_y = st.session_state.search_y
    X = st.session_state.X
    y = st.session_state.y

    # 최적화 결과가 이미 저장되어 있으면 search() 실행하지 않음
    if "optimal_solutions_df" not in st.session_state:
        with st.spinner('🔄 전체 데이터 최적화 진행 중입니다...(약 30분 소요 예정)'):
            _, optimal_solutions_df = search(X.head(200), y.head(200), model, search_x, search_y)
            st.session_state.optimal_solutions_df = optimal_solutions_df  # 결과 저장

    optimal_solutions_df = st.session_state.optimal_solutions_df  # 저장된 값 사용

    # CSV 변환 함수
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df(optimal_solutions_df)

    # 최적화 결과 시각화 (옵션)
    st.subheader("📈 최적화 결과 요약")
    # 데이터 로드
    X = st.session_state.X
    y = st.session_state.y
    search_x = st.session_state.search_x
    search_y = st.session_state.search_y

    def add_margin(y_min, y_max, margin_ratio=0.1):
        margin = (y_max - y_min) * margin_ratio
        return y_min - margin, y_max + margin

    original_col = list(search_y.keys())[0]
    solution_col = f"solution_{original_col}"

    chart_data = pd.concat([optimal_solutions_df[['y']].rename(columns={'y': solution_col}), y.head(200)], axis=1)
    original_mean = chart_data[original_col].mean()
    optimized_mean = chart_data[solution_col].mean()
    percentage_change = ((optimized_mean - original_mean) / abs(original_mean)) * 100

    y_min, y_max = add_margin(chart_data.min().min(), chart_data.max().max())

    fig = px.line(chart_data, labels={'index': 'Index', 'value': original_col}, title=f"Optimized vs Original {original_col}")
    fig.update_yaxes(range=[y_min, y_max])
    st.plotly_chart(fig)
    st.write(f"**{original_col}의 변화율:** {percentage_change:.2f}%")

    for i in search_x.keys():
        solution_col = f"solution_{i}"
        chart_data = pd.concat([optimal_solutions_df[[i]].rename(columns={i: solution_col}), X[i].head(200)], axis=1)
        original_mean = chart_data[i].mean()
        optimized_mean = chart_data[solution_col].mean()
        percentage_change = ((optimized_mean - original_mean) / abs(original_mean)) * 100

        y_min, y_max = add_margin(chart_data.min().min(), chart_data.max().max())

        fig = px.line(chart_data, labels={'index': 'Index', 'value': i}, title=f"Optimized vs Original {i}")
        fig.update_yaxes(range=[y_min, y_max])
        st.plotly_chart(fig)
        st.write(f"**{i}의 변화율:** {percentage_change:.2f}%")


    # 다운로드 버튼
    st.success("최적화가 완료되었습니다! 데이터를 다운로드하세요.")
    st.download_button(
        label="📥 최적화된 데이터 CSV 다운로드",
        data=csv,
        file_name="optimized_solution.csv",
        mime="text/csv",
    )



    st.divider()

    # 이동 버튼
    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ 이전 페이지"):
            st.switch_page("pages/3_Model_Training.py")

    with col2:
        # 홈으로 돌아가기 버튼
        if st.button("🏠 홈으로 돌아가기"):
            st.switch_page("Home.py")

else:
    st.warning("모델을 먼저 훈련 시켜주세요!")