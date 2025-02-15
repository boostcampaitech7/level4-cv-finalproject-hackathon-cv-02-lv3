import streamlit as st
import pandas as pd
import plotly.express as px

# 페이지 설정
st.set_page_config(
    page_title="LuckyVicky AI Solution",
    page_icon="🌟",
    layout="wide"
)

# 타이틀
st.title("📊 데이터 분석 결과")
st.write("전체 데이터와 개별 컬럼의 분포를 확인하세요!")
st.divider()

# 세션 상태에서 데이터프레임 불러오기
df = st.session_state.get('df')

if df is not None:
    # 전체 데이터 분석
    st.header("전체 데이터 분석")
    col1, col2 = st.columns([3, 1], border=True)

    with col1:
        st.subheader("데이터프레임")
        st.dataframe(df, height=400)

    with col2:
        st.subheader("데이터 요약")
        st.markdown(f"- 총 데이터 개수: **{len(df)}**")
        st.markdown(f"- 컬럼 수: **{len(df.columns)}**")
        # 데이터 용량
        mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB 단위로 변환
        st.markdown(f"- 데이터 용량: **{mem_usage:.2f} MB**")

    st.divider()

    # 개별 컬럼 분석
    st.header("개별 컬럼 분석")
    selected_column = st.selectbox("분석할 컬럼을 선택하세요:", df.columns)

    if selected_column:
        st.subheader(f"'{selected_column}' 분석")
        st.markdown(f"- 데이터 타입: **{df[selected_column].dtype}**")
        st.markdown(f"- 결측치 비율: **{df[selected_column].isnull().mean() * 100:.2f}%**")

        if pd.api.types.is_numeric_dtype(df[selected_column]):
            st.markdown(f"- 평균값: **{df[selected_column].mean():.2f}**")
            st.markdown(f"- 최대값: **{df[selected_column].max()}**")
            st.markdown(f"- 최소값: **{df[selected_column].min()}**")
        elif pd.api.types.is_object_dtype(df[selected_column]) or pd.api.types.is_categorical_dtype(df[selected_column]):
            st.markdown(f"- 카테고리 수: **{df[selected_column].nunique()}**")

        st.subheader("📊 시각화")
        col1, col2 = st.columns(2, border=True)

        with col1:
            st.markdown("**결측치 분포**")
            missing_df = pd.DataFrame({
                'Status': ['Missing', 'Non-Missing'],
                'Count': [df[selected_column].isnull().sum(), df[selected_column].notnull().sum()]
            })
            fig_pie = px.pie(missing_df, names='Status', values='Count', color='Status',
                             color_discrete_map={'Missing': 'red', 'Non-Missing': 'lightskyblue'})
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            if pd.api.types.is_numeric_dtype(df[selected_column]):
                st.markdown("**히스토그램**")
                fig_hist = px.histogram(df, x=selected_column, nbins=20, color_discrete_sequence=['blue'])
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.markdown("**카테고리 분포**")
                category_df = df[selected_column].value_counts().reset_index()
                category_df.columns = [selected_column, 'Count']
                fig_bar = px.bar(category_df, x=selected_column, y='Count', color_discrete_sequence=['blue'])
                st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.warning("CSV 파일이 업로드되지 않았습니다. Home 페이지로 돌아가 파일을 업로드하세요.")

# 페이지 전환 버튼
col1, col2 = st.columns((9,1))

with col1:
    if st.button("다시 제출하기"):
        st.session_state.pop('df', None)
        st.switch_page("Home.py")

with col2:
    if st.button("솔루션 시작하기"):
        st.switch_page("pages/2_AI_Solution.py")
