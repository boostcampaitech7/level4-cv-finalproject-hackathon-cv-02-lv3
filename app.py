import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 페이지 레이아웃 설정
st.set_page_config(layout="wide", page_title="AI solution", page_icon="📊")

# 상태를 저장할 Session State 초기화
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
    st.session_state.uploaded_file = None
    st.session_state.df = None  # 데이터프레임을 저장할 새로운 상태 변수

# 로그인 창처럼 구현된 파일 업로드 화면
if not st.session_state.is_authenticated:
    st.title("Prescript AI solution")
    st.write("분석하고 싶은 CSV 파일을 제출하세요.")
    
    # 파일 업로드 위젯
    uploaded_file = st.file_uploader("",type="csv")

    # 파일 업로드 후 로그인 상태로 전환
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.is_authenticated = True
        # 데이터프레임을 세션 상태에 저장
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("파일 업로드 성공! 분석 화면으로 이동합니다.")
        st.rerun()  # 화면 갱신
else:
    # 원래 화면: 데이터프레임과 분석 결과 표시
    st.title("📊 데이터 분석 결과")

    # 레이아웃 나누기
    col1, col2 = st.columns([4, 3])  # 왼쪽 4: 오른쪽 3 비율 설정

    try:
        # 세션 상태에서 데이터프레임 불러오기
        df = st.session_state.df

        # 왼쪽 열: 전체 데이터 분석
        with col1:
            st.header("전체 데이터 분석")
            st.write("### 데이터프레임")
            st.dataframe(df, height=400)

            st.write("### 데이터 요약")
            st.write(f"- 총 데이터 개수: {len(df)}")
            st.write(f"- 컬럼 수: {len(df.columns)}")

        # 오른쪽 열: 열별 상세 분석
        with col2:
            st.header("개별 columns 분석")
            column = st.selectbox("분석할 column을 선택하세요:", df.columns)
            # 선택된 열에 대한 분석

            # int 형일때
            if pd.api.types.is_integer_dtype(df[column]):

                # 데이터 기본 분석
                st.write(f"### '{column}' 분석")
                st.write(f'- Data type: int')
                st.write(f"- 평균값: {df[column].mean()}")
                st.write(f"- 최댓값: {df[column].max()}")
                st.write(f"- 최솟값: {df[column].min()}")
                st.write(f"- 결측치 비율: {df[column].isnull().mean() * 100:.2f}%")
                missing_ratio = df[column].isnull().mean()
                non_missing_ratio = 1 - missing_ratio

                # 데이터 시각화
                # pie chart
                st.write("### 결측치 시각화")
                fig, ax = plt.subplots()
                ax.pie([missing_ratio, non_missing_ratio], labels=['missing', 'non_missing'], 
                colors=['#ff9999', '#66b3ff'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # 원형 유지
                st.pyplot(fig)
                
                # histogram
                st.write("### 히스토그램")
                if df[column].isna().sum() != 0:
                    st.write("결측치 제거 후 분석!")
                st.bar_chart(df[column].dropna())
            
            # float 형일때
            elif pd.api.types.is_float_dtype(df[column]):
                st.write(f"### '{column}' 열의 분석")
                st.write(f"- 평균값: {df[column].mean()}")
                st.write(f"- 최댓값: {df[column].max()}")
                st.write(f"- 최솟값: {df[column].min()}")
                st.write(f"- 결측치 비율: {df[column].isnull().mean() * 100:.2f}%")
                st.write("### 히스토그램")
                if df[column].isna():
                    st.write("결측치 제거 후 분석!")
                st.bar_chart(df[column].dropna())
            
            # bool 형일때
            elif pd.api.types.is_bool_dtype(df[column]):
                st.write(f"### '{column}' 열의 분석")
                st.write(f"- 평균값: {df[column].mean()}")
                st.write(f"- 최댓값: {df[column].max()}")
                st.write(f"- 최솟값: {df[column].min()}")
                st.write(f"- 결측치 비율: {df[column].isnull().mean() * 100:.2f}%")
                st.write("### 히스토그램")
                if df[column].isna():
                    st.write("결측치 제거 후 분석!")
                st.bar_chart(df[column].dropna())

            # object 형일때
            elif pd.api.types.is_object_dtype(df[column]):
                st.write(f"### '{column}' 열의 분석")
                st.write(f"- 평균값: {df[column].mean()}")
                st.write(f"- 최댓값: {df[column].max()}")
                st.write(f"- 최솟값: {df[column].min()}")
                st.write(f"- 결측치 비율: {df[column].isnull().mean() * 100:.2f}%")
                st.write("### 히스토그램")
                if df[column].isna():
                    st.write("결측치 제거 후 분석!")
                st.bar_chart(df[column].dropna())

            # category 형일때
            elif pd.api.types.is_string_dtype(df[column]):
                st.write(f"### '{column}' 열의 분석")
                st.write(f"- 평균값: {df[column].mean()}")
                st.write(f"- 최댓값: {df[column].max()}")
                st.write(f"- 최솟값: {df[column].min()}")
                st.write(f"- 결측치 비율: {df[column].isnull().mean() * 100:.2f}%")
                st.write("### 히스토그램")
                if df[column].isna():
                    st.write("결측치 제거 후 분석!")
                st.bar_chart(df[column].dropna())
            
            # datetime 일때 (pd.api.types.is_datatime64_dtype)
            else:
                st.write(f"### '{column}' 열의 분석")
                st.write(f"- 평균값: {df[column].mean()}")
                st.write(f"- 최댓값: {df[column].max()}")
                st.write(f"- 최솟값: {df[column].min()}")
                st.write(f"- 결측치 비율: {df[column].isnull().mean() * 100:.2f}%")
                st.write("### 히스토그램")
                if df[column].isna():
                    st.write("결측치 제거 후 분석!")
                st.bar_chart(df[column].dropna())


    except Exception as e:
        st.error(f"데이터 분석 중 오류가 발생했습니다: {e}")


    # "다시 제출하기" 버튼
    st.markdown("---")
    if st.button("다시 제출하기"):
        st.session_state.is_authenticated = False
        st.session_state.uploaded_file = None
        st.rerun()  # 화면 갱신