import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    st.divider()

    try:
        # 세션 상태에서 데이터프레임 불러오기
        df = st.session_state.df

        # 1. 전체 데이터 분석
        st.markdown('<div class="col1">', unsafe_allow_html=True) 
        st.header("전체 데이터 분석")

        # 레이아웃 나누기
        col1, col2 = st.columns([3, 1])  # 왼쪽 3: 오른쪽 1 비율 설정

        with col1:
            st.write("### 데이터프레임")
            st.dataframe(df, height=400)
    
        with col2:
            st.write("### 데이터 요약")
            st.write(f"- 총 데이터 개수: {len(df)}")
            st.write(f"- 컬럼 수: {len(df.columns)}")

        st.divider()


        # 2. 열별 상세 분석
        st.markdown('<div class="col2">', unsafe_allow_html=True)
        st.header("개별 column 분석")
        column = st.selectbox("분석할 column을 선택하세요:", df.columns)
        # 선택된 열에 대한 분석

        # int 형 혹은 float형 일때
        if pd.api.types.is_integer_dtype(df[column]) or pd.api.types.is_float_dtype(df[column]):

            # 데이터 기본 분석
            st.write(f"### '{column}' 분석")
            st.write(f'- Data type: {df[column].dtype}')
            st.write(f"- 평균값: {df[column].mean()}")
            st.write(f"- 최댓값: {df[column].max()}")
            st.write(f"- 최솟값: {df[column].min()}")
            st.write(f"- 결측치 비율: {df[column].isnull().mean() * 100:.2f}%")
            missing_ratio = df[column].isnull().mean()
            non_missing_ratio = 1 - missing_ratio

            # 데이터 시각화
            st.write("### Visualization")

            # 레이아웃 나누기
            col_1, col_2 = st.columns([1, 1])  # 왼쪽 1: 오른쪽 1 비율 설정
            # pie chart
            with col_1:
                st.write('missing value')
                sns.set_style("whitegrid")  # Seaborn 스타일 설정

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.pie(
                    [missing_ratio, non_missing_ratio],
                    labels=['missing', 'non_missing'],
                    colors=['#FF0000', '#66b3ff'],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops={'edgecolor': 'black'}
                )
                plt.legend()
                ax.axis('equal')  # 원형 유지
                st.pyplot(fig)
            
            # histogram
            with col_2:
                st.write("Histogram")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[column].dropna(), bins=20, color="blue", ax=ax)
                st.pyplot(fig)
                
        
        # bool 형일때
        elif pd.api.types.is_bool_dtype(df[column]):
            # 데이터 기본 분석
            st.write(f"### '{column}' 분석")
            st.write(f'- Data type: {df[column].dtype}')
            st.write(f"- 결측치 비율: {df[column].isnull().mean() * 100:.2f}%")
            missing_ratio = df[column].isnull().mean()
            non_missing_ratio = 1 - missing_ratio

            # 데이터 시각화
            st.write("### Visualization")

            # 레이아웃 나누기
            col_1, col_2 = st.columns([1, 1])  # 왼쪽 1: 오른쪽 1 비율 설정
            # pie chart
            with col_1:
                st.write('missing value')
                sns.set_style("whitegrid")  # Seaborn 스타일 설정

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.pie(
                    [missing_ratio, non_missing_ratio],
                    labels=['missing', 'non_missing'],
                    colors=['#FF0000', '#66b3ff'],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops={'edgecolor': 'black'}
                )
                plt.legend()
                ax.axis('equal')  # 원형 유지
                st.pyplot(fig)
            
            # bar chart
            with col_2:
                st.write("bar chart")
                st.bar_chart(df[column].dropna(), facecolor="#0E1117")

        # object or category형 일때
        elif pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):

            # 레이아웃 나누기
            col_11, col_22 = st.columns([1, 2])  # 왼쪽 1: 오른쪽 1 비율 설정
            
            
            # 데이터 기본 분석
            with col_11:
                st.write(f"### '{column}' 분석")
                st.write(f'- Data type: {df[column].dtype}')
                st.write(f'- category 수: {df[column].nunique()}')
                st.write(f"- 결측치 비율: {df[column].isnull().mean() * 100:.2f}%")
                missing_ratio = df[column].isnull().mean()
                non_missing_ratio = 1 - missing_ratio

            # object or category에 뭐가 있는지 너무 많으면 상위 4개+other로 표시
                st.write("### Categories")
                length = len(df[column])
                na_count = df[column].isna().sum()
                if len(df[column].unique())>4:
                    top_values = df[column].value_counts().head(4)
                    other_count = df[column].value_counts()[4:].sum()

                    # 상위 4개와 기타 데이터 결합
                    data = pd.concat([top_values, pd.Series({'Other': other_count, 'NA': na_count})])
                    
                    # DataFrame 생성
                    result_df = pd.DataFrame({
                        'Category': data.index,
                        'Count': data.values,
                        'Percentage': (data.values / length * 100).round(2)  # 퍼센트 계산
                    })
                    
                    # DataFrame 표시
                    st.dataframe(result_df)

                else:
                    top_values = df[column].value_counts().head()
                    
                    # na 항목도 추가
                    data = pd.concat([top_values, pd.Series({'NA': na_count})])
                    # DataFrame 생성
                    result_df = pd.DataFrame({
                        'Category': data.index,
                        'Count': data.values,
                        'Percentage': (data.values / length * 100).round(2)  # 퍼센트 계산
                    })

                    # DataFrame 표시
                    st.dataframe(result_df)

            with col_22:
                #  결측치 비율 파이 차트
                st.write("### Missing Value Visualization")
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.pie(
                    [missing_ratio, non_missing_ratio],
                    labels=['missing', 'non_missing'],
                    colors=['#FF0000', '#66b3ff'],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops={'edgecolor': 'black'}
                )
                ax.legend(loc='lower right',fontsize=6)

                ax.axis('equal')  # 원형 비율 유지
                plt.tight_layout()  # 여백 최소화
                st.pyplot(fig,bbox_inches="tight", use_container_width=False)


        
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
    # "다시 제출하기"와 "솔루션 시작하기" 버튼 추가
    col1, col2 = st.columns(2)

    with col1:
        if st.button("다시 제출하기"):
            st.session_state.uploaded_file = None
            st.rerun()

    with col2:
        if st.button("솔루션 시작하기"):
            # 솔루션 페이지로 이동
            st.experimental_set_query_params(page="solution")