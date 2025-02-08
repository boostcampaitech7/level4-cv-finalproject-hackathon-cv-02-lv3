import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from aisolution import aisolution
from sklearn.inspection import PartialDependenceDisplay
from regplot import partial_dependence_with_error
from search import search
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE


@st.dialog("진행 불가")
def vote():
    st.write("속성을 지정하지 않아 다음 단계로 진행할 수 없습니다.")
    st.write("반드시 지정해주세요!")
    if st.button("다시 지정하기"):
        st.rerun()

@st.dialog("우선 순위를 정해주세요!")
def prior(option, opt):
    st.write("목표가 2개 이상이기 때문에 우선순위를 정해주세요!")

    options = [option]+opt
    selection = st.pills("Directions", options, selection_mode="multi")
    st.markdown(f"Your selected options:  \n {[f'{i+1}순위 : {j}'for i,j in enumerate(selection)]}.")

    if st.button("submit"):
        for i,j in enumerate(options):
            if j in st.session_state.search_x.keys():
                st.session_state.search_x[j]['순위'] = i+1
            
            else:
                st.session_state.search_y[j]['순위'] = i+1
        st.session_state.page = "train"  # train page로 넘어가기
        st.rerun()

@st.dialog('solution 진행 중')
def train(X, y, search_x, search_y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test

    with st.spinner('분석 및 최적화 진행 중입니다..(약 10분 소요 예정)'):
        train_score,test_score,train_time,model=aisolution(X_train=X_train, X_test=X_test, y_test=y_test, y_train=y_train,
                                                           task_type=st.session_state.type)

        # 50개의 데이터로 예시를 들어 준다.
        elapsed_time, optimal_solutions_df = search(X.head(50), y.head(50), model, search_x,search_y)

    st.session_state.train_score=train_score
    st.session_state.test_score=test_score
    st.session_state.train_time=train_time
    st.session_state.opt_time=elapsed_time
    st.session_state.df2=optimal_solutions_df
    st.session_state.model=model


# IQR을 이용한 이상치 제거 함수
def remove_outliers_iqr(df, option, method):
    if method == "제거하기":
        Q1 = df[option].quantile(0.25)  # 1사분위수 (Q1)
        Q3 = df[option].quantile(0.75)  # 3사분위수 (Q3)
        IQR = Q3 - Q1  # IQR 계산
        lower_bound = Q1 - 1.5 * IQR  # 이상치 하한값
        upper_bound = Q3 + 1.5 * IQR  # 이상치 상한값

        # 이상치가 아닌 데이터만 선택
        filtered_df = df[(df[option] >= lower_bound) & (df[option] <= upper_bound)].reset_index(drop=True)
        return filtered_df
    
    else:
        return df

def remove_na(df, option, method):
    df = df.copy()  # ✅ 원본을 보존하기 위해 복사 (중요!)

    if method == "관련 행 제거하기":
        return df.dropna(subset=[option]).reset_index(drop=True)  # ✅ 해당 열에서 결측치가 있는 행 제거

    elif method == "평균으로 채우기":
        mean_value = df[option].mean()  # 평균값 계산
        return df.fillna({option: mean_value})  # ✅ 특정 열만 채우기

    elif method == "0으로 채우기":
        return df.fillna({option: 0})  # ✅ 특정 열만 0으로 채우기

    elif method == "최빈값으로 채우기":
        mode_value = df[option].mode()[0]  # ✅ 최빈값 추출
        return df.fillna({option: mode_value})  # ✅ 특정 열만 채우기 (inplace=False)

    else:
        return df

def one_hot(df):
    for i in df.columns:
        if pd.api.types.is_string_dtype(df[i]) or pd.api.types.is_object_dtype(df[i]):
            if len(df[i])!=len(df[i].unique()):
                df = pd.get_dummies(df, columns=[i], drop_first=True)

    return df

# 페이지 레이아웃 설정
st.set_page_config(layout="wide", page_title="AI Solution", page_icon="📊")

# 상태를 저장할 page 초기화
if "page" not in st.session_state:
    st.session_state.page = False
    st.session_state.uploaded_files = None
    st.session_state.df = None  # 데이터프레임을 저장할 상태 변수

# 로그인 창처럼 구현된 파일 업로드 화면
if not st.session_state.page:
    st.title("Prescript AI Solution")
    st.write("분석하고 싶은 CSV 파일을 제출하세요.")

    # ✅ 여러 개의 파일 업로드 가능
    uploaded_files = st.file_uploader("CSV 파일을 업로드하세요", type="csv", accept_multiple_files=True)

    # ✅ 파일이 여러 개 업로드되었을 경우
    if uploaded_files and len(uploaded_files) > 1:
        merge_option = st.radio("파일을 합치는 방법을 선택하세요:", ("행으로 합치기", "열로 합치기"))

        if st.button("파일 병합 및 분석 시작"):
            try:
                dfs = [pd.read_csv(file) for file in uploaded_files]  # 모든 CSV 읽기
                
                if merge_option == "행으로 합치기":
                    # ✅ 모든 데이터프레임의 열 개수 및 열 이름이 동일한지 확인
                    columns_set = {tuple(df.columns) for df in dfs}
                    if len(columns_set) > 1:
                        st.error("오류: 모든 CSV 파일의 열 개수가 같아야 합니다. 다시 업로드해주세요.")
                        st.stop()

                    # ✅ 행으로 합치기
                    merged_df = pd.concat(dfs, axis=0, ignore_index=True)

                else:  # "열으로 합치기"
                    # ✅ 모든 데이터프레임의 행 개수가 동일한지 확인
                    row_counts = {df.shape[0] for df in dfs}
                    if len(row_counts) > 1:
                        st.error("오류: 모든 CSV 파일의 행 개수가 같아야 합니다. 다시 업로드해주세요.")
                        st.stop()

                    # ✅ 열으로 합치기
                    merged_df = pd.concat(dfs, axis=1)

                # ✅ 병합된 데이터프레임 저장 후 분석 화면으로 이동
                st.session_state.uploaded_files = uploaded_files
                st.session_state.df = merged_df
                st.session_state.page = "analysis"
                st.success("파일 병합 완료! 분석 화면으로 이동합니다.")
                st.rerun()

            except Exception as e:
                st.error(f"파일 병합 중 오류 발생: {e}")

    # ✅ 단일 파일 업로드 시 기존 방식 유지
    elif uploaded_files and len(uploaded_files) == 1:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.df = pd.read_csv(uploaded_files[0])
        st.session_state.page = "analysis"
        st.success("파일 업로드 성공! 분석 화면으로 이동합니다.")
        st.rerun()


# page - 데이터 eda 화면
elif st.session_state.page=="analysis":
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
        col1, col2 = st.columns([3, 1],border=True)  # 왼쪽 3: 오른쪽 1 비율 설정

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

            # 레이아웃 나누기 (비율 유지)
            col_1, col_2 = st.columns(2)  

            # ✅ Pie Chart (Missing Values)
            with col_1:
                st.write('Missing Value Distribution')

                missing_ratio = df[column].isna().sum()
                non_missing_ratio = df[column].notna().sum()

                pie_df = pd.DataFrame({
                    'Category': ['Missing', 'Non-Missing'],
                    'Count': [missing_ratio, non_missing_ratio]
                })

                fig_pie = px.pie(
                    pie_df,
                    names='Category',
                    values='Count',
                    color='Category',
                    color_discrete_map={'Missing': '#FF0000', 'Non-Missing': '#66b3ff'},
                    title="Missing Values"
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)

            # ✅ Histogram
            with col_2:
                st.write("Histogram of Selected Column")

                fig_hist = px.histogram(
                    df, 
                    x=column, 
                    nbins=20, 
                    title=f"Histogram of {column}",
                    color_discrete_sequence=["blue"]
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
                
        
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

            # 레이아웃 나누기 (비율 유지)
            col_1, col_2 = st.columns(2)  

            # ✅ Pie Chart (Missing Values)
            with col_1:
                st.write('Missing Value Distribution')

                missing_ratio = df[column].isna().sum()
                non_missing_ratio = df[column].notna().sum()

                pie_df = pd.DataFrame({
                    'Category': ['Missing', 'Non-Missing'],
                    'Count': [missing_ratio, non_missing_ratio]
                })

                fig_pie = px.pie(
                    pie_df,
                    names='Category',
                    values='Count',
                    color='Category',
                    color_discrete_map={'Missing': '#FF0000', 'Non-Missing': '#66b3ff'},
                    title="Missing Values"
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # bar chart
            with col_2:
                st.write("bar chart")
                st.bar_chart(df[column].dropna(), facecolor="#0E1117")

        # object or category형 일때
        elif pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):

            # 레이아웃 나누기
            col_11, col_22 = st.columns([1, 2], border=True)  # 왼쪽 1: 오른쪽 1 비율 설정
            
            
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
                # 📌 결측치 데이터 준비
                missing_data = pd.DataFrame({
                    "Category": ["Missing", "Non-Missing"],
                    "Ratio": [missing_ratio, non_missing_ratio]
                })
                st.write("### Missing Value Visualization")
                # 📌 Plotly Pie Chart 생성
                fig = px.pie(
                    missing_data,
                    names="Category",
                    values="Ratio",
                    #title="Missing Value Visualization",
                    color="Category",
                    color_discrete_map={"Missing": "#FF0000", "Non-Missing": "#66b3ff"},  # 기존 색상 유지
                    hole=0.3  # 도넛 차트 스타일 (완전 원형이면 제거 가능)
                )

                # 📌 Streamlit에 출력
                st.plotly_chart(fig, use_container_width=True)


        
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

    # 레이아웃 나누기
    col1, col2 = st.columns((9,1))

    with col1:
        if st.button("다시 제출하기"):
            st.session_state.uploaded_file = None
            st.session_state.page = False  # 인증 상태 초기화
            st.session_state.df=None
            st.rerun()  

    with col2:
        if st.button("솔루션 시작하기"):
            st.session_state.page="solution"
            st.rerun()




## page - solution
elif st.session_state.page=="solution":
    df=st.session_state.df
    # 제목 정하기
    st.title("💊AI 솔루션")
    st.write("진행하기 전에 분석하고 싶은 feature와 목표를 설정하세요!")
    st.divider()


    # 분석하고 싶은 feature와 목표 정하기

    # output 속성 정하기
    # 범주형 변수 안되고 수치형 변수만 선택 가능하게 하기 
    st.subheader("1️⃣ output 속성을 골라주세요!")
    st.write("(단, 수치형 변수, 이진 변수 (Binary Variable)만 가능)")

    # 수치형 변수: int 또는 float
    numerical_cols = [x for x in df.columns if pd.api.types.is_integer_dtype(df[x]) or pd.api.types.is_float_dtype(df[x])]

    # 이진 변수: 문자열/객체 타입이면서 고유값이 2개인 경우만 포함
    binary_cols = [x for x in df.columns 
                if (pd.api.types.is_string_dtype(df[x]) or pd.api.types.is_object_dtype(df[x]) or pd.api.types.is_categorical_dtype(df[x])) 
                and df[x].nunique() == 2]

    # 최종 리스트
    selected_columns = numerical_cols + binary_cols

    option = st.selectbox(
    "", selected_columns
    )

    # 레이 아웃 나누기
    col1, col2 , col3= st.columns(3, border=True)
    
    if option in numerical_cols:
        st.session_state.type='regression'
        # 이상치 설정
        with col1:
            st.write("* 이상치 설정")
            # Boxplot 생성
            fig, ax = plt.subplots(figsize=(8,2))

            # 가로형 Boxplot 생성
            ax.boxplot(df[option].dropna(), vert=False, patch_artist=False, showmeans=False, boxprops=dict(color="black"),
                    whiskerprops=dict(color="black"), capprops=dict(color="black"), flierprops=dict(marker="o", color="red"))

            # 불필요한 배경 제거
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.yaxis.set_visible(False)  # y축 숨김
            ax.xaxis.set_ticks_position('none')  # x축 눈금 숨김

            # Streamlit에 표시
            st.pyplot(fig)


            # 1사분위수(Q1)와 3사분위수(Q3) 계산
            Q1 = df[option].dropna().quantile(0.25)
            Q3 = df[option].dropna().quantile(0.75)
            IQR = Q3 - Q1

            # 이상치 기준 계산
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            method = None

            if (df[option].dropna()>upper_bound).any() or (df[option].dropna()<lower_bound).any:
                st.write("IQR 기준으로 이상치가 존재합니다! 어떻게 처리할까요?")
                method = st.selectbox(
                "",
                ("제거하기", "제거하지 않고 사용하기"),
                )

                st.write("You selected:", method)

            else:
                st.write("IQR 기준으로 이상치는 없고, 추가적인 설정은 필요 없어 보입니다!")


        # 결측치 설정
        with col2:
            cnt=len(df[option])
            missing_count=df[option].isnull().sum()
            missing_ratio = df[option].isnull().mean()
            st.write("* 결측치 설정")
            st.write(f"정상 데이터 수 : {cnt-missing_count}")
            st.write(f'결측치 수 : {missing_count}')
            st.write(f'결측치 비율 : {missing_ratio}')
            st.write("")
            st.write("")

            if missing_count:
                st.write("어떻게 처리할까요?")
                method2 = st.selectbox(
                "",
                ("관련 행 제거하기","평균으로 채우기","0으로 채우기"),
                )

                st.write("You selected:", method2)

            else:
                method2 = None
                st.write("결측치가 없어서 따로 설정은 필요 없어 보입니다!")
            
        # 범위 설정
        with col3:


            purpose=["최소화하기","최대화하기","범위에 맞추기","목표값에 맞추기"]
            method3 = st.radio("* 목표 설정",purpose)
            search_y={}
            if method3 == "범위에 맞추기":
                st.write("* output 범위 설정")
                values = st.slider("", min(df[option])-2*int(IQR), max(df[option])+2*int(IQR), (min(df[option]), max(df[option])))
                search_y[option]={'목표' : method3, '범위 설정' : values}

            elif method3 == "목표값에 맞추기":
                st.write("* 원하는 output 목표값 설정")
                number = st.number_input(
                "Insert a number", value=None, placeholder="Type a number..."
                )
                st.write("The current number is ", number)
                search_y[option]={'목표' : method3, '목표값' : number}
            
            else:
                search_y[option]={'목표' : method3}

    else:
        st.session_state.type="classification"
        with col1:
            st.write("0과 1 지정하기")
            st.write("1로 표시할 것을 정해주세요")
            value1 = df[option].unique()[0]
            value2 = df[option].unique()[1]
            label1 = st.checkbox(df[option].unique()[0])
            label2 = st.checkbox(df[option].unique()[1])
            df1=None
            if label1 and label2:
                st.write("둘 중에 하나만 선택해주세요!!!")
            
            elif label1:
                df1=df.copy()
                df1.loc[df1[option]==value1,option]=1
                df1.loc[df1[option]==value2,option]=0
                df1[option]=df1[option].astype(int)  # object → int 변환

            elif label2:
                df1=df.copy()
                df1.loc[df1[option]==value1,option]=0
                df1.loc[df1[option]==value2,option]=1
                df1[option]=df1[option].astype(int)  # object → int 변환
            else:
                st.write("")


            # 불균형 감지
            if df1 is not None:
                st.session_state.sampling=None
                class_counts = df1[option].value_counts()
                imbalance_ratio = class_counts.min() / class_counts.max()
                
                if imbalance_ratio < 0.33:  # 1:3 비율 이하라면 불균형으로 판단
                    st.write("분포가 불균형으로 감지되었습니다.")
                    oversampling = st.checkbox("오버 샘플링을 시도하겠습니까?")
                    st.write("(오버 샘플링이란 적은 데이터를 증강시키는 기법이다!)")
                    st.session_state.sampling = oversampling

                else:
                    st.write("데이터가 균형되어 보입니다!")



            
        # 결측치 설정
        with col2:
            cnt=len(df[option])
            missing_count=df[option].isnull().sum()
            missing_ratio = df[option].isnull().mean()
            st.write("* 결측치 설정")
            st.write(f"정상 데이터 수 : {cnt-missing_count}")
            st.write(f'결측치 수 : {missing_count}')
            st.write(f'결측치 비율 : {missing_ratio}')
            st.write("")
            st.write("")

            if missing_count:
                st.write("어떻게 처리할까요?")
                method2 = st.selectbox(
                "",
                ("관련 행 제거하기","평균으로 채우기","0으로 채우기"),
                )

                st.write("You selected:", method2)

            else:
                method2 = None
                st.write("결측치가 없어서 따로 설정은 필요 없어 보입니다!")
        
        with col3:
            search_y={}
            st.write("목표 설정하기")
            st.write("* 원하는 output 목표값 설정 (0 or 1)")
            method3="목표값에 맞추기"
            number = st.number_input(
                "Insert a number", value=None, placeholder="Type a number..."
                )
            search_y[option]={'목표' : method3, '목표값' : number}
            


            
    
    st.divider()

    # control할 제어 속성 정하기
    # 수치형만 가능하게 할 것인가?

    st.subheader("2️⃣ control할 제어 속성을 골라주세요!")
    st.write("(단, 수치형 변수만 가능)")

    option2 = st.multiselect(
    "",
    [x for x in df.columns if (x != option) & (pd.api.types.is_integer_dtype(df[x]) or pd.api.types.is_float_dtype(df[x]))],
    )
    tabs=None
    if option2:
        tabs = st.tabs(option2)
    control_feature={}  
    search_x={}
    if tabs:
        for ind,i in enumerate(tabs):
            with i:
                col1,col2,col3 = st.columns(3)

                with col1:
                    purpose=["최소화하기", "최대화하기", "최적화하지 않기"]
                    search_x[option2[ind]] = {"목표" : st.radio("목표 설정", purpose, key = option2[ind])}
                    

                with col2:
                    if df[option2[ind]].isnull().sum():
                        purpose2 = ["관련 행 제거하기","평균으로 채우기","0으로 채우기"]
                        control_feature[option2[ind]]=st.radio("결측치 설정", purpose2, key = option2[ind]+'1')
                    else:
                        st.write("결측치가 없어서 따로 설정은 필요 없어 보입니다!")
                        control_feature[option2[ind]]='X'

                with col3:
                    if min(df[option2[ind]]) == max(df[option2[ind]]):
                        min_val = min(df[option2[ind]]) - 1
                        max_val = max(df[option2[ind]]) + 1
                    else:
                        Q1 = df[option2[ind]].dropna().quantile(0.25)
                        Q3 = df[option2[ind]].dropna().quantile(0.75)
                        IQR = Q3 - Q1
                        min_val = min(df[option2[ind]]) - 2 * int(IQR)
                        max_val = max(df[option2[ind]]) + 2 * int(IQR)

                    values = st.slider(
                        "솔루션 최대 범위 설정", 
                        min_val, max_val, 
                        (min(df[option2[ind]]), max(df[option2[ind]])), 
                        key=option2[ind]+'2'
                    )
                    search_x[option2[ind]]['범위 설정'] = values
                    


    st.divider()


    # 환경 속성 정하기

    st.subheader("3️⃣ 환경 속성을 골라주세요!")

    # ✅ "설정하지 않기"를 추가하여 사용자 선택 가능하게 함
    option3 = st.multiselect(
        "(환경 속성이란 우리가 직접적으로 통제할 수 없는 외부 요인을 의미한다.)",
        [x for x in df.columns if x != option and x not in option2] + ["설정하지 않기"],
    )

    tabs = None
    env_feature = {}

    # ✅ "설정하지 않기"가 선택되었을 경우 빈 리스트로 변환
    if "설정하지 않기" in option3:
        option3 = []  # ✅ 빈 리스트로 변환

    # ✅ option3이 비어있지 않을 경우에만 `st.tabs()` 실행
    if option3:
        tabs = st.tabs(option3)

        for ind, i in enumerate(tabs):
            with i:
                if df[option3[ind]].isnull().sum():
                    if pd.api.types.is_string_dtype(df[option3[ind]]) or pd.api.types.is_object_dtype(df[option3[ind]]):
                        purpose3 = ["관련 행 제거하기", "최빈값으로 채우기"]
                        env_feature[option3[ind]] = st.radio("결측치 설정", purpose3, key=option3[ind]+'1')
                    else:
                        purpose3 = ["관련 행 제거하기", "평균으로 채우기", "0으로 채우기"]
                        env_feature[option3[ind]] = st.radio("결측치 설정", purpose3, key=option3[ind]+'1')
                else:
                    st.write("결측치가 없어서 따로 설정은 필요 없어 보입니다!")
                    env_feature[option3[ind]] = 'X'


    st.divider()

    #레이아웃 나누기
    col1, col2 = st.columns([14,1])

    with col1:
        if st.button("이전 페이지"):
            st.session_state.page="analysis"
            st.rerun()




    with col2:
        # 모델을 학습시키고 훈련시키는 과정으로 넘어가는 버튼 만들기

        if st.button("진행하기"):
            if option and option2:
                if option in binary_cols:
                    df=df1
                    df=remove_na(df,option,method2)
                else:
                    df=remove_outliers_iqr(df,option,method)
                    df=remove_na(df,option,method2)

                for i in control_feature.keys():
                    df=remove_na(df,i,control_feature[i]) 

                if option3:
                    for i in env_feature.keys():
                        df=remove_na(df,i,env_feature[i])
                    X= df[option2+option3]
                else:
                    X= df[option2]
                X= one_hot(X)
                y= df[option]

                if option in binary_cols:
                    if st.session_state.sampling:
                        # SMOTE 적용
                        smote = SMOTE(random_state=42)  # random_state는 재현성을 위해 설정
                        X_resampled, y_resampled = smote.fit_resample(X, y)
                        X = pd.DataFrame(X_resampled, columns=X.columns)
                        y = pd.Series(y_resampled, name=y.name)

                st.session_state.X= X
                st.session_state.y= y
                st.session_state.search_x=search_x
                st.session_state.search_y=search_y

                opt=[]
                for i,j in search_x.items():
                    if len(j)>=2 and j['목표']!="최적화하지 않기":
                        opt.append(i)
                
                if opt:
                    prior(option,opt)
                else:
                    st.session_state.page='train'
                    st.rerun()

            else:
                vote()
    


# page - train

elif st.session_state.page=="train":
    X=st.session_state.X
    y=st.session_state.y
    search_x=st.session_state.search_x
    search_y=st.session_state.search_y

    # 모델 훈련 및 최적화시키기
    train(X, y, search_x,search_y)

    st.session_state.page='result'
    st.rerun()

    
    
# page - result
# 결과 보여주는 건 tab을 이용하자

elif st.session_state.page=="result":
    train_score=st.session_state.train_score
    test_score=st.session_state.test_score
    train_time=st.session_state.train_time
    opt_time=st.session_state.opt_time

    st.success("Done!")
    st.title("🖥️ AI 솔루션 결과")
    st.write("빠른 결과를 보기 위해 최적화는 50개의 데이터만을 진행하였습니다!")
    st.write("좋은 결과라고 생각이 든다면 완성하기를 눌러주세요!")
    st.divider()

    col1, col2 = st.columns([2,3], border= True)
    with col1:
        st.subheader("모델 훈련 시간")
        st.metric('훈련에 든 시간', f'{train_time:.1f}초')
        st.metric('최적화(search)에 든 시간', f'{opt_time:.1f}초')

    with col2:
        st.subheader("모델 성능")
        col11,col22=st.columns((1,2))
        if st.session_state.type == "regression":
            with col11:
                score=test_score['R2']*100
                st.metric("모델 정확도(R2 기준)", f'{score:.1f}%')
            with col22:
                df=pd.DataFrame({'Train 성능' : train_score,'Test 성능': test_score})
                st.table(df)
        else:
            st.write("")
            with col11:
                score=test_score['F1 Score']*100
                st.metric("모델 정확도(F1 Score 기준)", f'{score:.1f}%')
            with col22:
                df=pd.DataFrame({'Train 성능' : train_score,'Test 성능': test_score})
                st.table(df)


    st.divider()

    st.subheader("각 변수와 output간의 관계")

    # 모델이랑 data 불러오기
    model = st.session_state.model
    X_test = st.session_state.X_test

    # PDP 플롯 그리기

    # Streamlit UI 추가 (사용자가 Feature 선택)
    search_x = st.session_state.search_x
    search_y = st.session_state.search_y
    tabs = st.tabs(list(search_x.keys()))


    for ind, tab in enumerate(tabs):
        with tab:
            col1, col2 = st.columns([2,1])
            with col1:
                # PDP 값 및 오차 계산
                x_vals, y_vals, error = partial_dependence_with_error(model, X_test, list(search_x.keys())[ind])

                # 리스트를 NumPy 배열로 변환
                y_vals = np.array(y_vals)
                errors = np.array(error)

                # Plotly Figure 생성
                fig = go.Figure()

                # ✅ 오차 범위 (± error) 추가 (투명한 fill 영역)
                fig.add_trace(go.Scatter(
                    x=x_vals, 
                    y=y_vals + errors, 
                    mode='lines', 
                    line=dict(color='rgba(255, 255, 255, 0)'),  # ✅ 완전히 투명하게
                    name="Upper Bound"
                ))

                fig.add_trace(go.Scatter(
                    x=x_vals, 
                    y=y_vals - errors, 
                    mode='lines', 
                    fill='tonexty',  # 아래 영역을 채우는 옵션
                    line=dict(color='rgba(255, 255, 255, 0)'),  # ✅ 완전히 투명하게
                    fillcolor='rgba(255, 165, 0, 0.5)',  # ✅ 오렌지색 + 투명도 50%
                    name="Lower Bound"
                ))

                # ✅ PDP 평균값 라인 추가
                fig.add_trace(go.Scatter(
                    x=x_vals, 
                    y=y_vals, 
                    mode='lines', 
                    line=dict(color='yellow', width=2),  # ✅ PDP 라인 색상 밝게 변경 (노란색)
                    name=f"PDP - {list(search_x.keys())[ind]}"
                ))

                # ✅ 레이아웃 설정
                fig.update_layout(
                    title=f"PDP - {list(search_x.keys())[ind]}",
                    xaxis_title=list(search_x.keys())[ind],
                    yaxis_title=f"Predicted {list(search_y.keys())[0]}",
                    template="plotly_white"
                )

                # ✅ Plotly 차트 출력
                st.plotly_chart(fig)

            with col2:
                st.write("🧐 PDP (Partial Dependence Plot)란?")
                st.write("PDP(부분 의존도 플롯, Partial Dependence Plot)은")
                st.write("머신러닝 모델에서 특정 변수(Feature)가 모델의 예측값에 어떻게 영향을 미치는지를 시각적으로 보여주는 그래프")



    st.divider()

    st.subheader("최적화 결과")

    df2 = st.session_state.df2
    X = st.session_state.X
    y = st.session_state.y

    # ✅ Y축 마진 설정 (10% 여유)
    def add_margin(y_min, y_max, margin_ratio=0.1):
        margin = (y_max - y_min) * margin_ratio
        return y_min - margin, y_max + margin

    # 📌 최적화 된 y값 vs 기존 y 값 비교
    original_col = list(search_y.keys())[0]
    solution_col = f"solution_{original_col}"

    chart_data = pd.DataFrame(
        pd.concat([df2[['y']].rename(columns={'y': solution_col}), y.head(50)], axis=1), 
        columns=[solution_col, original_col]
    )

    # ✅ 평균 차이 및 변화율(%) 계산
    original_mean = chart_data[original_col].mean()
    optimized_mean = chart_data[solution_col].mean()
    mean_difference = optimized_mean - original_mean
    percentage_change = (mean_difference / abs(original_mean)) * 100  # ✅ 변화율 계산

    # ✅ Y축 범위 계산 + 마진 추가
    y_min, y_max = add_margin(chart_data.min().min(), chart_data.max().max())

    fig = px.line(chart_data, labels={'index': 'Index', 'value': original_col}, title=f"Optimized vs Original {original_col}")
    fig.update_yaxes(range=[y_min, y_max])  # Y축 범위에 여유 추가

    st.plotly_chart(fig)

    # ✅ 변화율 출력
    st.write(f"**{original_col}의 변화율:** {percentage_change:.2f}%")

    # 📌 X 값에 대해 반복
    for i in search_x.keys():
        solution_col = f"solution_{i}"
        
        chart_data = pd.DataFrame(
            pd.concat([df2[[i]].rename(columns={i: solution_col}), X[i].head(50)], axis=1),
            columns=[solution_col, i]
        )

        # ✅ 평균 차이 및 변화율(%) 계산
        original_mean = chart_data[i].mean()
        optimized_mean = chart_data[solution_col].mean()
        mean_difference = optimized_mean - original_mean
        percentage_change = (mean_difference / abs(original_mean)) * 100  # ✅ 변화율 계산

        # ✅ Y축 범위 계산 + 마진 추가
        y_min, y_max = add_margin(chart_data.min().min(), chart_data.max().max())

        fig = px.line(chart_data, labels={'index': 'Index', 'value': i}, title=f"Optimized vs Original {i}")
        fig.update_yaxes(range=[y_min, y_max])  # Y축 범위에 여유 추가

        st.plotly_chart(fig)

        # ✅ 변화율 출력
        st.write(f"**{i}의 변화율:** {percentage_change:.2f}%")

    st.divider()



    # 피쳐 중요도
    st.subheader("Feature importance")
    col1, col2 = st.columns([1,2])

    with col1:
        # ✅ Feature Importance 값 (JSON 형식의 딕셔너리)
        feature_importance_dict = model.get_feature_importance()

        # ✅ 딕셔너리를 DataFrame으로 변환
        feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance'])

        # ✅ 중요도 값 기준으로 정렬 (내림차순)
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

        st.table(feature_importance_df)


    with col2:

        # ✅ Plotly Pie Chart 그리기
        fig1 = px.pie(
            feature_importance_df,  # DataFrame 전체를 전달
            names="Feature",        # Label로 사용할 컬럼
            values="Importance",    # 값으로 사용할 컬럼
            #title="Feature Importance (Pie Chart)",  # 제목 추가
            hole=0.3  # 도넛 차트 스타일 (원형이면 제거 가능)
        )

        # ✅ Streamlit에 Plotly 차트 출력
        st.plotly_chart(fig1)
    
    st.divider()
    if st.button:
        st.session_state.page='solution'
        st.rerun()




## page가 'solution'일때
else:
    model = st.session_state.model
    search_x = st.session_state.search_x
    search_y = st.session_state.search_y
    X = st.session_state.X
    y = st.session_state.y
    with st.spinner('전체 데이터 최적화 진행 중입니다..(약 30분 소요 예정)'):
        _, optimal_solutions_df = search(X, y, model, search_x,search_y)

    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")

    csv = convert_df(optimal_solutions_df)

    st.write("진행이 완료되었습니다!")

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="solution.csv",
        mime="text/csv",
    )
