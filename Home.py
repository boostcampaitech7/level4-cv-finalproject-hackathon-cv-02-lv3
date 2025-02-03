import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
# from sklearn.model_selection import train_test_split


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
        st.rerun()

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

    if method == "관련 행 제거하기":
        return df.dropna(subset=[option]).reset_index(drop=True)  # 해당 열에서 결측치가 있는 행 제거
    
    elif method == "평균으로 채우기":
        mean_value = df[option].mean()  # 평균값 계산
        return df.fillna({option: mean_value})  # 결측치 평균값으로 대체
    
    elif method == "0으로 채우기":
        return df.fillna({option: 0})
    
    else:
        return df

# 페이지 레이아웃 설정
st.set_page_config(layout="wide", page_title="AI solution", page_icon="📊")

# 상태를 저장할 page 초기화
if "page" not in st.session_state:
    st.session_state.page = False
    st.session_state.uploaded_file = None
    st.session_state.df = None  # 데이터프레임을 저장할 새로운 상태 변수


# 로그인 창처럼 구현된 파일 업로드 화면
if not st.session_state.page:
    st.title("Prescript AI solution")
    st.write("분석하고 싶은 CSV 파일을 제출하세요.")
    
    # 파일 업로드 위젯
    uploaded_file = st.file_uploader("",type="csv")

    # 파일 업로드 후 로그인 상태로 전환
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.page = "analysis"
        # 데이터프레임을 세션 상태에 저장
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("파일 업로드 성공! 분석 화면으로 이동합니다.")
        st.rerun()  # 화면 갱신





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

            # 레이아웃 나누기
            col_1, col_2 = st.columns([1, 1], border=True)  # 왼쪽 1: 오른쪽 1 비율 설정
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
            col_1, col_2 = st.columns([1, 1] , border=True)  # 왼쪽 1: 오른쪽 1 비율 설정
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
                #  결측치 비율 파이 차트
                st.write("### Missing Value Visualization")
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.pie(
                    [missing_ratio, non_missing_ratio],
                    colors=['#FF0000', '#66b3ff'],
                    startangle=90,
                    wedgeprops={'edgecolor': 'black'}
                )
                ax.legend(labels=['missing', 'non_missing'],loc='lower right',fontsize=6)

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
    st.write("(단, 수치형 변수만 가능)")
    option = st.selectbox(
    "",
    [x for x in df.columns if pd.api.types.is_integer_dtype(df[x]) or pd.api.types.is_float_dtype(df[x])],
    )

    # 레이 아웃 나누기
    col1, col2 , col3= st.columns(3, border=True)

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

    
    st.divider()

    # control할 제어 속성 정하기
    # 수치형만 가능하게 할 것인가?

    st.subheader("2️⃣ control할 제어 속성을 골라주세요!")
    option2 = st.multiselect(
    "",
    [x for x in df.columns if x != option],
    )
    tabs=None
    if option2:
        tabs = st.tabs(option2)
    control_feature={}  
    search_x={}
    if tabs:
        for ind,i in enumerate(tabs):
            with i:
                if pd.api.types.is_integer_dtype(df[option2[ind]]) or pd.api.types.is_float_dtype(df[option2[ind]]):
                    col1,col2,col3 = st.columns(3)

                    with col1:
                        purpose=["최소화하기", "최대화하기", "최적화하지 않기"]
                        search_x[option2[ind]] = {"목표" : st.radio("목표 설정", purpose, key = option2[ind])}
                        

                    with col2:
                        purpose2 = ["관련 행 제거하기","평균으로 채우기","0으로 채우기"]
                        control_feature[option2[ind]]=[st.radio("결측치 설정", purpose2, key = option2[ind]+'1')]

                    with col3:
                        # 1사분위수(Q1)와 3사분위수(Q3) 계산
                        Q1 = df[option2[ind]].dropna().quantile(0.25)
                        Q3 = df[option2[ind]].dropna().quantile(0.75)
                        IQR = Q3 - Q1

                        values = st.slider("솔루션 최대 범위 설정", min(df[option2[ind]])-2*int(IQR), max(df[option2[ind]])+2*int(IQR), 
                                (min(df[option2[ind]]), max(df[option2[ind]])), key = option2[ind]+'2')
                        search_x[option2[ind]]['범위 설정'] = values
                        
                
                else:
                    purpose2 = ["관련 행 제거하기","평균으로 채우기","0으로 채우기"]
                    control_feature[option2[ind]] = [st.radio("결측치 설정", purpose2, key = option2[ind]+'1')]
                    


    st.divider()


    # 환경 속성 정하기

    st.subheader("3️⃣ 환경 속성을 골라주세요!")
    option3 = st.multiselect(
    "(환경 속성이란 우리가 직접적으로 통제할 수 없는 외부 요인을 의미한다.)",
    [x for x in df.columns if x != option and x not in option2],
    )

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
            if option and option2 and option3:

                df=remove_outliers_iqr(df,option,method)
                df=remove_na(df,option,method)

                for i in control_feature.keys():
                    df=remove_na(df,i,control_feature[i]) 
                
                st.session_state.page = "train"  # train page로 넘어가기
                
                st.session_state.X=df[option2+option3]
                st.session_state.y=df[option]
                st.session_state.search_x=search_x
                st.session_state.search_y=search_y

                opt=[]
                for i,j in search_x.items():
                    if len(j)>=2 and j['목표']!="최적화하지 않기":
                        opt.append(i)
                
                if opt:
                    prior(option,opt)
                else:
                    st.rerun()

            else:
                vote()
    





# page - train
# 결과 보여주는 건 tab을 이용하자

else:
    with st.spinner('Wait for it...'):
        time.sleep(5)

    X=st.session_state.X
    y=st.session_state.y
    search_x=st.session_state.search_x
    search_y=st.session_state.search_y

    st.success("Done!")
    st.title("🖥️ AI 솔루션 결과")

    st.table(X.head())
    st.table(y.head())
    st.write(search_x)
    st.write(search_y)

    st.divider()

    st.subheader("모델 성능")

    st.divider()

    st.subheader("각 변수와 output간의 관계")


    st.divider()

    st.subheader("최적화 결과")


    st.divider()
    st.subheader("Feature importance")