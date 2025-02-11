import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_utils import remove_outliers_iqr, remove_na, one_hot

# 페이지 설정
st.set_page_config(
    page_title="LuckyVicky AI Solution",
    page_icon="🌟",
    layout="wide"
)

# 타이틀
st.title("💊 AI 솔루션")
st.write("진행하기 전에 분석하고 싶은 feature와 목표를 설정하세요!")
st.divider()

# 세션 상태에서 데이터프레임 불러오기
df = st.session_state.get('df')

# 설정 이전의 값 복사 전환할 목표와 치환 프로세싱을 저장
if 'previous_selections' not in st.session_state:
    st.session_state.previous_selections = {}

# 이전 선택 옵션이 필요할 경우 값 검사 연동
previous_selections = st.session_state.previous_selections

def vote():
    st.write("속성을 지정하지 않아 다음 단계로 진행할 수 없습니다.")
    st.write("반드시 지정해주세요!")
    if st.button("다시 지정하기"):
        st.rerun()


# 우선순위 설정 디어러리로 표시
@st.dialog("우선 순위를 정해주세요!")
def prior(option, opt):
    st.write("목표가 2개 이상이기 때문에 우선순위를 정해주세요!")

    options = [option] + opt
    # 이전 선택이 현재 옵션에 존재하는지 확인 후 유효성 검사
    previous_priority = st.session_state.previous_selections.get('priority_selection', [])
    valid_previous_priority = [item for item in previous_priority if item in options]

    # 유효한 이전 선택만 기본값으로 설정
    selection = st.multiselect(
        "우선순위를 정해주세요!",
        options,
        default=valid_previous_priority
    )
    #selection = st.pills("Directions", options, selection_mode="multi")
    #selection = st.multiselect("우선순위를 정해주세요!", options,default=previous_selections.get('priority_selection', []))
    st.markdown(f"Your selected options:  \n {[f'{i+1}순위 : {j}'for i,j in enumerate(selection)]}.")

    if st.button("submit"):
        for i, j in enumerate(selection):
            if j in st.session_state.search_x.keys():
                st.session_state.search_x[j]['순위'] = i + 1
            else:
                st.session_state.search_y[j]['순위'] = i + 1
        st.success("우선순위가 성공적으로 설정되었습니다!")
        
        # 우선순위 설정을 세션에 저장
        st.session_state.previous_selections['priority_selection'] = selection
        st.switch_page("pages/3_Model_Training.py")

if df is not None:
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

    st.session_state.sampling=None
    option = st.selectbox(
    "", selected_columns,
        index=selected_columns.index(previous_selections.get('output_selection', selected_columns[0]))
    )
    
    if option:
        st.session_state.previous_selections['output_selection'] = option
    
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

    # 이전 선택 값 불러오기
    default_selection = previous_selections.get('control_selection', [])
    
    option2 = st.multiselect(
    "",
    [x for x in df.columns if (x != option) & (pd.api.types.is_integer_dtype(df[x]) or pd.api.types.is_float_dtype(df[x]))],
    )
            
    if option2:
        st.session_state.previous_selections['control_selection'] = option2
        
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

                    for i in control_feature.keys():
                        df = remove_na(df, i, control_feature[i])

                    values = st.slider(
                        "솔루션 최대 범위 설정", 
                        min_val, max_val, 
                        (min(df[option2[ind]]), max(df[option2[ind]])), 
                        key=option2[ind]+'2'
                    )
                    search_x[option2[ind]]['범위 설정'] = values
                
    st.divider()


    st.subheader("3️⃣ 환경 속성을 골라주세요!")


    option3 = st.multiselect(
        "(환경 속성이란 우리가 직접적으로 통제할 수 없는 외부 요인을 의미한다.)",
        [x for x in df.columns if x != option and x not in option2]+["진행하지 않기"],
            default=previous_selections.get('environment_selection', [])
    )

    if option3:
        st.session_state.previous_selections['environment_selection'] = option3

    non_option3=False
    tabs = None
    env_feature = {}

    if option3 :
        if "진행하지 않기" not in option3:
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
        else:
            non_option3=True



    # 레이아웃 나누기
    col1, col2 = st.columns([14, 1])

    with col1:
        if st.button("이전 페이지"):
            st.switch_page("pages/1_Data_Analysis.py")

    with col2:
        # 모델을 학습시키고 훈련시키는 과정으로 넘어가는 버튼 만들기
        if st.button("진행하기"):
            # 설정 변경 플래그 추가
            st.session_state.settings_changed = True 
            
            if option and option2:
                if option in binary_cols:
                    df = df1
                    df = remove_na(df, option, method2)
                else:
                    df = remove_outliers_iqr(df, option, method)
                    df = remove_na(df, option, method2)

                for i in control_feature.keys():
                    df=remove_na(df,i,control_feature[i]) 

                if non_option3:
                    X= df[option2]
                
                else:
                    for i in env_feature.keys():
                        df=remove_na(df,i,env_feature[i])
                    X= df[option2 + option3]
                
                X = one_hot(X)
                y = df[option]

                # if option in binary_cols:
                #     if st.session_state.sampling:
                #         # SMOTE 적용
                #         smote = SMOTE(random_state=42)  # random_state는 재현성을 위해 설정
                #         X_resampled, y_resampled = smote.fit_resample(X, y)
                #         X = pd.DataFrame(X_resampled, columns=X.columns)
                #         y = pd.Series(y_resampled, name=y.name)
                st.session_state.method = method  # 이상치 처리 방법 저장
                st.session_state.method2 = method2  # 결측치 처리 방법 저장
                st.session_state.method3 = method3  # 목표 설정 방법 저장
                
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.search_x = search_x
                st.session_state.search_y = search_y
                st.session_state.control_feature = control_feature  # 제어 속성 저장
                st.session_state.env_feature = env_feature  # 환경 속성 저장
                
                # 제어 속성 및 환경 속성도 추가했다면 여기에 저장
                st.session_state.search_x = search_x if 'search_x' in locals() else {}

                opt = []
                for i, j in search_x.items():
                    if len(j) >= 2 and j["목표"] != "최적화하지 않기":
                        opt.append(i)

                if opt:
                    prior(option, opt)
                else:
                    st.switch_page("pages/3_Model_Training.py")

            else:
                vote()
            
else:
    st.warning("CSV 파일이 업로드되지 않았습니다. Home 페이지로 돌아가 파일을 업로드하세요.")
