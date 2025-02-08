import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from aisolution import aisolution
from search import search
from regplot import partial_dependence_with_error


# ✅ 경고 창
@st.dialog("진행 불가")
def vote():
    st.write("속성을 지정하지 않아 다음 단계로 진행할 수 없습니다.")
    st.write("반드시 지정해주세요!")
    if st.button("다시 지정하기"):
        st.rerun()


@st.dialog("우선 순위를 정해주세요!")
def prior(option, opt):
    st.write("목표가 2개 이상이기 때문에 우선순위를 정해주세요!")
    options = [option] + opt
    selection = st.pills("Directions", options, selection_mode="multi")
    st.markdown(f"Your selected options:  \n {[f'{i+1}순위 : {j}' for i,j in enumerate(selection)]}.")
    
    if st.button("Submit"):
        for i, j in enumerate(options):
            if j in st.session_state.search_x:
                st.session_state.search_x[j]["순위"] = i + 1
            else:
                st.session_state.search_y[j]["순위"] = i + 1
        
        st.session_state.page = "train"
        st.rerun()


@st.dialog("Solution 진행 중")
def train(X, y, search_x, search_y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.session_state.X_train, st.session_state.X_test = X_train, X_test

    with st.spinner("분석 및 최적화 진행 중입니다..(약 10분 소요 예정)"):
        train_score, test_score, train_time, model = aisolution(X_train, X_test, y_train, y_test, st.session_state.type)
        elapsed_time, optimal_solutions_df = search(X.head(50), y.head(50), model, search_x, search_y)

    # ✅ 결과 저장
    st.session_state.update({
        "train_score": train_score,
        "test_score": test_score,
        "train_time": train_time,
        "opt_time": elapsed_time,
        "df2": optimal_solutions_df,
        "model": model,
    })


# ✅ 데이터 전처리 함수
def remove_outliers_iqr(df, col, method):
    if method == "제거하기":
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)].reset_index(drop=True)
    return df


def remove_na(df, col, method):
    if method == "관련 행 제거하기":
        return df.dropna(subset=[col]).reset_index(drop=True)
    elif method == "평균으로 채우기":
        return df.fillna({col: df[col].mean()})
    elif method == "0으로 채우기":
        return df.fillna({col: 0})
    elif method == "최빈값으로 채우기":
        return df.fillna({col: df[col].mode()[0]})
    return df


def one_hot_encode(df):
    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# ✅ 페이지 설정
st.set_page_config(layout="wide", page_title="AI Solution", page_icon="📊")

if "page" not in st.session_state:
    st.session_state.page = False
    st.session_state.df = None

# ✅ 데이터 업로드
if not st.session_state.page:
    st.title("Prescript AI Solution")
    uploaded_files = st.file_uploader("CSV 파일을 업로드하세요", type="csv", accept_multiple_files=True)

    if uploaded_files:
        dfs = [pd.read_csv(file) for file in uploaded_files]
        if len(dfs) > 1:
            merge_option = st.radio("파일 합치기 방식", ["행으로 합치기", "열로 합치기"])
            if st.button("파일 병합 및 분석 시작"):
                try:
                    if merge_option == "행으로 합치기":
                        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
                    else:
                        merged_df = pd.concat(dfs, axis=1)
                    st.session_state.df = merged_df
                    st.session_state.page = "analysis"
                    st.success("파일 병합 완료! 분석 화면으로 이동합니다.")
                    st.rerun()
                except Exception as e:
                    st.error(f"파일 병합 중 오류 발생: {e}")
        else:
            st.session_state.df = dfs[0]
            st.session_state.page = "analysis"
            st.success("파일 업로드 성공! 분석 화면으로 이동합니다.")
            st.rerun()

# ✅ 데이터 분석 페이지
elif st.session_state.page == "analysis":
    st.title("📊 데이터 분석 결과")
    df = st.session_state.df

    # 📌 EDA (탭 활용)
    tab_overview, tab_columns = st.tabs(["전체 데이터", "개별 Column 분석"])
    
    with tab_overview:
        st.dataframe(df, height=400)
        st.write(f"- 총 데이터 개수: {len(df)}")
        st.write(f"- 컬럼 수: {len(df.columns)}")

    with tab_columns:
        col = st.selectbox("분석할 컬럼 선택", df.columns)
        st.write(f"📌 '{col}' 컬럼 정보")
        st.write(f"- 데이터 타입: {df[col].dtype}")
        st.write(f"- 결측치 비율: {df[col].isnull().mean() * 100:.2f}%")

        # 📌 데이터 시각화
        if df[col].dtype in ["int64", "float64"]:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), bins=20, kde=True, ax=ax)
            st.pyplot(fig)

        elif df[col].dtype == "object":
            st.bar_chart(df[col].value_counts())

    # 📌 다음 단계 버튼
    col1, col2 = st.columns((9, 1))
    with col1:
        if st.button("다시 제출하기"):
            st.session_state.page = False
            st.session_state.df = None
            st.rerun()
    with col2:
        if st.button("솔루션 시작하기"):
            st.session_state.page = "solution"
            st.rerun()

