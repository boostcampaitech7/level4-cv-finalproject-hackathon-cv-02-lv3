import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from regplot import partial_dependence_with_error
from sklearn.model_selection import train_test_split
from aisolution import aisolution
from search import search
from imblearn.over_sampling import SMOTE
# 페이지 설정
st.set_page_config(
    page_title="LuckyVicky AI Solution",
    page_icon="🌟",
    layout="wide"
)
# 타이틀
st.title("🖥️ AI 솔루션 결과")
st.write("빠른 결과를 보기 위해 최적화는 50개의 데이터만을 진행합니다!")
st.write("좋은 결과라고 생각이 든다면 완성하기를 눌러주세요!")
st.divider()

def train(X, y, search_x, search_y):
    
    if "train_score" not in st.session_state:
        with st.spinner('🔄 모델 훈련 및 최적화 진행 중입니다...(약 10분 소요 예정)'):

            search_x = {key: search_x[key] for key in sorted(search_x.keys())}
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test

            # 모델 훈련
            train_score, test_score, train_time, model = aisolution(
                X_train=X_train, X_test=X_test, y_test=y_test, y_train=y_train,
                task_type=st.session_state.type
            )

            # 50개 데이터로 최적화 수행
            elapsed_time, optimal_solutions_df = search(X.head(50), y.head(50), model, search_x, search_y)

            # 결과 저장
            st.session_state.train_score = train_score
            st.session_state.test_score = test_score
            st.session_state.train_time = train_time
            st.session_state.opt_time = elapsed_time
            st.session_state.df2 = optimal_solutions_df
            st.session_state.model = model

if st.session_state.previous_selections:

    # 데이터 로드
    X = st.session_state.X
    y = st.session_state.y
    search_x = st.session_state.search_x
    search_y = st.session_state.search_y

    # 페이지 로드 시 설정 변경 여부 확인
    if st.session_state.get('settings_changed', False):
        # 설정 변경 감지 시 이전 훈련 결과 삭제
        st.session_state.pop('train_score', None)
        st.session_state.pop('test_score', None)
        st.session_state.pop('train_time', None)
        st.session_state.pop('opt_time', None)
        st.session_state.pop('df2', None)
        st.session_state.pop('model', None)
        
        # 설정 변경 플래그 초기화
        st.session_state.settings_changed = False

    # 모델 훈련 및 최적화 진행
    train(X, y, search_x, search_y)


    # 결과 페이지
    train_score = st.session_state.train_score
    test_score = st.session_state.test_score
    train_time = st.session_state.train_time
    opt_time = st.session_state.opt_time

    st.success("Done!")
    st.divider()


    # 모델 훈련 및 최적화 시간
    col1, col2 = st.columns([2, 3], border=True)
    with col1:
        st.subheader("모델 훈련 시간")
        st.metric('훈련에 든 시간', f'{train_time:.1f}초')
        st.metric('최적화(search)에 든 시간', f'{opt_time:.1f}초')

    with col2:
        st.subheader("모델 성능")
        col11, col22 = st.columns((1, 2))
        if st.session_state.type == "regression":
            with col11:
                score = test_score['R2'] * 100
                st.metric("모델 정확도(R2 기준)", f'{score:.1f}%')
            with col22:
                df = pd.DataFrame({'Train 성능': train_score, 'Test 성능': test_score})
                st.table(df)
        else:
            with col11:
                score = test_score['F1 Score'] * 100
                st.metric("모델 정확도(F1 Score 기준)", f'{score:.1f}%')
            with col22:
                df = pd.DataFrame({'Train 성능': train_score, 'Test 성능': test_score})
                st.table(df)

    st.divider()

    # 각 변수와 output 간의 관계
    st.subheader("각 변수와 output간의 관계")
    model = st.session_state.model
    X_test = st.session_state.X_test

    search_x_keys = sorted(list(search_x.keys()))  # 고정된 순서 유지
    tabs = st.tabs(search_x_keys)

    for ind, tab in enumerate(tabs):
        with tab:
            col1, col2 = st.columns([2, 1])
            with col1:
                x_vals, y_vals, error = partial_dependence_with_error(model, X_test, search_x_keys[ind])
                y_vals, errors = np.array(y_vals), np.array(error)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals + errors, mode='lines', line=dict(color='rgba(255, 255, 255, 0)'), name="Upper Bound"))
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals - errors, mode='lines', fill='tonexty', line=dict(color='rgba(255, 255, 255, 0)'), fillcolor='rgba(255, 165, 0, 0.5)', name="Lower Bound"))
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='yellow', width=2), name=f"PDP - {list(search_x.keys())[ind]}"))
                fig.update_layout(title=f"PDP - {search_x_keys[ind]}", xaxis_title=search_x_keys[ind], yaxis_title=f"Predicted {list(search_y.keys())[0]}", template="plotly_white")

                st.plotly_chart(fig)

            with col2:
                st.write("🧐 PDP (Partial Dependence Plot)란?")
                st.write("PDP(부분 의존도 플롯, Partial Dependence Plot)은 머신러닝 모델에서 특정 변수(Feature)가 모델의 예측값에 어떻게 영향을 미치는지를 시각적으로 보여주는 그래프")

    st.divider()

    # 최적화 결과
    st.subheader("최적화 결과")
    df2 = st.session_state.df2
    X = st.session_state.X
    y = st.session_state.y

    def add_margin(y_min, y_max, margin_ratio=0.1):
        margin = (y_max - y_min) * margin_ratio
        return y_min - margin, y_max + margin

    original_col = list(search_y.keys())[0]
    solution_col = f"solution_{original_col}"

    chart_data = pd.concat([df2[['y']].rename(columns={'y': solution_col}), y.head(50)], axis=1)
    original_mean = chart_data[original_col].mean()
    optimized_mean = chart_data[solution_col].mean()
    percentage_change = ((optimized_mean - original_mean) / abs(original_mean)) * 100

    y_min, y_max = add_margin(chart_data.min().min(), chart_data.max().max())

    fig = px.line(chart_data, labels={'index': 'Index', 'value': original_col}, title=f"Optimized vs Original {original_col}")
    fig.update_yaxes(range=[y_min, y_max])
    st.plotly_chart(fig)
    st.write(f"**{original_col}의 변화율:** {percentage_change:.2f}%")

    for i in sorted(search_x.keys()):
        solution_col = f"solution_{i}"
        chart_data = pd.concat([df2[[i]].rename(columns={i: solution_col}), X[i].head(50)], axis=1)
        original_mean = chart_data[i].mean()
        optimized_mean = chart_data[solution_col].mean()
        percentage_change = ((optimized_mean - original_mean) / abs(original_mean)) * 100

        y_min, y_max = add_margin(chart_data.min().min(), chart_data.max().max())

        fig = px.line(chart_data, labels={'index': 'Index', 'value': i}, title=f"Optimized vs Original {i}")
        fig.update_yaxes(range=[y_min, y_max])
        st.plotly_chart(fig)
        st.write(f"**{i}의 변화율:** {percentage_change:.2f}%")

    st.divider()

    # 피쳐 중요도
    st.subheader("Feature importance")
    col1, col2 = st.columns([1, 2], border=True)

    with col1:
        feature_importance_dict = model.get_feature_importance()
        feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance']).sort_values(by="Importance", ascending=False)
        st.table(feature_importance_df)

    with col2:
        fig1 = px.pie(feature_importance_df, names="Feature", values="Importance", hole=0.3)
        st.plotly_chart(fig1)

    st.divider()
    if st.button("이대로 진행하기"):
        st.switch_page("pages/4_Results.py")

else:
    st.warning("Solution 설정을 하지 않았습니다! Solution 부터 설정해주세요!")