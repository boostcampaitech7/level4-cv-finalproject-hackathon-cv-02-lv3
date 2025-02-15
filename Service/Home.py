import streamlit as st
import pandas as pd
import os

# 페이지 설정
st.set_page_config(
    page_title="LuckyVicky AI Solution",
    page_icon="🌟",
    layout="wide"
)

# 타이틀과 소개 문구
st.title("💊 Prescriptive AI 분석을 시작하세요")
st.markdown("CSV 파일을 업로드하고 데이터 분석을 시작하세요.")

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

            st.session_state['df'] = merged_df
            st.success("파일 업로드 성공! 데이터 분석 페이지로 이동하세요.")

        except Exception as e:
            st.error(f"파일 병합 중 오류 발생: {e}")

# ✅ 단일 파일 업로드 시 기존 방식 유지
elif uploaded_files and len(uploaded_files) == 1:
    st.session_state['df'] = pd.read_csv(uploaded_files[0])
    st.session_state.page = "analysis"
    st.success("파일 업로드 성공! 데이터 분석 페이지로 이동하세요.")

# 이미지 추가 (이미지가 'images' 폴더에 있는지 확인)
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "images", "image1.png")

if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.warning("이미지를 불러올 수 없습니다. 이미지 경로를 확인하세요.")

# 데이터 분석 페이지로 이동하는 버튼
if st.session_state.get('df') is not None:
    if st.button("데이터 분석 페이지로 이동"):
        st.switch_page("pages/1_Data_Analysis.py")









# # 파일 업로드 시 세션 상태에 저장하고 페이지 리로딩
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.session_state['df'] = df
#     st.success("파일 업로드 성공! 데이터 분석 페이지로 이동하세요.")




# # 이미지 추가 (이미지가 'images' 폴더에 있는지 확인)
# current_dir = os.path.dirname(__file__)
# image_path = os.path.join(current_dir, "images", "image1.png")

# if os.path.exists(image_path):
#     st.image(image_path, use_container_width=True)
# else:
#     st.warning("이미지를 불러올 수 없습니다. 이미지 경로를 확인하세요.")


# # 데이터 분석 페이지로 이동하는 버튼
# if st.session_state.get('df') is not None:
#     if st.button("데이터 분석 페이지로 이동"):
#         st.switch_page("pages/1_Data_Analysis.py")
