import os
import json
import pandas as pd
import streamlit as st

# 1. 페이지 기본 설정 (브라우저 탭 제목 및 아이콘 설정)
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 데이터 저장 폴더 및 JSON 파일 경로
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "data_info.json")

# 2. 저장 데이터 불러오기/저장 함수 정의
def load_saved_data():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"meanings": {}, "source": ""}

def save_data(data):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

saved_content = load_saved_data()

# 3. 데이터 로드 (캐싱 활용)
@st.cache_data
def load_stroke_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_stroke_data()

# 4. 상단 헤더
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("---")

# 5. 핵심 지표 카드 (4개)
col1, col2, col3, col4 = st.columns(4)

total_people = len(df)
total_cols = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = (stroke_count / total_people) * 100 if total_people > 0 else 0.0

col1.metric("전체 사람 수", f"{total_people:,} 명")
col2.metric("열 개수", f"{total_cols} 개")
col3.metric("뇌졸중 환자 수(stroke=1)", f"{stroke_count:,} 명")
col4.metric("뇌졸중 비율", f"{stroke_ratio:.2f} %")

st.markdown("---")

# 6. 열 정보 표 및 우리말 뜻 입력 UI
st.subheader("📊 데이터 열 정보")
st.write("우리말 뜻을 입력하고 저장 버튼을 누르면 `내_기록` 폴더에 자동 저장됩니다.")

columns_list = list(df.columns)
meanings_input = {}

# 데이터 프레임 정보 생성
info_rows = []
for col in columns_list:
    dtype_str = str(df[col].dtype)
    null_count = int(df[col].isnull().sum())
    info_rows.append({
        "열 이름": col,
        "값의 종류": dtype_str,
        "빈 값 개수": null_count
    })

info_df = pd.DataFrame(info_rows)

# 폼 형태로 구성하여 직접 입력 처리
with st.form("meanings_form"):
    cols = st.columns([2, 3, 2, 2])
    cols[0].markdown("**열 이름**")
    cols[1].markdown("**우리말 뜻**")
    cols[2].markdown("**값의 종류**")
    cols[3].markdown("**빈 값 개수**")
    
    for idx, row in info_df.iterrows():
        col_name = row["열 이름"]
        c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
        c1.text(col_name)
        
        default_val = saved_content.get("meanings", {}).get(col_name, "")
        user_input = c2.text_input(
            f"{col_name}_meaning",
            value=default_val,
            label_visibility="collapsed"
        )
        meanings_input[col_name] = user_input
        
        c3.text(row["값의 종류"])
        c4.text(str(row["빈 값 개수"]))

    submit_meanings = st.form_submit_button("우리말 뜻 저장")
    if submit_meanings:
        saved_content["meanings"] = meanings_input
        save_data(saved_content)
        st.success("우리말 뜻이 성공적으로 저장되었습니다!")

st.markdown("---")

# 7. 상위 5개 데이터 출력
st.subheader("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# 8. 데이터 출처 작성 영역
st.subheader("📌 데이터 출처")

with st.form("source_form"):
    saved_source = saved_content.get("source", "")
    source_input = st.text_area(
        "교재에 기재된 데이터 출처 정보를 적어주세요:",
        value=saved_source,
        height=100
    )
    submit_source = st.form_submit_button("출처 저장")
    if submit_source:
        saved_content["source"] = source_input
        save_data(saved_content)
        st.success("데이터 출처가 성공적으로 저장되었습니다!")
