
import streamlit as st
import pandas as pd
import json
from pathlib import Path

# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "data_dictionary.json"

COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
]


# --------------------------------------------------
# 2. 기록 파일 관리
# --------------------------------------------------

def load_records():
    """JSON 파일에서 기존 기록 불러오기"""

    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    if RECORD_FILE.exists():
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as file:
                records = json.load(file)

            if not isinstance(records, dict):
                return {"meanings": {}, "source": ""}

            records.setdefault("meanings", {})
            records.setdefault("source", "")

            return records

        except (json.JSONDecodeError, OSError):
            st.warning("기존 기록 파일을 읽을 수 없습니다.")
    
    return {
        "meanings": {},
        "source": ""
    }


def save_records(meanings, source):
    """우리말 뜻과 데이터 출처를 JSON 파일에 저장"""

    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    records = {
        "meanings": meanings,
        "source": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=4
        )


# --------------------------------------------------
# 3. 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 열 순서 확인
    df = df[COLUMNS]

    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.error(f"오류 내용: {e}")
    st.info("인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.stop()


# --------------------------------------------------
# 4. 기존 기록 불러오기
# --------------------------------------------------

records = load_records()

saved_meanings = records["meanings"]
saved_source = records["source"]


# --------------------------------------------------
# 5. 화면 제목
# --------------------------------------------------

st.title("🧠 뇌졸중 예측 실습실")

st.markdown(
    """
    ### 데이터 탐색하기

    이 실습실에서는 뇌졸중 관련 데이터를 살펴보고,
    데이터의 특징을 이해하는 활동을 합니다.

    먼저 데이터의 전체적인 구조와 각 열의 의미를 확인해 봅시다.
    """
)

st.divider()


# --------------------------------------------------
# 6. 데이터 기본 정보
# --------------------------------------------------

total_people = len(df)
total_columns = len(df.columns)

stroke_count = int((df["stroke"] == 1).sum())

stroke_rate = (
    stroke_count / total_people * 100
    if total_people > 0 else 0
)


st.subheader("📊 데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="전체 사람 수",
        value=f"{total_people:,}명"
    )

with col2:
    st.metric(
        label="열 개수",
        value=f"{total_columns}개"
    )

with col3:
    st.metric(
        label="stroke가 1인 사람 수",
        value=f"{stroke_count:,}명"
    )

with col4:
    st.metric(
        label="뇌졸중 비율",
        value=f"{stroke_rate:.2f}%"
    )


st.divider()


# --------------------------------------------------
# 7. 열 정보 표
# --------------------------------------------------

st.subheader("📋 데이터 열 살펴보기")

st.write(
    "각 열의 이름과 값의 종류를 확인해 보세요. "
    "우리말 뜻은 교재를 참고하여 직접 입력할 수 있습니다."
)

column_info = []

for column in df.columns:

    # 값의 종류
    value_types = df[column].dropna().nunique()

    # 빈 값 개수
    missing_count = int(df[column].isna().sum())

    column_info.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": value_types,
        "빈 값 개수": missing_count
    })

info_df = pd.DataFrame(column_info)

# 우리말 뜻 입력 기능
edited_info = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            width="medium"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻 (직접 입력)",
            width="large",
            help="교재를 참고하여 각 열의 우리말 뜻을 입력하세요."
        ),
        "값의 종류": st.column_config.NumberColumn(
            "값의 종류"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수"
        )
    },
    key="meaning_editor"
)


st.divider()


# --------------------------------------------------
# 8. 데이터 처음 다섯 줄
# --------------------------------------------------

st.subheader("🔎 데이터 처음 다섯 줄")

st.write(
    "실제 데이터가 어떤 형태로 저장되어 있는지 "
    "처음 다섯 개의 행을 확인해 봅시다."
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


st.divider()


# --------------------------------------------------
# 9. 데이터 출처 입력
# --------------------------------------------------

st.subheader("📚 데이터 출처")

st.write(
    "교재에 안내된 데이터 출처를 아래에 입력하세요."
)

source_text = st.text_area(
    "데이터 출처 입력",
    value=saved_source,
    placeholder="교재를 참고하여 데이터 출처를 입력하세요.",
    height=120,
    key="source_editor"
)


# --------------------------------------------------
# 10. 저장 기능
# --------------------------------------------------

st.divider()

if st.button("💾 내 기록 저장하기", type="primary"):

    meanings = {}

    for _, row in edited_info.iterrows():
        meanings[row["열 이름"]] = (
            str(row["우리말 뜻"])
            if pd.notna(row["우리말 뜻"])
            else ""
        )

    save_records(
        meanings=meanings,
        source=source_text
    )

    st.success(
        "입력한 우리말 뜻과 데이터 출처가 저장되었습니다!"
    )

    st.caption(
        f"저장 위치: {RECORD_FILE}"
    )


# --------------------------------------------------
# 11. 하단 안내
# --------------------------------------------------

st.divider()

st.caption(
    "🧠 뇌졸중 예측 실습실 | 데이터 탐색 활동"
)

st.caption(
    "※ 이 데이터는 교육 및 실습 목적으로 사용합니다. "
    "실제 의료 진단이나 예측 결과로 사용하지 않습니다."
)
