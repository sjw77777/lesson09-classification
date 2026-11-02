
import streamlit as st
import pandas as pd
import plotly.express as px


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔍",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# 뇌졸중 여부를 우리말로 변환한 열 추가
df["stroke_label"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


# --------------------------------------------------
# 3. 제목
# --------------------------------------------------

st.title("🔍 데이터 탐색")

st.markdown(
    """
    ### 데이터를 다양한 그래프로 살펴보기

    히스토그램과 상자그림을 이용해 나이와 평균 혈당의 분포를 확인하고,
    고혈압, 심장병, 체질량지수, 흡연 상태에 따라
    뇌졸중 데이터가 어떻게 달라지는지 살펴봅시다.
    """
)

st.divider()


# --------------------------------------------------
# 4. 나이와 평균 혈당의 분포
# --------------------------------------------------

st.header("1. 나이와 평균 혈당의 분포")

st.write(
    "히스토그램은 데이터가 어느 구간에 많이 분포하는지 "
    "확인할 수 있는 그래프입니다."
)

col1, col2 = st.columns(2)

with col1:

    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이의 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        },
        color_discrete_sequence=["#4C78A8"]
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


with col2:

    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당의 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        },
        color_discrete_sequence=["#F58518"]
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )


st.divider()


# --------------------------------------------------
# 5. 뇌졸중 유무에 따른 나이와 평균 혈당 비교
# --------------------------------------------------

st.header("2. 뇌졸중 유무에 따른 비교")

st.write(
    "뇌졸중을 겪은 사람과 겪지 않은 사람의 "
    "나이와 평균 혈당 분포를 상자그림으로 비교해 봅시다."
)

# 결측값 제외
box_df = df.dropna(
    subset=["age", "avg_glucose_level", "stroke_label"]
)

col1, col2 = st.columns(2)

with col1:

    fig_box_age = px.box(
        box_df,
        x="stroke_label",
        y="age",
        color="stroke_label",
        title="뇌졸중 유무에 따른 나이 비교",
        labels={
            "stroke_label": "뇌졸중 여부",
            "age": "나이"
        },
        category_orders={
            "stroke_label": [
                "뇌졸중 없음",
                "뇌졸중 있음"
            ]
        },
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        }
    )

    fig_box_age.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_box_age,
        use_container_width=True
    )


with col2:

    fig_box_glucose = px.box(
        box_df,
        x="stroke_label",
        y="avg_glucose_level",
        color="stroke_label",
        title="뇌졸중 유무에 따른 평균 혈당 비교",
        labels={
            "stroke_label": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        },
        category_orders={
            "stroke_label": [
                "뇌졸중 없음",
                "뇌졸중 있음"
            ]
        },
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        }
    )

    fig_box_glucose.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_box_glucose,
        use_container_width=True
    )


# 평균값 계산
mean_table = (
    box_df.groupby("stroke_label")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reindex([
        "뇌졸중 없음",
        "뇌졸중 있음"
    ])
    .reset_index()
)

mean_table.columns = [
    "뇌졸중 여부",
    "평균 나이",
    "평균 혈당"
]

mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.subheader("두 그룹의 평균값 비교")

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# --------------------------------------------------
# 6. 고혈압 유무에 따른 뇌졸중 비율
# --------------------------------------------------

st.header("3. 고혈압 유무에 따른 뇌졸중 비율")

st.write(
    "고혈압이 있는 사람과 없는 사람의 뇌졸중 비율을 비교합니다."
)


def make_stroke_rate_table(data, column):

    result = (
        data.groupby(column, dropna=False)
        .agg(
            전체_사람_수=("stroke", "size"),
            뇌졸중_사람_수=("stroke", "sum")
        )
        .reset_index()
    )

    result["뇌졸중_비율"] = (
        result["뇌졸중_사람_수"]
        / result["전체_사람_수"]
        * 100
    )

    return result


hypertension_table = make_stroke_rate_table(
    df,
    "hypertension"
)

hypertension_table["고혈압 여부"] = (
    hypertension_table["hypertension"]
    .map({
        0: "고혈압 없음",
        1: "고혈압 있음"
    })
    .fillna("정보 없음")
)

fig_hypertension = px.bar(
    hypertension_table,
    x="고혈압 여부",
    y="뇌졸중_비율",
    text=hypertension_table["뇌졸중_비율"].map(
        lambda x: f"{x:.2f}%"
    ),
    title="고혈압 유무에 따른 뇌졸중 비율",
    labels={
        "고혈압 여부": "고혈압 여부",
        "뇌졸중_비율": "뇌졸중 비율(%)"
    },
    color="고혈압 여부",
    color_discrete_map={
        "고혈압 없음": "#4C78A8",
        "고혈압 있음": "#E45756"
    }
)

fig_hypertension.update_traces(
    textposition="outside"
)

fig_hypertension.update_layout(
    showlegend=False,
    yaxis_title="뇌졸중 비율(%)"
)

st.plotly_chart(
    fig_hypertension,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# 7. 심장병 유무에 따른 뇌졸중 비율
# --------------------------------------------------

st.header("4. 심장병 유무에 따른 뇌졸중 비율")

st.write(
    "심장병이 있는 사람과 없는 사람의 뇌졸중 비율을 비교합니다."
)

heart_table = make_stroke_rate_table(
    df,
    "heart_disease"
)

heart_table["심장병 여부"] = (
    heart_table["heart_disease"]
    .map({
        0: "심장병 없음",
        1: "심장병 있음"
    })
    .fillna("정보 없음")
)

fig_heart = px.bar(
    heart_table,
    x="심장병 여부",
    y="뇌졸중_비율",
    text=heart_table["뇌졸중_비율"].map(
        lambda x: f"{x:.2f}%"
    ),
    title="심장병 유무에 따른 뇌졸중 비율",
    labels={
        "심장병 여부": "심장병 여부",
        "뇌졸중_비율": "뇌졸중 비율(%)"
    },
    color="심장병 여부",
    color_discrete_map={
        "심장병 없음": "#4C78A8",
        "심장병 있음": "#E45756"
    }
)

fig_heart.update_traces(
    textposition="outside"
)

fig_heart.update_layout(
    showlegend=False,
    yaxis_title="뇌졸중 비율(%)"
)

st.plotly_chart(
    fig_heart,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# 8. BMI 결측값 분석
# --------------------------------------------------

st.header("5. 체질량지수(BMI) 빈 값 분석")

st.write(
    "체질량지수가 비어 있는 사람들만 따로 모아 "
    "뇌졸중 비율을 전체 데이터의 뇌졸중 비율과 비교합니다."
)

# BMI가 비어 있는 사람
missing_bmi_df = df[df["bmi"].isna()]

# BMI가 비어 있지 않은 사람
valid_bmi_df = df[df["bmi"].notna()]

# 전체 뇌졸중 비율
overall_stroke_rate = (
    df["stroke"].mean() * 100
)

# BMI 결측 그룹의 뇌졸중 비율
missing_bmi_stroke_rate = (
    missing_bmi_df["stroke"].mean() * 100
    if len(missing_bmi_df) > 0
    else 0
)

bmi_comparison = pd.DataFrame({
    "구분": [
        "BMI가 비어 있는 사람",
        "전체 사람"
    ],
    "사람 수": [
        len(missing_bmi_df),
        len(df)
    ],
    "뇌졸중 사람 수": [
        int(missing_bmi_df["stroke"].sum()),
        int(df["stroke"].sum())
    ],
    "뇌졸중 비율(%)": [
        round(missing_bmi_stroke_rate, 2),
        round(overall_stroke_rate, 2)
    ]
})

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True
)

st.info(
    f"BMI 결측 인원: {len(missing_bmi_df):,}명 "
    f"/ 전체 인원: {len(df):,}명"
)


st.divider()


# --------------------------------------------------
# 9. 흡연 상태별 사람 수
# --------------------------------------------------

st.header("6. 흡연 상태별 사람 수")

st.write(
    "흡연 상태의 종류별로 사람 수를 세어 봅시다. "
    "흡연 상태가 비어 있는 경우도 별도로 표시합니다."
)

smoking_table = (
    df["smoking_status"]
    .fillna("정보 없음")
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# --------------------------------------------------
# 10. 마무리
# --------------------------------------------------

st.subheader("💡 탐색 활동 정리")

st.markdown(
    """
    다음 질문에 답하면서 데이터를 정리해 봅시다.

    1. 나이와 평균 혈당은 어떤 구간에 가장 많이 분포하나요?
    2. 뇌졸중을 겪은 사람과 겪지 않은 사람의 평균 나이는 어떻게 다른가요?
    3. 고혈압이 있는 사람과 없는 사람의 뇌졸중 비율은 어떻게 다른가요?
    4. 심장병이 있는 사람의 뇌졸중 비율은 어떻게 나타나나요?
    5. BMI가 비어 있는 사람들의 뇌졸중 비율은 전체 비율과 어떤 차이가 있나요?
    6. 가장 많은 사람이 속한 흡연 상태는 무엇인가요?
    """
)

st.caption(
    "※ 이 분석은 데이터의 통계적 특징을 살펴보는 교육 활동입니다. "
    "그룹 간 차이가 인과관계를 의미하지는 않습니다."
)

st.caption(
    "🧠 뇌졸중 예측 실습실 | 데이터 탐색 페이지"
)
