import streamlit as st
import pandas as pd
import plotly.express as px

# 1. 페이지 설정
st.set_page_config(
    page_title="데이터 탐색 - 뇌졸중 예측 실습실",
    page_icon="🔍",
    layout="wide"
)

# 2. 데이터 로드 (동일 주소 및 UTF-8 인코딩)
@st.cache_data
def load_stroke_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_stroke_data()

st.title("🔍 데이터 탐색 페이지")
st.markdown("---")

# ----------------------------------------------------
# 1) 나이와 평균 혈당의 분포 (히스토그램 두 개 나란히)
# ----------------------------------------------------
st.subheader("1. 나이 및 평균 혈당 분포")
col_hist1, col_hist2 = st.columns(2)

with col_hist1:
    fig_age_hist = px.histogram(
        df, x="age", nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이"},
        color_discrete_sequence=["#4C78A8"]
    )
    st.plotly_chart(fig_age_hist, use_container_width=True)

with col_hist2:
    fig_glucose_hist = px.histogram(
        df, x="avg_glucose_level", nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당"},
        color_discrete_sequence=["#F28E2B"]
    )
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------
# 2) 뇌졸중 여부에 따른 나이/평균 혈당 비교 (상자그림 & 평균값 표)
# ----------------------------------------------------
st.subheader("2. 뇌졸중 여부별 나이 및 평균 혈당 비교")

# 레이블 가독성을 위한 열 추가
df_copy = df.copy()
df_copy["stroke_label"] = df_copy["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

col_box1, col_box2 = st.columns(2)

with col_box1:
    fig_age_box = px.box(
        df_copy, x="stroke_label", y="age", color="stroke_label",
        title="뇌졸중 여부별 나이 분포",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"}
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with col_box2:
    fig_glucose_box = px.box(
        df_copy, x="stroke_label", y="avg_glucose_level", color="stroke_label",
        title="뇌졸중 여부별 평균 혈당 분포",
        labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

# 두 그룹의 평균값 표 작성
mean_table = df.groupby("stroke")[["age", "avg_glucose_level"]].mean().reset_index()
mean_table["stroke"] = mean_table["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})
mean_table.columns = ["그룹", "평균 나이", "평균 혈당"]
st.markdown("**[두 그룹의 평균값 비교 표]**")
st.dataframe(mean_table, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------
# 3) 고혈압 및 심장병 유무에 따른 뇌졸중 비율 (막대그래프)
# ----------------------------------------------------
st.subheader("3. 기저질환(고혈압, 심장병) 유무에 따른 뇌졸중 비율")

col_bar1, col_bar2 = st.columns(2)

# 고혈압 비율 계산
ht_group = df.groupby("hypertension")["stroke"].mean().reset_index()
ht_group["hypertension_label"] = ht_group["hypertension"].map({0: "고혈압 없음 (0)", 1: "고혈압 있음 (1)"})
ht_group["stroke_rate_pct"] = ht_group["stroke"] * 100

with col_bar1:
    fig_ht = px.bar(
        ht_group, x="hypertension_label", y="stroke_rate_pct", text_auto=".2f",
        title="고혈압 유무별 뇌졸중 비율(%)",
        labels={"hypertension_label": "고혈압 유무", "stroke_rate_pct": "뇌졸중 비율 (%)"},
        color="hypertension_label"
    )
    fig_ht.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    st.plotly_chart(fig_ht, use_container_width=True)

# 심장병 비율 계산
hd_group = df.groupby("heart_disease")["stroke"].mean().reset_index()
hd_group["heart_disease_label"] = hd_group["heart_disease"].map({0: "심장병 없음 (0)", 1: "심장병 있음 (1)"})
hd_group["stroke_rate_pct"] = hd_group["stroke"] * 100

with col_bar2:
    fig_hd = px.bar(
        hd_group, x="heart_disease_label", y="stroke_rate_pct", text_auto=".2f",
        title="심장병 유무별 뇌졸중 비율(%)",
        labels={"heart_disease_label": "심장병 유무", "stroke_rate_pct": "뇌졸중 비율 (%)"},
        color="heart_disease_label"
    )
    fig_hd.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    st.plotly_chart(fig_hd, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------
# 4) BMI 결측치 대상 뇌졸중 비율 비교 표
# ----------------------------------------------------
st.subheader("4. 체질량지수(bmi) 결측치 그룹과 전체의 뇌졸중 비율 비교")

null_bmi_df = df[df["bmi"].isnull()]
null_bmi_count = len(null_bmi_df)
null_bmi_stroke_ratio = (null_bmi_df["stroke"].mean() * 100) if null_bmi_count > 0 else 0.0
total_stroke_ratio = df["stroke"].mean() * 100

bmi_compare_df = pd.DataFrame([
    {
        "구분": "bmi 결측치 그룹",
        "사람 수": null_bmi_count,
        "뇌졸중 비율 (%)": f"{null_bmi_stroke_ratio:.2f}%"
    },
    {
        "구분": "전체 그룹",
        "사람 수": len(df),
        "뇌졸중 비율 (%)": f"{total_stroke_ratio:.2f}%"
    }
])

st.dataframe(bmi_compare_df, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------
# 5) 흡연 상태별 사람 수 표
# ----------------------------------------------------
st.subheader("5. 흡연 상태(smoking_status)별 인원수")

smoke_counts = df["smoking_status"].value_counts().reset_index()
smoke_counts.columns = ["흡연 상태 (smoking_status)", "사람 수 (명)"]

st.dataframe(smoke_counts, use_container_width=True)
