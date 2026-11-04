import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 1. 페이지 설정
st.set_page_config(
    page_title="채점표 - 뇌졸중 예측 실습실",
    page_icon="📋",
    layout="wide"
)

# 2. 데이터 로드
@st.cache_data
def load_stroke_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

raw_df = load_stroke_data()

st.title("📋 모델 평가 및 채점표")
st.markdown("---")

# 속성 매핑 사전
FEATURE_MAP = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
REVERSE_FEATURE_MAP = {v: k for k, v in FEATURE_MAP.items()}

# 3. 데이터 분할 및 전처리 ('분류 모델' 페이지와 동일 방식)
df_sorted = raw_df.sort_index().copy()

# 번호 순 정렬 후 10명씩 묶어 앞 3명 테스트용(30%), 나머지 7명 훈련용(70%) 고정 분할
indices = np.arange(len(df_sorted))
test_mask = (indices % 10) < 3

train_df = df_sorted.iloc[~test_mask].copy()
test_df = df_sorted.iloc[test_mask].copy()

# 훈련 속성은 분류 모델 페이지의 기본값(bmi 제외 4개)과 동일하게 기본 설정
selected_features = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

X_train_raw = train_df[selected_features]
y_train = train_df["stroke"]
X_test_raw = test_df[selected_features]
y_test = test_df["stroke"]

# 스케일링 (로지스틱 회귀용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 4. 모델 학습 및 예측
# (1) 로지스틱 회귀(확률로 답하는 모델)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# (2) 의사결정트리(질문으로 답하는 모델)
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train_raw, y_train)
dt_pred = dt_model.predict(X_test_raw)

# (3) 훈련용에서 많은 쪽으로만 답하는 모델 (Baseline)
most_frequent_class = y_train.mode()[0]
base_pred = np.full_like(y_test, fill_value=most_frequent_class)

# 5. 혼동행렬 및 성능 지표 계산 함수
def compute_confusion_matrix_and_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (tp + fp) == 0:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
    else:
        precision = tp / (tp + fp)
        precision_str = f"{precision:.4f} ({precision*100:.2f}%)"
        if (precision + recall) == 0:
            f1_str = "계산할 수 없음"
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_str = f"{f1:.4f}"

    accuracy_str = f"{accuracy:.4f} ({accuracy*100:.2f}%)"
    recall_str = f"{recall:.4f} ({recall*100:.2f}%)"
    
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": accuracy_str,
        "Recall": recall_str,
        "Precision": precision_str,
        "F1": f1_str
    }

lr_metrics = compute_confusion_matrix_and_metrics(y_test.values, lr_pred)
dt_metrics = compute_confusion_matrix_and_metrics(y_test.values, dt_pred)
base_metrics = compute_confusion_matrix_and_metrics(y_test.values, base_pred)

# 6. 혼동행렬 표 출력
st.subheader("📊 테스트 데이터 혼동행렬 (Confusion Matrix)")

col_m1, col_m2 = st.columns(2)

def make_cm_dataframe(m):
    return pd.DataFrame([
        {
            "실제 \ 예측": "실제 뇌졸중 (1)",
            "예측 뇌졸중 (1)": f"{m['TP']} (TP)",
            "예측 아님 (0)": f"{m['FN']} (FN)",
            "행 합계": f"{m['TP'] + m['FN']} 명"
        },
        {
            "실제 \ 예측": "실제 아님 (0)",
            "예측 뇌졸중 (1)": f"{m['FP']} (FP)",
            "예측 아님 (0)": f"{m['TN']} (TN)",
            "행 합계": f"{m['FP'] + m['TN']} 명"
        },
        {
            "실제 \ 예측": "열 합계",
            "예측 뇌졸중 (1)": f"{m['TP'] + m['FP']} 명",
            "예측 아님 (0)": f"{m['FN'] + m['TN']} 명",
            "행 합계": f"{m['TP'] + m['FP'] + m['FN'] + m['TN']} 명"
        }
    ])

with col_m1:
    st.markdown("**[로지스틱 회귀(확률로 답하는 모델)]**")
    st.table(make_cm_dataframe(lr_metrics))

with col_m2:
    st.markdown("**[의사결정트리(질문으로 답하는 모델)]**")
    st.table(make_cm_dataframe(dt_metrics))

st.markdown("---")

# 7. 성능 지표 종합 나란히 비교 표
st.subheader("📈 모델 성능 지표 종합 비교")

metrics_summary_df = pd.DataFrame([
    {
        "모델 구분": "훈련용에서 많은 쪽으로만 답하는 모델",
        "정확도 (Accuracy)": base_metrics["Accuracy"],
        "재현율 (Recall)": base_metrics["Recall"],
        "정밀도 (Precision)": base_metrics["Precision"],
        "F1 점수": base_metrics["F1"]
    },
    {
        "모델 구분": "로지스틱 회귀(확률로 답하는 모델)",
        "정확도 (Accuracy)": lr_metrics["Accuracy"],
        "재현율 (Recall)": lr_metrics["Recall"],
        "정밀도 (Precision)": lr_metrics["Precision"],
        "F1 점수": lr_metrics["F1"]
    },
    {
        "모델 구분": "의사결정트리(질문으로 답하는 모델)",
        "정확도 (Accuracy)": dt_metrics["Accuracy"],
        "재현율 (Recall)": dt_metrics["Recall"],
        "정밀도 (Precision)": dt_metrics["Precision"],
        "F1 점수": dt_metrics["F1"]
    }
])

st.dataframe(metrics_summary_df, use_container_width=True)

st.markdown("---")

# 8. 오답 상세 분석 (놓친 환자 vs 헛짚은 사람)
st.subheader("🔍 오답 사례 상세 분석")

total_test_count = len(test_df)
stroke_test_count = int((y_test == 1).sum())
st.write(f"📌 **테스트용 전체 사람 수:** {total_test_count:,}명 | **그중 실제 뇌졸중(1) 환자 수:** {stroke_test_count:,}명")

model_choice = st.selectbox(
    "오답 목록을 확인할 모델을 선택하세요:",
    options=["로지스틱 회귀(확률로 답하는 모델)", "의사결정트리(질문으로 답하는 모델)"]
)

if model_choice == "로지스틱 회귀(확률로 답하는 모델)":
    current_pred = lr_pred
    current_metrics = lr_metrics
else:
    current_pred = dt_pred
    current_metrics = dt_metrics

eval_df = test_df.copy()
eval_df["pred"] = current_pred
eval_df["id_idx"] = eval_df.index

fn_df = eval_df[(eval_df["stroke"] == 1) & (eval_df["pred"] == 0)].copy()
fp_df = eval_df[(eval_df["stroke"] == 0) & (eval_df["pred"] == 1)].copy()

fn_df = fn_df.sort_values(by="age", ascending=False)
fp_df = fp_df.sort_values(by="age", ascending=False)

display_cols = ["id_idx", "age", "avg_glucose_level", "hypertension", "heart_disease"]
col_rename = {
    "id_idx": "번호",
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}

col_list1, col_list2 = st.columns(2)

with col_list1:
    st.markdown(f"#### 🚨 놓친 환자 (실제 뇌졸중이나 아니라고 예측) [전체: {len(fn_df)}명]")
    if len(fn_df) > 0:
        show_fn_df = fn_df[display_cols].rename(columns=col_rename)
        if len(fn_df) > 20:
            st.caption("※ 나이가 많은 순으로 상위 20명만 표시합니다.")
            st.dataframe(show_fn_df.head(20), use_container_width=True)
        else:
            st.dataframe(show_fn_df, use_container_width=True)
    else:
        st.info("놓친 환자가 없습니다.")

with col_list2:
    st.markdown(f"#### ⚠️ 헛짚은 사람 (뇌졸중이 아니나 뇌졸중으로 예측) [전체: {len(fp_df)}명]")
    if len(fp_df) > 0:
        show_fp_df = fp_df[display_cols].rename(columns=col_rename)
        if len(fp_df) > 20:
            st.caption("※ 나이가 많은 순으로 상위 20명만 표시합니다.")
            st.dataframe(show_fp_df.head(20), use_container_width=True)
        else:
            st.dataframe(show_fp_df, use_container_width=True)
    else:
        st.info("헛짚은 사람이 없습니다.")

st.info(f"💡 위 두 목록의 인원수(놓친 환자: {len(fn_df)}명, 헛짚은 사람: {len(fp_df)}명)는 상단 혼동행렬 표의 **FN({current_metrics['FN']}명)**, **FP({current_metrics['FP']}명)** 칸의 사람 수와 정확히 일치합니다.")
