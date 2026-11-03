
import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# 1. 기본 설정
st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/sjw77777/data/refs/heads/main/stroke.csv"

RANDOM_STATE = 42

FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

KOREAN_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}


# 2. 데이터 불러오기
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    data = data.sort_values("id").reset_index(drop=True)
    return data


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# 3. 제목
st.title("📝 채점표")

st.markdown("""
### 분류 모델의 예측 결과 채점하기

- 양성(1): 뇌졸중
- 음성(0): 뇌졸중 아님

혼동행렬을 이용하여 두 모델의 예측 결과를 비교해 봅시다.
""")

st.divider()


# 4. 데이터 분리
st.header("1. 훈련 데이터와 테스트 데이터")

test_positions = [
    i for i in range(len(df))
    if i % 10 < 3
]

train_positions = [
    i for i in range(len(df))
    if i % 10 >= 3
]

train_df = df.iloc[train_positions].copy()
test_df = df.iloc[test_positions].copy()

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# 훈련 데이터로만 표준화
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

test_people = len(test_df)
test_positive = int((y_test == 1).sum())

st.write(
    f"**테스트용 사람 수: {test_people:,}명 | "
    f"그중 실제 뇌졸중인 사람 수: {test_positive:,}명**"
)

col1, col2 = st.columns(2)

col1.metric("훈련용 사람 수", f"{len(train_df):,}명")
col2.metric("테스트용 사람 수", f"{test_people:,}명")

if test_people != 1533:
    st.warning(
        f"테스트 데이터가 1,533명이 아닙니다. "
        f"현재 {test_people:,}명입니다."
    )

st.caption(
    "사용 속성: " +
    ", ".join(KOREAN_NAMES[f] for f in FEATURES)
)

st.divider()


# 5. 로지스틱 회귀 모델
st.header("2. 분류 모델 학습")

logistic_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000
)

logistic_model.fit(X_train_scaled, y_train)

logistic_train_pred = logistic_model.predict(X_train_scaled)
logistic_test_pred = logistic_model.predict(X_test_scaled)


# 6. 의사결정트리 모델
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

tree_model.fit(X_train, y_train)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)


# 7. 기준 모델
majority_class = int(y_train.mode().iloc[0])

baseline_train_pred = [majority_class] * len(y_train)
baseline_test_pred = [majority_class] * len(y_test)


# 8. 혼동행렬 계산
def calculate_confusion(y_true, y_pred):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


logistic_cm = calculate_confusion(y_test, logistic_test_pred)
tree_cm = calculate_confusion(y_test, tree_test_pred)
baseline_cm = calculate_confusion(y_test, baseline_test_pred)


# 9. 혼동행렬 표
def make_confusion_table(cm):
    tn = cm["TN"]
    fp = cm["FP"]
    fn = cm["FN"]
    tp = cm["TP"]

    return pd.DataFrame(
        [
            [
                f"TN: {tn:,}명",
                f"FP: {fp:,}명",
                f"{tn + fp:,}명"
            ],
            [
                f"FN: {fn:,}명",
                f"TP: {tp:,}명",
                f"{fn + tp:,}명"
            ],
            [
                f"{tn + fn:,}명",
                f"{fp + tp:,}명",
                f"{tn + fp + fn + tp:,}명"
            ]
        ],
        index=[
            "실제 뇌졸중 아님 (0)",
            "실제 뇌졸중 (1)",
            "열 합계"
        ],
        columns=[
            "예측 뇌졸중 아님 (0)",
            "예측 뇌졸중 (1)",
            "행 합계"
        ]
    )


st.header("3. 테스트 데이터 혼동행렬")

st.subheader("로지스틱 회귀(확률로 답하는 모델)")
st.table(make_confusion_table(logistic_cm))

st.subheader("의사결정트리(질문으로 답하는 모델)")
st.table(make_confusion_table(tree_cm))

st.subheader("기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)")
st.write(f"훈련 데이터의 다수 클래스: {majority_class}")
st.table(make_confusion_table(baseline_cm))

st.divider()


# 10. 평가 지표 계산
st.header("4. 네 가지 평가 지표 비교")


def calculate_metrics(cm):
    tp = cm["TP"]
    tn = cm["TN"]
    fp = cm["FP"]
    fn = cm["FN"]

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else None

    recall_denominator = tp + fn
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else None
    )

    precision_denominator = tp + fp

    if precision_denominator == 0:
        precision = None
        f1 = None
    else:
        precision = tp / precision_denominator

        if recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def format_metric(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value * 100:.2f}%"


logistic_metrics = calculate_metrics(logistic_cm)
tree_metrics = calculate_metrics(tree_cm)
baseline_metrics = calculate_metrics(baseline_cm)

metrics_table = pd.DataFrame([
    {
        "모델": "로지스틱 회귀(확률로 답하는 모델)",
        "정확도": format_metric(logistic_metrics["정확도"]),
        "재현율": format_metric(logistic_metrics["재현율"]),
        "정밀도": format_metric(logistic_metrics["정밀도"]),
        "F1": format_metric(logistic_metrics["F1"])
    },
    {
        "모델": "의사결정트리(질문으로 답하는 모델)",
        "정확도": format_metric(tree_metrics["정확도"]),
        "재현율": format_metric(tree_metrics["재현율"]),
        "정밀도": format_metric(tree_metrics["정밀도"]),
        "F1": format_metric(tree_metrics["F1"])
    },
    {
        "모델": "기준 모델(훈련용 다수 쪽)",
        "정확도": format_metric(baseline_metrics["정확도"]),
        "재현율": format_metric(baseline_metrics["재현율"]),
        "정밀도": format_metric(baseline_metrics["정밀도"]),
        "F1": format_metric(baseline_metrics["F1"])
    }
])

st.dataframe(
    metrics_table,
    use_container_width=True,
    hide_index=True
)


with st.expander("평가 지표 계산 방법"):
    st.markdown("""
    - **정확도**: (TP + TN) / 전체 사람 수
    - **재현율**: TP / (TP + FN)
    - **정밀도**: TP / (TP + FP)
    - **F1**: 2 × 정밀도 × 재현율 / (정밀도 + 재현율)

    뇌졸중이라고 예측한 사람이 0명이면 정밀도와 F1은
    계산할 수 없음으로 표시합니다.
    """)

st.divider()


# 11. 오답 목록
st.header("5. 놓친 환자와 헛짚은 사람 확인하기")

st.write("""
- **놓친 환자(FN)**: 실제 뇌졸중인데 아니라고 예측한 사람
- **헛짚은 사람(FP)**: 실제 뇌졸중이 아닌데 뇌졸중이라고 예측한 사람
""")

model_names = {
    "로지스틱 회귀(확률로 답하는 모델)": logistic_test_pred,
    "의사결정트리(질문으로 답하는 모델)": tree_test_pred,
    "기준 모델(훈련용 다수 쪽)": baseline_test_pred
}

selected_model_name = st.selectbox(
    "확인할 모델을 선택하세요.",
    options=list(model_names.keys())
)

selected_predictions = model_names[selected_model_name]


# 12. 실제 정답과 예측 결과 결합
result_df = test_df.copy()

result_df["실제 정답"] = y_test.to_numpy()
result_df["예측 결과"] = selected_predictions


# 놓친 환자
missed_patients = result_df[
    (result_df["실제 정답"] == 1) &
    (result_df["예측 결과"] == 0)
].copy()

# 헛짚은 사람
false_alarm_people = result_df[
    (result_df["실제 정답"] == 0) &
    (result_df["예측 결과"] == 1)
].copy()

# 나이가 많은 순서로 정렬
missed_patients = missed_patients.sort_values(
    "age",
    ascending=False,
    kind="stable"
)

false_alarm_people = false_alarm_people.sort_values(
    "age",
    ascending=False,
    kind="stable"
)


# 13. 표시용 표
def make_person_table(data):
    display_df = data[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease"
        ]
    ].copy()

    display_df.columns = [
        "번호",
        "나이",
        "평균 혈당",
        "고혈압 여부",
        "심장병 여부"
    ]

    display_df["고혈압 여부"] = (
        display_df["고혈압 여부"]
        .map({0: "없음", 1: "있음"})
    )

    display_df["심장병 여부"] = (
        display_df["심장병 여부"]
        .map({0: "없음", 1: "있음"})
    )

    display_df["나이"] = display_df["나이"].round(1)
    display_df["평균 혈당"] = display_df["평균 혈당"].round(2)

    return display_df


# 14. 두 목록 나란히 표시
left_col, right_col = st.columns(2)

with left_col:
    st.subheader(
        f"놓친 환자(FN): {len(missed_patients):,}명"
    )

    st.caption(
        "실제 뇌졸중인데 뇌졸중이 아니라고 예측한 사람"
    )

    st.dataframe(
        make_person_table(missed_patients.head(20)),
        use_container_width=True,
        hide_index=True
    )

    if len(missed_patients) > 20:
        st.caption(
            f"전체 {len(missed_patients):,}명 중 "
            "나이가 많은 앞 20명만 표시했습니다."
        )


with right_col:
    st.subheader(
        f"헛짚은 사람(FP): {len(false_alarm_people):,}명"
    )

    st.caption(
        "실제 뇌졸중이 아닌데 뇌졸중이라고 예측한 사람"
    )

    st.dataframe(
        make_person_table(false_alarm_people.head(20)),
        use_container_width=True,
        hide_index=True
    )

    if len(false_alarm_people) > 20:
        st.caption(
            f"전체 {len(false_alarm_people):,}명 중 "
            "나이가 많은 앞 20명만 표시했습니다."
        )


# 15. 혼동행렬과 목록 인원 확인
selected_cm = calculate_confusion(
    y_test,
    selected_predictions
)

st.divider()

st.success(
    f"놓친 환자 {len(missed_patients):,}명 = "
    f"혼동행렬 FN {selected_cm['FN']:,}명 | "
    f"헛짚은 사람 {len(false_alarm_people):,}명 = "
    f"혼동행렬 FP {selected_cm['FP']:,}명"
)


# 16. 마무리
st.divider()

st.header("6. 채점 활동 정리")

st.info("""
1. 정확도가 가장 높은 모델은 무엇인가요?
2. 실제 뇌졸중 환자를 가장 많이 찾아낸 모델은 무엇인가요?
3. 놓친 환자가 가장 적은 모델은 무엇인가요?
4. 헛짚은 사람이 가장 적은 모델은 무엇인가요?
5. 정확도만으로 좋은 뇌졸중 예측 모델을 판단해도 될까요?
6. 뇌졸중 예측에서는 재현율과 정밀도 중 어떤 지표가 중요할까요?
""")

st.caption("이 페이지는 교육용 분류 모델 평가 실습을 위한 자료입니다.")
st.caption("© 뇌졸중 예측 실습실")
