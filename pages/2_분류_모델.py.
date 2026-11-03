
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 페이지 설정 및 데이터 불러오기
# --------------------------------------------------
st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "sjw77777/data/refs/heads/main/stroke.csv"
)

KOREAN_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
]

st.title("🧠 분류 모델")
st.write(
    "선택한 건강 관련 속성을 이용해 뇌졸중 여부를 분류합니다. "
    "이 페이지는 분류 모델을 학습하는 교육용 실습이며, "
    "실제 질병 진단이나 개인의 발병 위험 판단에 사용할 수 없습니다."
)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    required = ["stroke"] + FEATURES
    missing_columns = [
        col for col in required if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"데이터에 필요한 열이 없습니다: {missing_columns}"
        )

    # 번호 순으로 정렬한다. id 열이 없으면 원래 순서를 사용한다.
    if "id" in data.columns:
        data = data.sort_values(
            "id", kind="stable"
        ).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    return data


try:
    data = load_data()
except Exception as error:
    st.error(f"데이터를 불러오지 못했습니다: {error}")
    st.stop()

# 이진 분류의 정답은 0 또는 1이어야 한다.
data = data.dropna(subset=["stroke"]).copy()
data["stroke"] = pd.to_numeric(
    data["stroke"], errors="coerce"
)
data = data.dropna(subset=["stroke"]).copy()
data["stroke"] = data["stroke"].astype(int)

if not data["stroke"].isin([0, 1]).all():
    st.error("stroke 열에는 0 또는 1만 있어야 합니다.")
    st.stop()

# 선택된 열은 숫자로 변환한다.
for col in FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# --------------------------------------------------
# 속성 선택
# --------------------------------------------------
st.subheader("1. 입력 속성 선택")

selected_features = st.multiselect(
    "모델 학습에 사용할 속성을 고르세요.",
    options=FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
    ],
    format_func=lambda col: KOREAN_NAMES[col],
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해 주세요.")
    st.stop()

# --------------------------------------------------
# 훈련용 / 테스트용 데이터 분리
# 열 명씩 묶었을 때 앞 세 명은 테스트용,
# 나머지 일곱 명은 훈련용으로 고정한다.
# --------------------------------------------------
positions = np.arange(len(data))
test_mask = positions % 10 < 3

train_data = data.loc[~test_mask].copy()
test_data = data.loc[test_mask].copy()

if len(test_data) != 1533:
    st.warning(
        f"테스트 데이터가 {len(test_data):,}명입니다. "
        "요청한 1,533명과 다르므로 데이터의 행 수를 확인해 주세요."
    )

X_train = train_data[selected_features].copy()
X_test = test_data[selected_features].copy()

y_train = train_data["stroke"].copy()
y_test = test_data["stroke"].copy()

# bmi를 선택한 경우에만 훈련용 중앙값으로 결측값을 채운다.
bmi_median = None

if "bmi" in selected_features:
    bmi_imputer = SimpleImputer(strategy="median")

    X_train.loc[:, ["bmi"]] = bmi_imputer.fit_transform(
        X_train[["bmi"]]
    )
    X_test.loc[:, ["bmi"]] = bmi_imputer.transform(
        X_test[["bmi"]]
    )
    bmi_median = float(bmi_imputer.statistics_[0])

# 다른 선택 속성에 결측값이 있으면 모델 학습을 중단하고 알린다.
if X_train.isna().any().any() or X_test.isna().any().any():
    st.error(
        "선택한 속성에 BMI 외의 결측값이 있습니다. "
        "이 실습에서는 BMI만 훈련용 중앙값으로 채우도록 설정되어 있습니다."
    )
    st.stop()

# --------------------------------------------------
# 모델 학습
# --------------------------------------------------

# 로지스틱 회귀: 크기 맞추기는 훈련 데이터만 사용한다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    random_state=42,
    max_iter=2000,
)
logistic_model.fit(X_train_scaled, y_train)

# 의사결정트리: 질문 최대 3번, 마지막 마디 최소 5명,
# 난수 고정.
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
)
tree_model.fit(X_train, y_train)

# 입력을 보지 않고 훈련용에서 더 많은 쪽으로만 답하는 기준 모델
majority_class = int(y_train.value_counts().idxmax())

# --------------------------------------------------
# 정확도 계산
# --------------------------------------------------
log_train_pred = logistic_model.predict(X_train_scaled)
log_test_pred = logistic_model.predict(X_test_scaled)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

baseline_train_pred = np.full(len(y_train), majority_class)
baseline_test_pred = np.full(len(y_test), majority_class)

scores = [
    {
        "name": "로지스틱 회귀(확률로 답하는 모델)",
        "train": accuracy_score(y_train, log_train_pred),
        "test": accuracy_score(y_test, log_test_pred),
    },
    {
        "name": "의사결정트리(질문으로 답하는 모델)",
        "train": accuracy_score(y_train, tree_train_pred),
        "test": accuracy_score(y_test, tree_test_pred),
    },
    {
        "name": "입력을 보지 않는 기준 모델",
        "train": accuracy_score(y_train, baseline_train_pred),
        "test": accuracy_score(y_test, baseline_test_pred),
    },
]

st.subheader("2. 모델 정확도")

st.caption(
    f"훈련용: {len(X_train):,}명 · "
    f"테스트용: {len(X_test):,}명 · "
    f"양성은 stroke=1(뇌졸중), 음성은 stroke=0(뇌졸중 아님)"
)

score_columns = st.columns(3)

for column, score in zip(score_columns, scores):
    with column:
        st.markdown(f"**{score['name']}**")
        st.metric(
            label="테스트 데이터 정확도",
            value=f"{score['test']:.1%}",
        )
        small_train, small_test = st.columns(2)
        small_train.metric(
            "훈련 데이터",
            f"{score['train']:.1%}",
        )
        small_test.metric(
            "테스트 데이터",
            f"{score['test']:.1%}",
        )

st.caption(
    "정확도는 전체 예측 중 정답을 맞힌 비율입니다. "
    "데이터에 뇌졸중이 아닌 사람이 훨씬 많다면, "
    "기준 모델도 높은 정확도를 보일 수 있습니다."
)

if "bmi" in selected_features:
    st.info(
        f"체질량지수의 결측값은 훈련 데이터의 중앙값 "
        f"{bmi_median:.3f}으로 채웠습니다."
    )
else:
    st.info("체질량지수(bmi)는 선택하지 않아 결측값 처리를 하지 않았습니다.")

# --------------------------------------------------
# 축 선택
# --------------------------------------------------
st.subheader("3. 두 속성으로 분류 결과 살펴보기")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축",
        options=selected_features,
        index=0,
        format_func=lambda col: KOREAN_NAMES[col],
    )

remaining_features = [
    col for col in selected_features if col != x_feature
]

with axis_col2:
    y_feature = st.selectbox(
        "세로축",
        options=remaining_features,
        index=0,
        format_func=lambda col: KOREAN_NAMES[col],
    )

# 축 이외의 속성은 테스트 데이터의 중앙값에 고정한다.
fixed_values = {}

for col in selected_features:
    if col not in [x_feature, y_feature]:
        fixed_values[col] = float(X_test[col].median())

if fixed_values:
    fixed_text = ", ".join(
        f"{KOREAN_NAMES[col]} = {value:.3f}"
        for col, value in fixed_values.items()
    )
    st.write(
        "**두 축 이외의 속성을 고정한 값:** " + fixed_text
    )
else:
    st.write("선택한 속성이 두 개뿐이므로 고정한 다른 속성은 없습니다.")

# --------------------------------------------------
# Plotly 산점도 및 모델 영역
# --------------------------------------------------
x_min = float(min(X_train[x_feature].min(), X_test[x_feature].min()))
x_max = float(max(X_train[x_feature].max(), X_test[x_feature].max()))
y_min = float(min(X_train[y_feature].min(), X_test[y_feature].min()))
y_max = float(max(X_train[y_feature].max(), X_test[y_feature].max()))

# 그림 가장자리에 약간의 여백을 둔다.
x_padding = (x_max - x_min) * 0.06 if x_max != x_min else 1.0
y_padding = (y_max - y_min) * 0.06 if y_max != y_min else 1.0

x_range = [x_min - x_padding, x_max + x_padding]
y_range = [y_min - y_padding, y_max + y_padding]

x_grid = np.linspace(x_range[0], x_range[1], 140)
y_grid = np.linspace(y_range[0], y_range[1], 140)
xx, yy = np.meshgrid(x_grid, y_grid)

grid_data = pd.DataFrame(index=np.arange(xx.size))

for col in selected_features:
    if col == x_feature:
        grid_data[col] = xx.ravel()
    elif col == y_feature:
        grid_data[col] = yy.ravel()
    else:
        grid_data[col] = fixed_values[col]

# 트리의 분류 영역
tree_grid_pred = tree_model.predict(grid_data).reshape(xx.shape)

# 로지스틱 회귀의 양성 확률
grid_scaled = scaler.transform(grid_data[selected_features])
log_grid_prob = logistic_model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

fig = go.Figure()

# 트리의 0/1 분류 영역을 옅은 색으로 표시
fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=tree_grid_pred,
        name="의사결정트리 분류 영역",
        showscale=False,
        opacity=0.20,
        colorscale=[
            [0.0, "royalblue"],
            [0.499, "royalblue"],
            [0.5, "tomato"],
            [1.0, "tomato"],
        ],
        contours=dict(
            start=0,
            end=1,
            size=1,
            coloring="heatmap",
            showlines=False,
        ),
        hoverinfo="skip",
    )
)

# 로지스틱 회귀가 양성 확률 0.5로 가르는 선
fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=log_grid_prob,
        name="로지스틱 회귀 확률 0.5 경계",
        showscale=False,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="lines",
            showlines=True,
        ),
        line=dict(color="black", width=3),
        hoverinfo="skip",
    )
)

# 실제 테스트 데이터: stroke=0 / stroke=1로 구분
for label, color, name in [
    (0, "royalblue", "실제 뇌졸중 아님 (stroke=0)"),
    (1, "tomato", "실제 뇌졸중 (stroke=1)"),
]:
    subset = test_data.loc[test_data["stroke"] == label]

    fig.add_trace(
        go.Scatter(
            x=subset[x_feature],
            y=subset[y_feature],
            mode="markers",
            name=name,
            marker=dict(
                color=color,
                size=7,
                opacity=0.8,
                line=dict(color="white", width=0.5),
            ),
            customdata=np.stack(
                [
                    subset["stroke"].to_numpy(),
                    subset.index.to_numpy(),
                ],
                axis=-1,
            ),
            hovertemplate=(
                f"{KOREAN_NAMES[x_feature]}: %{{x:.3f}}<br>"
                f"{KOREAN_NAMES[y_feature]}: %{{y:.3f}}<br>"
                "실제 뇌졸중 여부: %{customdata[0]}<extra></extra>"
            ),
        )
    )

fig.update_layout(
    title="테스트 데이터와 두 모델의 분류 결과",
    xaxis_title=KOREAN_NAMES[x_feature],
    yaxis_title=KOREAN_NAMES[y_feature],
    legend_title="범례",
    height=650,
    template="plotly_white",
)

fig.update_xaxes(range=x_range)
fig.update_yaxes(range=y_range)

st.plotly_chart(fig, use_container_width=True)

# 0.5 경계선이 실제 표시 범위 안에 존재하는지 확인한다.
inside_boundary = (
    np.nanmin(log_grid_prob) <= 0.5
    and np.nanmax(log_grid_prob) >= 0.5
)

if inside_boundary:
    st.write(
        "검은 선은 로지스틱 회귀 모델이 뇌졸중 양성 확률을 "
        "0.5로 판단하는 경계입니다."
    )
else:
    st.write(
        "로지스틱 회귀의 확률 0.5 경계선은 현재 그림 범위 안에 "
        "나타나지 않습니다."
    )

st.caption(
    "배경의 옅은 파란색과 붉은색 영역은 의사결정트리의 예측입니다. "
    "점의 색은 모델의 예측이 아니라 실제 테스트 데이터의 정답을 나타냅니다."
)

# --------------------------------------------------
# 의사결정트리 DOT 그림
# --------------------------------------------------
st.subheader("4. 의사결정트리가 던진 질문")

tree = tree_model.tree_
decision_path = tree_model.decision_path(X_train)
leaf_ids = tree_model.apply(X_train)

# 각 노드에 실제로 도달한 훈련 데이터의 정답을 모아 정확한 수를 센다.
node_train_indices = [
    decision_path[:, node_id].nonzero()[0]
    for node_id in range(tree.node_count)
]

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white", nodesep=0.35, ranksep=0.5];',
    'node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];',
    'edge [fontname="Arial", fontsize=10];',
]

used_features = set()
leaf_count = 0
negative_leaf_count = 0

for node_id in range(tree.node_count):
    sample_indices = node_train_indices[node_id]
    people_count = len(sample_indices)

    node_labels = y_train.iloc[sample_indices]
    positive_count = int((node_labels == 1).sum())
    positive_rate = (
        positive_count / people_count if people_count else 0.0
    )

    if tree.children_left[node_id] == tree.children_right[node_id]:
        # 리프 노드
        leaf_count += 1
        predicted_class = int(tree_model.classes_[np.argmax(tree.value[node_id][0])])

        if predicted_class == 0:
            negative_leaf_count += 1
            fill_color = "#DCEBFF"
            answer_text = "답: 뇌졸중 아님"
        else:
            fill_color = "#FFDCDC"
            answer_text = "답: 뇌졸중"

        node_text = (
            f"마지막 마디 {node_id}\\n"
            f"훈련용 사람: {people_count}명\\n"
            f"실제 뇌졸중: {positive_count}명 ({positive_rate:.1%})\\n"
            f"{answer_text}"
        )
    else:
        # 질문 노드
        feature_index = tree.feature[node_id]
        feature_name = FEATURES[feature_index]
        used_features.add(feature_name)

        threshold = tree.threshold[node_id]
        fill_color = "#F3F3F3"

        node_text = (
            f"질문 {node_id}\\n"
            f"{KOREAN_NAMES[feature_name]} ≤ {threshold:.3f}?\\n"
            f"훈련용 사람: {people_count}명\\n"
            f"실제 뇌졸중: {positive_count}명 ({positive_rate:.1%})"
        )

    escaped_text = html.escape(node_text, quote=True)
    dot_lines.append(
        f'{node_id} [label="{escaped_text}", fillcolor="{fill_color}"];'
    )

    if tree.children_left[node_id] != tree.children_right[node_id]:
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        dot_lines.append(
            f'{node_id} -> {left_child} [label="예"];'
        )
        dot_lines.append(
            f'{node_id} -> {right_child} [label="아니요"];'
        )

dot_lines.append("}")
dot_string = "\n".join(dot_lines)

st.graphviz_chart(dot_string, use_container_width=True)

st.write(
    f"**답을 내는 마지막 마디:** {leaf_count}칸"
)
st.write(
    f"**그중 뇌졸중 아님(0)이라고 답하는 마디:** "
    f"{negative_leaf_count}칸"
)

if used_features:
    used_names = [KOREAN_NAMES[col] for col in FEATURES if col in used_features]
    st.write(
        "**이 나무가 실제로 물은 속성:** " + ", ".join(used_names)
    )
else:
    st.write("**이 나무가 실제로 물은 속성:** 질문 없음")

st.caption(
    "마디의 사람 수와 실제 뇌졸중 인원·비율은 훈련 데이터만을 기준으로 계산했습니다. "
    "가지의 '예'는 해당 질문의 조건을 만족하는 경우이고, "
    "'아니요'는 만족하지 않는 경우입니다."
)
