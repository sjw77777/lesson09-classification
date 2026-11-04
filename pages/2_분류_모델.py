import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 1. 페이지 설정
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

# 2. 데이터 로드
@st.cache_data
def load_stroke_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

raw_df = load_stroke_data()

st.title("🤖 머신러닝 분류 모델 실습")
st.markdown("---")

# 속성 우리말 이름 매핑 Dictionary
FEATURE_MAP = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
REVERSE_FEATURE_MAP = {v: k for k, v in FEATURE_MAP.items()}

# 3. 입력 속성 선택 (기본값: bmi 제외 4개)
default_selected_korean = ["나이", "평균 혈당", "고혈압", "심장병"]
selected_korean_features = st.multiselect(
    "학습에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=list(FEATURE_MAP.values()),
    default=default_selected_korean
)

# 선택 속성 검증
if len(selected_korean_features) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해 주세요. 속성이 2개 미만이면 모델 학습 및 시각화를 진행할 수 없습니다.")
    st.stop()

selected_features = [REVERSE_FEATURE_MAP[k] for k in selected_korean_features]

# 4. 데이터 분할 및 전처리
# (1) 번호(index) 순 정렬
df_sorted = raw_df.sort_index().copy()

# (2) 10명씩 묶어 앞 3명 테스트용(30%), 나머지 7명 훈련용(70%) 고정 분할
indices = np.arange(len(df_sorted))
test_mask = (indices % 10) < 3

train_df = df_sorted.iloc[~test_mask].copy()
test_df = df_sorted.iloc[test_mask].copy()

# (3) bmi 선택 시 훈련용 데이터 중앙값으로 결측치 채우기
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)

X_train_raw = train_df[selected_features]
y_train = train_df["stroke"]
X_test_raw = test_df[selected_features]
y_test = test_df["stroke"]

# (4) 스케일링 (로지스틱 회귀용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 5. 모델 학습 및 평가
# (1) 다수결 다수파 모델 (Baseline)
most_frequent_class = y_train.mode()[0]
base_train_acc = (y_train == most_frequent_class).mean()
base_test_acc = (y_test == most_frequent_class).mean()

# (2) 로지스틱 회귀
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
lr_train_acc = log_reg.score(X_train_scaled, y_train)
lr_test_acc = log_reg.score(X_test_scaled, y_test)

# (3) 의사결정트리
dt_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_clf.fit(X_train_raw, y_train)
dt_train_acc = dt_clf.score(X_train_raw, y_train)
dt_test_acc = dt_clf.score(X_test_raw, y_test)

# 6. 모델 성능 정확도 카드 출력
st.subheader("🎯 모델 정확도 비교")

col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    st.metric(
        label="입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델",
        value=f"{base_test_acc * 100:.2f}%"
    )
    st.caption(f"훈련 데이터 정확도: {base_train_acc * 100:.2f}% | 테스트 데이터 정확도: {base_test_acc * 100:.2f}%")

with col_c2:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{lr_test_acc * 100:.2f}%"
    )
    st.caption(f"훈련 데이터 정확도: {lr_train_acc * 100:.2f}% | 테스트 데이터 정확도: {lr_test_acc * 100:.2f}%")

with col_c3:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{dt_test_acc * 100:.2f}%"
    )
    st.caption(f"훈련 데이터 정확도: {dt_train_acc * 100:.2f}% | 테스트 데이터 정확도: {dt_test_acc * 100:.2f}%")

st.markdown("---")

# 7. 산점도 및 결정 경계 시각화
st.subheader("📐 2차원 시각화 및 결정 경계")

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    x_axis_korean = st.selectbox("가로축(X축) 속성 고르기", selected_korean_features, index=0)
with col_sel2:
    y_default_idx = 1 if len(selected_korean_features) > 1 else 0
    y_axis_korean = st.selectbox("세로축(Y축) 속성 고르기", selected_korean_features, index=y_default_idx)

if x_axis_korean == y_axis_korean:
    st.info("💡 가로축과 세로축으로 서로 다른 속성을 고르시면 2차원 평면으로 더 폭넓게 관찰할 수 있습니다.")

x_feat = REVERSE_FEATURE_MAP[x_axis_korean]
y_feat = REVERSE_FEATURE_MAP[y_axis_korean]

# 축 이외 속성들 중앙값 고정 안내 작성
fixed_features_str = []
fixed_values_dict = {}

for f in selected_features:
    med_val = X_test_raw[f].median()
    fixed_values_dict[f] = med_val
    if f not in [x_feat, y_feat]:
        fixed_features_str.append(f"{FEATURE_MAP[f]}: {med_val:.2f}")

if fixed_features_str:
    st.write(f"📌 **선택한 두 축 이외의 속성 고정값 (테스트 데이터 중앙값):** {', '.join(fixed_features_str)}")
else:
    st.write("📌 선택한 속성 2개가 가로축과 세로축에 모두 할당되었습니다.")

# 격자 그리드 생성
x_min, x_max = X_test_raw[x_feat].min(), X_test_raw[x_feat].max()
y_min, y_max = X_test_raw[y_feat].min(), X_test_raw[y_feat].max()

x_padding = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
y_padding = (y_max - y_min) * 0.05 if y_max != y_min else 1.0

grid_x = np.linspace(x_min - x_padding, x_max + x_padding, 100)
grid_y = np.linspace(y_min - y_padding, y_max + y_padding, 100)
xx, yy = np.meshgrid(grid_x, grid_y)

grid_df = pd.DataFrame(index=range(xx.size))
for f in selected_features:
    if f == x_feat:
        grid_df[f] = xx.ravel()
    elif f == y_feat:
        grid_df[f] = yy.ravel()
    else:
        grid_df[f] = fixed_values_dict[f]

# 의사결정트리 그리드 예측
dt_pred_grid = dt_clf.predict(grid_df[selected_features]).reshape(xx.shape)

# 로지스틱 회귀 그리드 확률 예측
grid_scaled = scaler.transform(grid_df[selected_features])
lr_prob_grid = log_reg.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

# Plotly 바탕 산점도 생성
fig = go.Figure()

# (1) 의사결정트리 분할 영역 (옅은 배경색)
fig.add_trace(go.Contour(
    x=grid_x, y=grid_y, z=dt_pred_grid,
    showscale=False,
    colorscale=[[0, "rgba(173, 216, 230, 0.25)"], [1, "rgba(255, 182, 193, 0.25)"]],
    hoverinfo="skip",
    name="의사결정트리 영역"
))

# (2) 로지스틱 회귀 Decision Boundary Line (P=0.5)
has_contour_line = False
if (lr_prob_grid.min() <= 0.5 <= lr_prob_grid.max()):
    has_contour_line = True
    fig.add_trace(go.Contour(
        x=grid_x, y=grid_y, z=lr_prob_grid,
        contours_start=0.5, contours_end=0.5, contours_coloring="lines",
        line_width=3, line_color="black", line_dash="dash",
        showscale=False, name="로지스틱 경계선(0.5)"
    ))

# (3) 테스트 데이터 점 찍기
test_plot_df = X_test_raw.copy()
test_plot_df["stroke"] = y_test.values
test_plot_df["stroke_label"] = test_plot_df["stroke"].map({0: "정상(0)", 1: "뇌졸중(1)"})

for label, color in zip(["정상(0)", "뇌졸중(1)"], ["blue", "red"]):
    sub = test_plot_df[test_plot_df["stroke_label"] == label]
    fig.add_trace(go.Scatter(
        x=sub[x_feat], y=sub[y_feat],
        mode="markers",
        name=label,
        marker=dict(color=color, opacity=0.7, size=7)
    ))

fig.update_layout(
    title=f"로지스틱 회귀 경계선 & 의사결정트리 분할 영역 ({x_axis_korean} vs {y_axis_korean})",
    xaxis_title=x_axis_korean,
    yaxis_title=y_axis_korean,
    legend_title="실제 뇌졸중 여부"
)

st.plotly_chart(fig, use_container_width=True)

if not has_contour_line:
    st.info("ℹ️ 현재 선택된 두 축의 범위 안에는 로지스틱 회귀 모델의 0.5 경계선이 포함되지 않아 선이 화면 밖에 위치합니다.")

st.markdown("---")

# 8. 의사결정트리 질문 구조 시각화 (Graphviz DOT)
st.subheader("🌳 의사결정트리 구조 (질문 가지)")

tree_ = dt_clf.tree_
feature_names = selected_features

nodes_info = []
leaf_nodes = []
asked_features = set()

def build_dot(tree, feature_names):
    dot_lines = [
        'digraph Tree {',
        'node [shape=box, style="filled, rounded", color="black", fontname="NanumGothic, Malgun Gothic, sans-serif"] ;',
        'edge [fontname="NanumGothic, Malgun Gothic, sans-serif"] ;'
    ]
    
    def recurse(node_id):
        samples = tree.n_node_samples[node_id]
        values = tree.value[node_id][0]
        stroke_pos = int(values[1]) if len(values) > 1 else 0
        ratio = (stroke_pos / samples * 100) if samples > 0 else 0.0
        
        is_leaf = (tree.children_left[node_id] == tree.children_right[node_id])
        
        if is_leaf:
            pred_cls = np.argmax(values)
            fillcolor = "#FFD1D1" if pred_cls == 1 else "#D1E8FF"
            label_text = f"사람 수: {samples}명\\n뇌졸중: {stroke_pos}명 ({ratio:.1f}%)\\n👉 답: {'뇌졸중(1)' if pred_cls == 1 else '아님(0)'}"
            dot_lines.append(f'{node_id} [label="{label_text}", fillcolor="{fillcolor}"] ;')
            leaf_nodes.append((node_id, pred_cls))
        else:
            feat_idx = tree.feature[node_id]
            feat_name = feature_names[feat_idx]
            feat_korean = FEATURE_MAP[feat_name]
            asked_features.add(feat_korean)
            threshold = tree.threshold[node_id]
            
            label_text = f"{feat_korean} <= {threshold:.2f}?\\n사람 수: {samples}명\\n뇌졸중: {stroke_pos}명 ({ratio:.1f}%)"
            dot_lines.append(f'{node_id} [label="{label_text}", fillcolor="#FFFFFF"] ;')
            
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            dot_lines.append(f'{node_id} -> {left_child} [label="예"] ;')
            dot_lines.append(f'{node_id} -> {right_child} [label="아니요"] ;')
            
            recurse(left_child)
            recurse(right_child)
            
    recurse(0)
    dot_lines.append('}')
    return "\n".join(dot_lines)

dot_graph = build_dot(tree_, feature_names)
st.graphviz_chart(dot_graph)

# 가지 요약 요약 정보 출력
total_leaves = len(leaf_nodes)
no_stroke_leaves = sum(1 for _, pred in leaf_nodes if pred == 0)
asked_features_str = ", ".join(list(asked_features)) if asked_features else "없음"

st.write(f"1. 답을 내는 마디는 모두 **{total_leaves}칸**이고, 그중 **{no_stroke_leaves}칸**이 '아님(0)'이라고 답합니다.")
st.write(f"2. 선택한 속성 가운데 이 나무가 실제로 물어본 속성: **{asked_features_str}**")
