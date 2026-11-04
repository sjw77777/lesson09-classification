import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 1. 페이지 설정
st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실",
    page_icon="⚡",
    layout="wide"
)

# 2. 데이터 로드
@st.cache_data
def load_stroke_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

raw_df = load_stroke_data()

st.title("⚡ 데이터 불균형 해결 및 모델 성능 높이기")
st.markdown("---")

# 속성 우리말 이름 매핑 사전
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

if len(selected_korean_features) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해 주세요. 속성이 2개 미만이면 모델 학습 및 시각화를 진행할 수 없습니다.")
    st.stop()

selected_features = [REVERSE_FEATURE_MAP[k] for k in selected_korean_features]

# 4. 데이터 분할 및 전처리 (이전 페이지들과 동일한 방식)
df_sorted = raw_df.sort_index().copy()

indices = np.arange(len(df_sorted))
test_mask = (indices % 10) < 3

train_df = df_sorted.iloc[~test_mask].copy()
test_df = df_sorted.iloc[test_mask].copy()

# bmi 결측치 훈련용 중앙값으로 채우기
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)

X_train_raw = train_df[selected_features]
y_train = train_df["stroke"]
X_test_raw = test_df[selected_features]
y_test = test_df["stroke"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 5. 성능 평가 계산 함수
def compute_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (tp + fp) == 0:
        prec_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
    else:
        prec = tp / (tp + fp)
        prec_str = f"{prec*100:.2f}%"
        if (prec + rec) == 0:
            f1_str = "계산할 수 없음"
        else:
            f1 = 2 * (prec * rec) / (prec + rec)
            f1_str = f"{f1:.4f}"
            
    acc_str = f"{acc*100:.2f}%"
    rec_str = f"{rec*100:.2f}%"
    
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": acc_str, "Recall": rec_str, "Precision": prec_str, "F1": f1_str
    }

# 6. 모델 학습 (가중치 맞추기 전 vs 맞춘 후)
# (1) 로지스틱 회귀
lr_before = LogisticRegression(random_state=42)
lr_before.fit(X_train_scaled, y_train)
lr_pred_before = lr_before.predict(X_test_scaled)

lr_after = LogisticRegression(class_weight="balanced", random_state=42)
lr_after.fit(X_train_scaled, y_train)
lr_pred_after = lr_after.predict(X_test_scaled)

# (2) 의사결정트리
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train_raw, y_train)
dt_pred_before = dt_before.predict(X_test_raw)

dt_after = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42)
dt_after.fit(X_train_raw, y_train)
dt_pred_after = dt_after.predict(X_test_raw)

# 평가 측정
m_lr_bef = compute_metrics(y_test.values, lr_pred_before)
m_lr_aft = compute_metrics(y_test.values, lr_pred_after)

m_dt_bef = compute_metrics(y_test.values, dt_pred_before)
m_dt_aft = compute_metrics(y_test.values, dt_pred_after)

# 7. 모델별 가중치 전/후 비교 표 출력
st.subheader("📊 가중치 조절 전·후 성능 비교 표")

def make_comparison_table(m1, m2):
    return pd.DataFrame([
        {"지표 / 항목": "진짜 양성 (TP - 찾아낸 환자)", "조정 전": f"{m1['TP']} 명", "조정 후(balanced)": f"{m2['TP']} 명"},
        {"지표 / 항목": "가짜 양성 (FP - 헛짚은 사람)", "조정 전": f"{m1['FP']} 명", "조정 후(balanced)": f"{m2['FP']} 명"},
        {"지표 / 항목": "가짜 음성 (FN - 놓친 환자)", "조정 전": f"{m1['FN']} 명", "조정 후(balanced)": f"{m2['FN']} 명"},
        {"지표 / 항목": "진짜 음성 (TN - 맞춘 정상)", "조정 전": f"{m1['TN']} 명", "조정 후(balanced)": f"{m2['TN']} 명"},
        {"지표 / 항목": "정확도 (Accuracy)", "조정 전": m1["Accuracy"], "조정 후(balanced)": m2["Accuracy"]},
        {"지표 / 항목": "재현율 (Recall)", "조정 전": m1["Recall"], "조정 후(balanced)": m2["Recall"]},
        {"지표 / 항목": "정밀도 (Precision)", "조정 전": m1["Precision"], "조정 후(balanced)": m2["Precision"]},
        {"지표 / 항목": "F1 점수", "조정 전": m1["F1"], "조정 후(balanced)": m2["F1"]}
    ])

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.markdown("#### 1. 로지스틱 회귀(확률로 답하는 모델)")
    st.table(make_comparison_table(m_lr_bef, m_lr_aft))
    st.write(f"👉 **변화 요약:** 찾아낸 환자 수(**{m_lr_bef['TP']}명 ➔ {m_lr_aft['TP']}명**), 놓친 환자 수(**{m_lr_bef['FN']}명 ➔ {m_lr_aft['FN']}명**), 헛짚은 사람 수(**{m_lr_bef['FP']}명 ➔ {m_lr_aft['FP']}명**)")

with col_t2:
    st.markdown("#### 2. 의사결정트리(질문으로 답하는 모델)")
    st.table(make_comparison_table(m_dt_bef, m_dt_aft))
    st.write(f"👉 **변화 요약:** 찾아낸 환자 수(**{m_dt_bef['TP']}명 ➔ {m_dt_aft['TP']}명**), 놓친 환자 수(**{m_dt_bef['FN']}명 ➔ {m_dt_aft['FN']}명**), 헛짚은 사람 수(**{m_dt_bef['FP']}명 ➔ {m_dt_aft['FP']}명**)")

st.markdown("---")

# 8. 2차원 산점도 및 결정 경계 비교 시각화 (Plotly)
st.subheader("📐 2차원 가중치 조정 전·후 경계선 비교")

col_s1, col_s2 = st.columns(2)
with col_s1:
    x_korean = st.selectbox("가로축(X축) 속성 선택", selected_korean_features, index=0)
with col_s2:
    y_default_idx = 1 if len(selected_korean_features) > 1 else 0
    y_korean = st.selectbox("세로축(Y축) 속성 선택", selected_korean_features, index=y_default_idx)

x_f = REVERSE_FEATURE_MAP[x_korean]
y_f = REVERSE_FEATURE_MAP[y_korean]

# 다른 속성 중앙값 고정
fixed_dict = {}
for f in selected_features:
    fixed_dict[f] = X_test_raw[f].median()

# 격자 설정
x_min, x_max = X_test_raw[x_f].min(), X_test_raw[x_f].max()
y_min, y_max = X_test_raw[y_f].min(), X_test_raw[y_f].max()

x_pad = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0

grid_x = np.linspace(x_min - x_pad, x_max + x_pad, 100)
grid_y = np.linspace(y_min - y_pad, y_max + y_pad, 100)
xx, yy = np.meshgrid(grid_x, grid_y)

grid_df = pd.DataFrame(index=range(xx.size))
for f in selected_features:
    if f == x_f:
        grid_df[f] = xx.ravel()
    elif f == y_f:
        grid_df[f] = yy.ravel()
    else:
        grid_df[f] = fixed_dict[f]

grid_scaled = scaler.transform(grid_df[selected_features])
prob_bef = lr_before.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
prob_aft = lr_after.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

fig2d = go.Figure()

# 실제 테스트 데이터 점 찍기
test_plot_df = X_test_raw.copy()
test_plot_df["stroke"] = y_test.values
test_plot_df["stroke_label"] = test_plot_df["stroke"].map({0: "정상(0)", 1: "뇌졸중(1)"})

for label, color in zip(["정상(0)", "뇌졸중(1)"], ["#1f77b4", "#d62728"]):
    sub = test_plot_df[test_plot_df["stroke_label"] == label]
    fig2d.add_trace(go.Scatter(
        x=sub[x_f], y=sub[y_f],
        mode="markers",
        name=label,
        marker=dict(color=color, opacity=0.6, size=7)
    ))

# 0.5 경계선 추가 (조정 전 vs 조정 후)
has_bef_line = (prob_bef.min() <= 0.5 <= prob_bef.max())
has_aft_line = (prob_aft.min() <= 0.5 <= prob_aft.max())

if has_bef_line:
    fig2d.add_trace(go.Contour(
        x=grid_x, y=grid_y, z=prob_bef,
        contours_start=0.5, contours_end=0.5, contours_coloring="lines",
        line_width=3, line_color="orange", line_dash="dash",
        showscale=False, name="가중치 조정 전 경계선(0.5)"
    ))

if has_aft_line:
    fig2d.add_trace(go.Contour(
        x=grid_x, y=grid_y, z=prob_aft,
        contours_start=0.5, contours_end=0.5, contours_coloring="lines",
        line_width=3, line_color="green",
        showscale=False, name="가중치 조정 후 경계선(0.5)"
    ))

fig2d.update_layout(
    title=f"로지스틱 회귀 2D 경계선 비교 ({x_korean} vs {y_korean})",
    xaxis_title=x_korean,
    yaxis_title=y_korean,
    legend_title="구분"
)

st.plotly_chart(fig2d, use_container_width=True)

# 경계선 범위 이탈 안내
out_messages = []
if not has_bef_line:
    out_messages.append("가중치 조정 전의 0.5 경계선이 선택한 두 축의 표시 범위 밖에 있습니다.")
if not has_aft_line:
    out_messages.append("가중치 조정 후의 0.5 경계선이 선택한 두 축의 표시 범위 밖에 있습니다.")

if out_messages:
    for msg in out_messages:
        st.info(f"ℹ️ {msg}")

st.markdown("---")

# 9. 3차원 산점도 및 결정 평면 비교 시각화 (속성 3개 이상 선택 시)
if len(selected_features) >= 3:
    st.subheader("🧊 3차원 공간 산점도 및 가중치 전·후 평면 시각화")
    st.write("마우스 클릭 및 드래그로 3차원 공간을 자유롭게 회전시켜 볼 수 있습니다.")

    col_3d1, col_3d2, col_3d3 = st.columns(3)
    with col_3d1:
        x3_korean = st.selectbox("3D X축 속성", selected_korean_features, index=0)
    with col_3d2:
        y3_korean = st.selectbox("3D Y축 속성", selected_korean_features, index=1)
    with col_3d3:
        z3_korean = st.selectbox("3D Z축 속성", selected_korean_features, index=2)

    x3_f = REVERSE_FEATURE_MAP[x3_korean]
    y3_f = REVERSE_FEATURE_MAP[y3_korean]
    z3_f = REVERSE_FEATURE_MAP[z3_korean]

    if len({x3_f, y3_f, z3_f}) < 3:
        st.info("💡 3D 그래프 관찰을 위해 서로 다른 3개의 속성을 선택해 주세요.")
    else:
        # 3D 그리드 생성
        g3_x = np.linspace(X_test_raw[x3_f].min(), X_test_raw[x3_f].max(), 25)
        g3_y = np.linspace(X_test_raw[y3_f].min(), X_test_raw[y3_f].max(), 25)
        xx3, yy3 = np.meshgrid(g3_x, g3_y)

        # 회귀 평면 방정식 z값 계산 (P=0.5 ➔ w*X + b = 0)
        # 로지스틱 회귀: z_scaled = w_x*x_s + w_y*y_s + w_z*z_s + sum(other) + intercept = 0
        w_bef = lr_before.coef_[0]
        b_bef = lr_before.intercept_[0]
        
        w_aft = lr_after.coef_[0]
        b_aft = lr_after.intercept_[0]

        idx_x = selected_features.index(x3_f)
        idx_y = selected_features.index(y3_f)
        idx_z = selected_features.index(z3_f)

        means = scaler.mean_
        scales = scaler.scale_

        def get_z_plane(w, b, xx_grid, yy_grid):
            # z_scaled_coef * (z - mean_z)/scale_z = - intercept - sum(w_i * (x_i - mean_i)/scale_i)
            # => z = mean_z + (scale_z / w_z) * (- b - sum_other)
            other_sum = b
            for idx, f in enumerate(selected_features):
                if idx not in [idx_x, idx_y, idx_z]:
                    val = fixed_dict[f]
                    val_s = (val - means[idx]) / scales[idx]
                    other_sum += w[idx] * val_s

            x_s = (xx_grid - means[idx_x]) / scales[idx_x]
            y_s = (yy_grid - means[idx_y]) / scales[idx_y]

            if abs(w[idx_z]) < 1e-6:
                return np.full_like(xx_grid, np.nan)
            
            z_s = (- other_sum - w[idx_x] * x_s - w[idx_y] * y_s) / w[idx_z]
            z_unscaled = z_s * scales[idx_z] + means[idx_z]
            return z_unscaled

        zz3_bef = get_z_plane(w_bef, b_bef, xx3, yy3)
        zz3_aft = get_z_plane(w_aft, b_aft, xx3, yy3)

        fig3d = go.Figure()

        # 3D 점 찍기
        for label, color in zip(["정상(0)", "뇌졸중(1)"], ["blue", "red"]):
            sub = test_plot_df[test_plot_df["stroke_label"] == label]
            fig3d.add_trace(go.Scatter3d(
                x=sub[x3_f], y=sub[y3_f], z=sub[z3_f],
                mode="markers",
                name=label,
                marker=dict(color=color, size=4, opacity=0.7)
            ))

        # 조정 전 평면 (주황색)
        if not np.all(np.isnan(zz3_bef)):
            fig3d.add_trace(go.Surface(
                x=xx3, y=yy3, z=zz3_bef,
                colorscale=[[0, "orange"], [1, "orange"]],
                opacity=0.4,
                showscale=False,
                name="가중치 조정 전 평면(0.5)"
            ))

        # 조정 후 평면 (초록색)
        if not np.all(np.isnan(zz3_aft)):
            fig3d.add_trace(go.Surface(
                x=xx3, y=yy3, z=zz3_aft,
                colorscale=[[0, "green"], [1, "green"]],
                opacity=0.4,
                showscale=False,
                name="가중치 조정 후 평면(0.5)"
            ))

        fig3d.update_layout(
            title="3차원 결정 평면 겹쳐보기",
            scene=dict(
                xaxis_title=x3_korean,
                yaxis_title=y3_korean,
                zaxis_title=z3_korean
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        st.plotly_chart(fig3d, use_container_width=True)
else:
    st.info("💡 속성을 3개 이상 선택하시면 3차원 산점도 및 결정 평면 시각화 기능이 추가로 제공됩니다.")
