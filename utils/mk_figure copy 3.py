import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# 1. 데이터 생성 (예: 정규분포)
# -----------------------------
np.random.seed(0)
data = np.random.normal(loc=0.0, scale=1.0, size=10000)

# 실제 데이터 범위
x_min, x_max = data.min(), data.max()

# ------------------------------------------
# 2. 균등(Uniform) quantization용 구간 경계 설정
# ------------------------------------------
b = 3  # 예: 3비트
num_levels = 2 ** b  # 8개 레벨
# 절댓값 최댓값 기준으로 범위 설정 (보통)
abs_max = max(abs(x_min), abs(x_max))
uni_edges = np.linspace(-abs_max, abs_max, num_levels + 1)  # 균등 경계


# 각 구간의 대표값(계단 높이)
uni_centers = 0.5 * (uni_edges[:-1] + uni_edges[1:])

# --------------------------------------------------
# 3. 비균등(Non-uniform) quantization용 구간 경계 설정
#    (단순히 분위수를 이용한 예시)
# --------------------------------------------------
sorted_data = np.sort(data)
percentiles = np.linspace(0, 100, num_levels + 1)
nonuni_edges = np.percentile(sorted_data, percentiles)

# 각 구간의 대표값 (중앙값)
nonuni_centers = 0.5 * (nonuni_edges[:-1] + nonuni_edges[1:])

# -------------------------------------------------
# 4. 히스토그램 + PDF + 계단함수 + 범례 추가
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 공통 x 축 범위 설정(시각적 통일감)
common_x = np.linspace(-abs_max, abs_max, 200)
pdf = norm.pdf(common_x, loc=0.0, scale=1.0)

# ===== (왼쪽) 균등 quantization =====
ax = axes[0]
# ax.set_title("Uniform Quantization (3-bit)")

# 히스토그램 (데이터 분포)
ax.hist(data, bins=40, density=True, alpha=0.5, color='skyblue', edgecolor='k', label='Histogram')

# PDF 곡선
ax.plot(common_x, pdf, 'g--', label="PDF")

# 균등 quantization 경계선
for edge in uni_edges:
    ax.axvline(edge, color='red', ls=':', alpha=0.8)

# 균등 quantization '계단함수' 시각화
for i in range(len(uni_centers)):
    x_left, x_right = uni_edges[i], uni_edges[i+1]
    step_height = norm.pdf(uni_centers[i])
    ax.plot([x_left, x_right], [step_height, step_height], 
            color='blue', linewidth=2, label='Interval' if i==0 else "")

ax.set_xlabel("x")
ax.set_ylabel("Density")

# ===== (오른쪽) 비균등 quantization =====
ax = axes[1]
# ax.set_title("Non-uniform Quantization (3-bit)")

# 히스토그램 (데이터 분포)
ax.hist(data, bins=40, density=True, alpha=0.5, color='lightgreen', edgecolor='k', label='Histogram')

# PDF 곡선
ax.plot(common_x, pdf, 'g--', label="PDF")

# 비균등 quantization 경계선
for edge in nonuni_edges:
    ax.axvline(edge, color='red', ls=':', alpha=0.8)

# 비균등 quantization '계단함수' 시각화
for i in range(len(nonuni_centers)):
    x_left, x_right = nonuni_edges[i], nonuni_edges[i+1]
    step_height = norm.pdf(nonuni_centers[i])
    ax.plot([x_left, x_right], [step_height, step_height], 
            color='green', linewidth=2, label='Interval' if i==0 else "")

# ======== 공통 범례 추가 ========
# 왼쪽 그래프에 범례 추가
ax.set_xlabel("x")
ax.set_ylabel("Density")
axes[0].legend(loc='upper right')
# 오른쪽 그래프에 범례 추가
axes[1].legend(loc='upper right')


# 추가 설명: 전체 그래프에 대한 공통 설명
# fig.text(0.5, 0.01, 
#          "Blue line: Uniform Interval | Green line: Non-uniform Interval | Red dotted line: Quantization Boundaries | Green dotted line: PDF",
#          ha='center', fontsize=12, color='black')

plt.tight_layout()
plt.show()
