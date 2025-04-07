import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# 1. 가우시안 분포 데이터 준비
# -----------------------------
np.random.seed(0)
N = 2000
mu, sigma = 0, 1  # 평균, 표준편차
data = np.random.normal(mu, sigma, N)

# 데이터의 최소·최대값
min_val, max_val = data.min(), data.max()

# 가우시안 이론적 분포(PDF) 시각화를 위한 x 범위
x_pdf = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
pdf = norm.pdf(x_pdf, mu, sigma)

# -----------------------------
# 2. 균일 양자화 (Uniform)
# -----------------------------
num_levels = 8
# 균일 분할을 위한 bin 경계 (9개 지점)
boundaries_uniform = np.linspace(min_val, max_val, num_levels + 1)

def quantize_uniform(x):
    """
    x를 boundaries_uniform에 따라
    1~8의 정수 레벨로 매핑
    """
    # 각 x가 어느 구간에 속하는지 찾는다.
    # 예: boundaries [b0, b1, b2, ... , b8] 이면
    # b0 <= x < b1 -> 레벨 1
    # b1 <= x < b2 -> 레벨 2
    # ...
    idx = np.searchsorted(boundaries_uniform, x, side='right') - 1
    # searchsorted 결과가 0 이하, 8 이상으로 넘어가지 않도록 클리핑
    idx = np.clip(idx, 0, num_levels - 1)
    # 정수 레벨은 1부터 시작
    return idx + 1

# -----------------------------
# 3. 비균일 양자화 (Non-uniform)
# -----------------------------
# 분위수(Percentile)를 사용해 전체 데이터가
# 각 구간에 골고루 분포되도록 경계를 구한다.
sorted_data = np.sort(data)
boundaries_nonuniform = [np.percentile(sorted_data, 100 * i / num_levels)
                         for i in range(num_levels + 1)]

def quantize_nonuniform(x):
    """
    x를 boundaries_nonuniform에 따라
    1~8의 정수 레벨로 매핑
    """
    idx = np.searchsorted(boundaries_nonuniform, x, side='right') - 1
    idx = np.clip(idx, 0, num_levels - 1)
    return idx + 1

# -----------------------------
# 4. 시각화
#    - (왼쪽) 균일 분할 히스토그램 + 경계
#    - (오른쪽) 비균일 분할 히스토그램 + 경계
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (왼쪽) Uniform
ax1 = axes[0]
ax1.hist(data, bins=30, density=True, alpha=0.6, edgecolor='black', color='skyblue')
ax1.plot(x_pdf, pdf, 'g--', label='PDF', linewidth=0.8)

# bin 경계 표시
for b in boundaries_uniform:
    ax1.axvline(b, color='red', linestyle='--', linewidth=1.5)

# 그래프 타이틀 / 범례
# ax1.set_title('Uniform Quantization')
ax1.set_xlabel('x')
ax1.set_ylabel('Density')
ax1.legend()

# (오른쪽) Non-uniform
ax2 = axes[1]
ax2.hist(data, bins=30, density=True, alpha=0.6, edgecolor='black', color='lightgreen')
ax2.plot(x_pdf, pdf, 'g--', label='PDF', linewidth=0.8)

# bin 경계 표시
for b in boundaries_nonuniform:
    ax2.axvline(b, color='red', linestyle='--', linewidth=1.5)

# ax2.set_title('Non-uniform Quantization')
ax2.set_xlabel('x')
ax2.set_ylabel('Density')
ax2.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# 5. 예시: 실제 양자화된 값 살펴보기
# -----------------------------
sample_data = np.array([-1.5, -0.2, 0.1, 0.5, 1.2, 2.0])
print("Sample data:", sample_data)
print("Uniform quantized:", quantize_uniform(sample_data))
print("Non-uniform quantized:", quantize_nonuniform(sample_data))
