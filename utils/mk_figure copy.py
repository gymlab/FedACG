import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --------------------------------------------------------
# 1. 가우시안 분포 데이터 준비
# --------------------------------------------------------
np.random.seed(0)
N = 2000                 # 데이터 개수
mu, sigma = 0, 1         # 정규분포 평균, 표준편차
data = np.random.normal(mu, sigma, N)

# 시각화를 위해 x 범위 설정(-4 ~ 4)
x = np.linspace(-4, 4, 200)
pdf = norm.pdf(x, mu, sigma)

# --------------------------------------------------------
# 2. Uniform Quantization (균일 양자화)
# --------------------------------------------------------
num_levels_uniform = 8   # 양자화 레벨 개수
min_val, max_val = data.min(), data.max()

# 데이터 범위를 균등 분할하기 위한 step 계산
step = (max_val - min_val) / num_levels_uniform

# 각 구간의 중앙값으로 매핑해 주기 위해, 
# 구간별 대표 값을 만들기
uniform_levels = np.linspace(min_val + step/2, max_val - step/2, num_levels_uniform)

def uniform_quantize(x):
    # 어떤 값 x가 들어오면, uniform_levels 중 가장 가까운 레벨로 매핑
    idx = np.floor((x - min_val)/step)
    # 범위를 벗어나지 않도록 클리핑
    idx = np.clip(idx, 0, num_levels_uniform - 1).astype(int)
    return uniform_levels[idx]

quantized_uniform = uniform_quantize(data)

# --------------------------------------------------------
# 3. Non-uniform Quantization (비균일 양자화)
# --------------------------------------------------------
num_levels_nonuniform = 8

# 비균일 양자화에서는 가우시안 분포의 CDF를 이용해
# 각 구간별로 확률질량이 동일하도록 boundary 설정
sorted_data = np.sort(data)
# N+1개 지점 중 (균등하게 선택)으로 레벨 구분 -> 분위수(quantiles)
boundaries = [np.percentile(sorted_data, 100*i/num_levels_nonuniform)
              for i in range(num_levels_nonuniform+1)]

# 각 구간마다 중앙값(혹은 평균)을 대표값으로 사용
nonuniform_levels = []
for i in range(num_levels_nonuniform):
    lower = boundaries[i]
    upper = boundaries[i+1]
    # 해당 구간의 중앙값(또는 평균)
    mid_val = 0.5 * (lower + upper)
    nonuniform_levels.append(mid_val)

def nonuniform_quantize(x):
    # 데이터 x가 어느 구간에 속하는지 확인 후 해당 구간 대표값으로 매핑
    # boundaries[i] ~ boundaries[i+1] 에 속하면 nonuniform_levels[i]
    idx = np.searchsorted(boundaries, x) - 1
    idx = np.clip(idx, 0, num_levels_nonuniform-1)
    return np.array([nonuniform_levels[i] for i in idx])

quantized_nonuniform = nonuniform_quantize(data)

# --------------------------------------------------------
# 4. 결과 시각화
# --------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# (a) 원본 가우시안 분포
axs[0].hist(data, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
axs[0].plot(x, pdf, 'r--', label='PDF')
axs[0].set_title('Original Gaussian')
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Density')
axs[0].legend()

# (b) Uniform Quantization
axs[1].scatter(data, quantized_uniform, alpha=0.3, s=10, color='green')
axs[1].set_title('Uniform Quantization')
axs[1].set_xlabel('Original Value')
axs[1].set_ylabel('Quantized Value')

# 레벨 선을 시각적으로 표시
for level in uniform_levels:
    axs[1].axhline(y=level, color='gray', linestyle='--', linewidth=0.5)

# (c) Non-uniform Quantization
axs[2].scatter(data, quantized_nonuniform, alpha=0.3, s=10, color='orange')
axs[2].set_title('Non-uniform Quantization')
axs[2].set_xlabel('Original Value')
axs[2].set_ylabel('Quantized Value')

# 레벨 선(대표값)을 시각적으로 표시
for level in nonuniform_levels:
    axs[2].axhline(y=level, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
