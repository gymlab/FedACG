import numpy as np
import matplotlib.pyplot as plt

# 샘플용 x좌표 (0부터 8까지 300개 점)
x = np.linspace(0, 8, 300)

###################################
# (a) Uniform Quantization (3비트)
###################################
# - 전체 구간 [0,8]을 8개 동일 구간으로 나누어
# - 각 구간은 폭이 1, 중심 레벨은 0.5, 1.5, ..., 7.5
def uniform_quantization_3bit(x):
    q = np.zeros_like(x)
    q[(x >= 0) & (x < 1)] = 0.5
    q[(x >= 1) & (x < 2)] = 1.5
    q[(x >= 2) & (x < 3)] = 2.5
    q[(x >= 3) & (x < 4)] = 3.5
    q[(x >= 4) & (x < 5)] = 4.5
    q[(x >= 5) & (x < 6)] = 5.5
    q[(x >= 6) & (x < 7)] = 6.5
    q[(x >= 7) & (x <= 8)] = 7.5
    return q

###################################
# (b) Non-uniform Quantization (3비트)
###################################
# - 주어진 숫자에 맞춰 구간을 비균등하게 나누고 레벨 할당
#   0<=x<1   -> Q=0
#   1<=x<2   -> Q=2
#   2<=x<3   -> Q=3
#   3<=x<4   -> Q=3.5
#   4<=x<5.5 -> Q=4
#   5.5<=x<6 -> Q=4.5
#   6<=x<6.5 -> Q=5
#   6.5<=x<=8-> Q=8
def nonuniform_quantization_3bit(x):
    q = np.zeros_like(x)
    q[(x >= 0) & (x < 1)]    = 0
    q[(x >= 1) & (x < 2)]    = 2
    q[(x >= 2) & (x < 3)]    = 3
    q[(x >= 3) & (x < 4)]    = 3.5
    q[(x >= 4) & (x < 5.5)]  = 4
    q[(x >= 5.5) & (x < 6)]  = 4.5
    q[(x >= 6) & (x < 6.5)]  = 5
    q[(x >= 6.5) & (x <= 8)] = 8
    return q

# 실제 Q(x) 계산
y_uniform    = uniform_quantization_3bit(x)
y_nonuniform = nonuniform_quantization_3bit(x)

# 플롯 생성
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axes[0].step(x, y_uniform, where='post', color='green')
# axes[0].set_title('(a) 3-bit Uniform Quantization')
axes[0].set_xlabel('x')
axes[0].set_ylabel('quantized value')
axes[0].set_xticks(range(0,9))
axes[0].set_yticks(range(0,9))
axes[0].grid(True, alpha=0.3)

for boundary in range(1, 8):
    # 경계 x=boundary 에서 Q는 (boundary - 0.5) -> (boundary + 0.5)
    axes[0].plot([boundary, boundary],
                 [boundary - 0.5, boundary + 0.5],
                 '--k')
    # axes[0].text(boundary + 0.05, boundary,
    #              'Interval=1',
    #              va='center')

axes[1].step(x, y_nonuniform, where='post', color='green')
# axes[1].set_title('(b) 3-bit Non-uniform Quantization')
axes[1].set_xlabel('x')
axes[1].set_xticks(range(0,9))
axes[1].set_yticks(range(0,9))
axes[1].grid(True, alpha=0.3)

# Non-uniform 구간 경계와 스텝 크기
boundaries = [1, 2, 3, 4, 5.5, 6, 6.5, 8]
q_starts = [0, 2, 3, 3.5, 4, 4.5, 5]   # 경계 직전 구간의 레벨
q_ends   = [2, 3, 3.5, 4, 4.5, 5, 8]   # 경계 직후 구간의 레벨

for i, b in enumerate(boundaries[:-1]):  
    x0 = b
    y1, y2 = q_starts[i], q_ends[i]
    step_size = y2 - y1
    axes[1].plot([x0, x0], [y1, y2], '--k')
    # axes[1].text(x0 + 0.05, (y1 + y2) / 2,
    #              f'Interval={step_size}',
    #              va='center')

plt.tight_layout()
plt.show()
