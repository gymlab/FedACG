import numpy as np
import matplotlib.pyplot as plt

def uniform_quantization_2bit(x):
    """
    2비트 -> 4개 레벨
    구간을 균등하게 [0~2), [2~4), [4~6), [6~8]
    각각 Q(x) = 1, 3, 5, 7
    """
    q = np.zeros_like(x)
    q[(x >= 0) & (x < 2)] = 1
    q[(x >= 2) & (x < 4)] = 3
    q[(x >= 4) & (x < 6)] = 5
    q[(x >= 6) & (x <= 8)] = 7
    return q

def nonuniform_quantization_2bit(x):
    """
    2비트 -> 4개 레벨
    구간을 비균등하게 [0~1), [1~3), [3~5), [5~8)
    각각 Q(x) = 1, 3, 5, 7
    """
    q = np.zeros_like(x)
    q[(x >= 0) & (x < 1)]  = 1
    q[(x >= 1) & (x < 3)]  = 3
    q[(x >= 3) & (x < 5)]  = 5
    q[(x >= 5) & (x <= 8)] = 7
    return q

# x: 입력 값(0부터 8까지)
x = np.linspace(0, 8, 300)

y_uni = uniform_quantization_2bit(x)
y_non = nonuniform_quantization_2bit(x)

#-----------------------------------------
# 2x1 서브플롯:
#   (a) Uniform (위)
#   (b) Non-uniform (아래)
#   => "가로축=Q(x)", "세로축=x" 형태
#-----------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(6, 8))
plt.subplots_adjust(hspace=0.3)

# (a) Uniform
axes[0].step(y_uni, x, where='post', color='C0', lw=2)
axes[0].set_title('(a) 2-bit Uniform (Horizontal = Q(x), Vertical = x)', 
                  fontsize=11, fontweight='bold')
axes[0].set_xlabel('Quantized value Q(x)')
axes[0].set_ylabel('Input x')
axes[0].set_xlim(0, 8)   # Q(x)의 범위
axes[0].set_ylim(0, 8)   # x의 범위
axes[0].grid(True, alpha=0.3)

# (b) Non-uniform
axes[1].step(y_non, x, where='post', color='C1', lw=2)
axes[1].set_title('(b) 2-bit Non-uniform (Horizontal = Q(x), Vertical = x)', 
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('Quantized value Q(x)')
axes[1].set_ylabel('Input x')
axes[1].set_xlim(0, 8)
axes[1].set_ylim(0, 8)
axes[1].grid(True, alpha=0.3)

plt.show()
