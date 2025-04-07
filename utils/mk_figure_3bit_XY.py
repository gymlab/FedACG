import numpy as np
import matplotlib.pyplot as plt

# ----- 예시: 3비트 Uniform Quantization -----
def uniform_quantization_3bit(x):
    """ x in [0,8], 8개 구간 각각 레벨: 0.5, 1.5, 2.5, ..., 7.5 """
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

# ----- 예시: 3비트 Non-uniform Quantization -----
def nonuniform_quantization_3bit(x):
    """
    0<=x<1    -> Q=0
    1<=x<2    -> Q=2
    2<=x<3    -> Q=3
    3<=x<4    -> Q=3.5
    4<=x<5.5  -> Q=4
    5.5<=x<6  -> Q=4.5
    6<=x<6.5  -> Q=5
    6.5<=x<=8 -> Q=8
    """
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

# 입력 구간 x = 0~8
x = np.linspace(0, 8, 300)
y_uni = uniform_quantization_3bit(x)
y_non = nonuniform_quantization_3bit(x)

# ------------------------------------------------------------------------
#  그림: 2x1 서브플롯 (a) Uniform, (b) Non-uniform
#       가로축 = Q(x), 세로축 = x 로 뒤집어서 그리기
# ------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=False, sharey=False)
plt.subplots_adjust(hspace=0.25)

# ------------------------
# (a) Uniform
# ------------------------
axes[0].step(y_uni, x, where='post', color='C0', lw=2)
axes[0].set_title('(a) 3-bit Uniform (Horizontal = Q(x), Vertical = x)', 
                  fontsize=11, fontweight='bold')
axes[0].set_xlabel('Quantized value Q(x)')
axes[0].set_ylabel('Input x')
axes[0].set_xlim([-0.5, 8.5])
axes[0].set_ylim([0, 8])

# ------------------------
# (b) Non-uniform
# ------------------------
axes[1].step(y_non, x, where='post', color='C1', lw=2)
axes[1].set_title('(b) 3-bit Non-uniform (Horizontal = Q(x), Vertical = x)', 
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('Quantized value Q(x)')
axes[1].set_ylabel('Input x')
axes[1].set_xlim([-0.5, 8.5])
axes[1].set_ylim([0, 8])

plt.show()
