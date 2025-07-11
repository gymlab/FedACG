import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")  # seaborn 스타일 설정

# 샘플용 x좌표 (0부터 8까지)
x = np.linspace(0, 8, 300)

###################################
# (a) Uniform Quantization (3비트)
###################################
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
# 요청해 주신 레벨 및 구간
def nonuniform_quantization_3bit(x):
    q = np.zeros_like(x)
    q[(x >= 0) & (x < 1)]    = 0
    q[(x >= 1) & (x < 2)]    = 2
    q[(x >= 2) & (x < 3)]    = 3
    q[(x >= 3) & (x < 4)]    = 3.5
    q[(x >= 4) & (x < 5.5)]  = 4
    q[(x >= 5.5) & (x < 6)]  = 4.5
    q[(x >= 6) & (x < 6.5)]  = 5r
    q[(x >= 6.5) & (x <= 8)] = 8
    return q

# 실제 Q(x) 계산
y_uniform    = uniform_quantization_3bit(x)
y_nonuniform = nonuniform_quantization_3bit(x)

# -----------------------------
# Figure & Subplots(2x1) 생성
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
plt.subplots_adjust(hspace=0.25)  # 위/아래 플롯 간격

# -----------------------------
# (a) 3-bit Uniform Quantization
# -----------------------------
ax1 = axes[0]
ax1.step(x, y_uniform, where='post', color='C0', linewidth=2)
ax1.set_title('(a) 3-bit Uniform Quantization', fontsize=12, fontweight='bold')
ax1.set_ylabel('Quantized Value Q(x)')
ax1.set_ylim(0, 8.5)
ax1.set_yticks(range(0, 9))

# 구간 경계 표시: x=1~7
boundaries_uni = [1, 2, 3, 4, 5, 6, 7]
for b in boundaries_uni:
    # 경계 직전/직후 레벨 (ex. x=3이면 2.5 -> 3.5)
    y_start = b - 0.5
    y_end   = b + 0.5
    step_size = (y_end - y_start)
    
    # 화살표 주석 (annotate)를 사용해 표시
    ax1.annotate(
        f"Step\nsize={int(step_size)}",       # 표시할 텍스트
        xy=(b, (y_start+y_end)/2),            # 화살표가 가리키는 좌표
        xytext=(b+0.2, (y_start+y_end)/2),    # 텍스트 위치
        va='center',
        arrowprops=dict(arrowstyle="->", color='gray', lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

# -----------------------------
# (b) 3-bit Non-uniform Quantization
# -----------------------------
ax2 = axes[1]
ax2.step(x, y_nonuniform, where='post', color='C1', linewidth=2)
ax2.set_title('(b) 3-bit Non-uniform Quantization', fontsize=12, fontweight='bold')
ax2.set_xlabel('Input x')
ax2.set_ylabel('Quantized Value Q(x)')
ax2.set_ylim(0, 8.5)
ax2.set_xticks(range(0, 9))
ax2.set_yticks(range(0, 9))

# 비균등 구간 경계 & 레벨 변화
boundaries_nonuni = [1, 2, 3, 4, 5.5, 6, 6.5, 8]
q_starts = [0, 2, 3, 3.5, 4, 4.5, 5]
q_ends   = [2, 3, 3.5, 4, 4.5, 5, 8]

for i, b in enumerate(boundaries_nonuni[:-1]):
    y_start = q_starts[i]
    y_end   = q_ends[i]
    step_size = y_end - y_start
    # annotate로 표시
    ax2.annotate(
        f"Step\nsize={step_size}", 
        xy=(b, (y_start+y_end)/2),
        xytext=(b+0.2, (y_start+y_end)/2),
        va='center',
        arrowprops=dict(arrowstyle="->", color='gray', lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

plt.show()
