import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 8, 200)

def uniform_quantization(x):
    q = np.zeros_like(x)
    q[(x >= 0) & (x < 2)] = 1
    q[(x >= 2) & (x < 4)] = 3
    q[(x >= 4) & (x < 6)] = 5
    q[(x >= 6) & (x <= 8)] = 7
    return q


def nonuniform_quantization(x):
    q = np.zeros_like(x)
    q[(x >= 0) & (x < 1)] = 1
    q[(x >= 1) & (x < 3)] = 3
    q[(x >= 3) & (x < 5)] = 6
    q[(x >= 5) & (x <= 8)] = 7
    return q

y_uniform = uniform_quantization(x)
y_nonuniform = nonuniform_quantization(x)


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axes[0].step(x, y_uniform, where='post', color='green')
axes[0].set_title('(a) Uniform Quantization')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Q(x)')
axes[0].set_xticks(range(0,9))
axes[0].set_yticks(range(0,9))
axes[0].grid(True, alpha=0.3)

axes[0].plot([2,2], [1,3], '--k')
axes[0].text(2.1, 2, 'Interval=2', va='center')

axes[0].plot([4,4], [3,5], '--k')
axes[0].text(4.1, 4, 'Interval=2', va='center')

axes[0].plot([6,6], [5,7], '--k')
axes[0].text(6.1, 6, 'Interval=2', va='center')


axes[1].step(x, y_nonuniform, where='post', color='green')
axes[1].set_title('(b) Non-uniform Quantization')
axes[1].set_xlabel('x')
axes[1].set_xticks(range(0,9))
axes[1].set_yticks(range(0,9))
axes[1].grid(True, alpha=0.3)

axes[1].plot([1,1], [1,3], '--k')
axes[1].text(1.1, 2, 'Interval=2', va='center')


axes[1].plot([3,3], [3,6], '--k')
axes[1].text(3.1, 4.5, 'Interval=3', va='center')

axes[1].plot([5,5], [6,7], '--k')
axes[1].text(5.1, 6.5, 'Interval=1', va='center')

plt.tight_layout()
plt.show()
