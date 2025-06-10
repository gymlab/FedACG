import torch
import torch.nn as nn
import math
# from typing import Literal
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
from scipy.stats import shapiro, skew, kurtosis, norm, entropy, wasserstein_distance
from torch.distributions.normal import Normal
from scipy.stats import norm
'''
# 전체 누적확률값
cdf_value = 0.1

# 해당 누적확률에 대응하는 z값 계산
z_value = norm.ppf(cdf_value)
print(f"CDF = {cdf_value} 일 때, z값 = {z_value:.3f}")


z_value = 1.32    # 0.0625 ~ 0.9394

# 해당 z값에 대응하는 누적확률(CDF) 계산
cdf_value = norm.cdf(z_value)
print(f"z = {z_value} 일 때, CDF = {cdf_value:.4f}")
'''
num_buckets = 2 ** 6 - 2 * 15    # 64개의 버킷에서 6개를 제외한 58개의 버킷 사용
cdf_left = norm.cdf(-1.32)   # ≈ 0.0934
cdf_right = norm.cdf(1.32)   # ≈ 0.9066
quantile_edges = np.linspace(cdf_left, cdf_right, num_buckets)
print(quantile_edges)
print(len(quantile_edges))

q_values = np.round(norm.ppf(quantile_edges), 2)
print(torch.tensor(q_values))

quantile_centers = 0.5 * (quantile_edges[:-1] + quantile_edges[1:])
edges = norm.ppf(quantile_centers)
print(edges)

