
import itertools

import numpy as np
import math
from scipy import special

def err(q, s1, s2, cache):
    key = (q, s1, s2)
    if key in cache:
        return cache[key]
    if s2 == np.inf:
        ss1 = s1 / math.sqrt(2)
        coeff = (1. / math.sqrt(2 * math.pi))
        result = -coeff * (2 * q - s1) * math.exp(-ss1 ** 2) + 0.5 * (q ** 2 + 1.) * (1. - math.erf(ss1))
    else:
        ss1 = s1 / math.sqrt(2)
        ss2 = s2 / math.sqrt(2)
        coeff = (1. / math.sqrt(2 * math.pi))
        result = coeff * (2 * q - s2) * math.exp(-ss2 ** 2) - coeff * (2 * q - s1) * math.exp(-ss1 ** 2) + \
                 0.5 * (q ** 2 + 1.) * (math.erf(ss2) - math.erf(ss1))
    cache[key] = result
    return result

def refined_search(selected_alpha, refined_step, max_value, refinement_range=0.05):
    min_val = float('inf')
    M = len(selected_alpha)
    cache = {}

    ranges = [(max(refined_step, a - refinement_range), min(max_value, a + refinement_range)) for a in selected_alpha]
    values_list = [np.arange(start, end + refined_step, refined_step) for start, end in ranges]

    # Generate all combinations of alpha values
    alpha_combinations = itertools.product(*values_list)

    for k, alpha in enumerate(alpha_combinations):
        # Generate all combinations of b_k in {-1, 1} for 2^(M-1) terms
        b_combinations = np.array(np.meshgrid(*[[-1, 1]] * len(alpha))).T.reshape(-1, len(alpha))

        # Compute q for each b_combination
        q_values = np.array([np.sum(np.array(alpha) * b) for b in b_combinations])

        # Filter unique positive q values
        unique_positive_q = np.array(sorted(set(q_values[q_values > 0])))

        q_minus = unique_positive_q[:-1] - unique_positive_q[1:]
        q_plus = unique_positive_q[:-1] + unique_positive_q[1:]
        s = q_plus * 0.5
        s1 = unique_positive_q[0] * 0.5
        q_square_diff = q_minus * q_plus
        
        if M > 2:
            total_error = np.sqrt(2./np.pi) * np.sum(q_minus * np.exp(- 0.5 * (s ** 2))) \
                        + 0.5 * np.sum(q_square_diff * special.erf(s / np.sqrt(2))) \
                        + np.sqrt(1./(2*np.pi)) * (0.5 * s1 - 2 * unique_positive_q[0]) * np.exp(- 0.5 * (s1 ** 2)) \
                        - 0.25 * (2 * unique_positive_q[0] ** 2 + 1) * special.erf(s1 / np.sqrt(2)) \
                        + (1. / (2 * np.sqrt(2 * np.pi))) * (2 * unique_positive_q[-1] - s[-1]) * np.exp(- 0.5 * (s[-1] ** 2)) \
                        - (1. / np.sqrt(2 * np.pi)) * (2 * unique_positive_q[-2] - s[-2]) * np.exp(- 0.5 * (s[-2] ** 2)) \
                        + 0.5 * ((unique_positive_q[-1] ** 2 + 1) * (1. - special.erf(s[-1] / np.sqrt(2))) + (unique_positive_q[-2] ** 2 + 1) * (1. - special.erf(s[-2] / np.sqrt(2)))) \
                        + 0.5 * (unique_positive_q[-1] ** 2 + 1)
        elif M == 2:
            total_error = 2 * err(0, 0, s1, cache)
            total_error += err(unique_positive_q[0], s1, s[0], cache)
            total_error += err(unique_positive_q[1], s[0], np.inf, cache)
            total_error += err(unique_positive_q[0], s1, np.inf, cache)
        elif M == 1:
            total_error = 2 * err(unique_positive_q[0], 0, np.inf)

        if min_val > total_error:
            min_val = total_error
            selected_alpha = alpha

        if k % 10000 == 0:
            print(f"{k} iterations | Alpha: {selected_alpha} | Min Err.: {min_val}")

    print(f"Selected | Alpha: {selected_alpha} | Min Err.: {min_val}")

# Example usage
selected_alpha =  (0.438, 1.507)
refined_step = 0.0001  # Step size for refined search
max_value = 3 # Maximum value for alpha
refinement_range = 0.002  # Range around found alpha values for refinement

refined_search(selected_alpha, refined_step, max_value, refinement_range)
