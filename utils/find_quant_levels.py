from itertools import combinations
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

def generate_alpha_and_q(M, step=0.001, max_value=3):
    """
    Generate all possible q values and corresponding alpha combinations where:
    - 0 <= alpha_1 < alpha_2 < ... < alpha_M <= max_value
    - Each alpha_i is a multiple of `step`.
    - q = sum(alpha_k * b_k) with b_k in {-1, 1} for all combinations of b_k.
    - Only positive q values are considered, as q is symmetric.

    Parameters:
        M (int): Number of alpha variables.
        step (float): Increment step for alpha values.
        max_value (float): Maximum value of alpha.

    Returns:
        None
    """
    min_val = 1000000.
    selected_alpha = -1
    cache = {}
    
    # Generate the range of values for alpha
    values = np.arange(step, max_value + step, step)

    # Generate combinations without replacement (to ensure no duplicates and order)
    alpha_combinations = combinations(values, M)

    total_iters = math.comb(len(values), M)
    print(f"Total iters: {total_iters}")
    # Process each combination
    for k, alpha in enumerate(alpha_combinations):
        if sum(alpha) > 6:
            continue
        
        # Generate all combinations of b_k in {-1, 1} for 2^(M-1) terms
        b_combinations = np.array(np.meshgrid(*[[-1, 1]] * len(alpha))).T.reshape(-1, len(alpha))

        # Compute q for each b_combination
        q_values = np.array([np.sum(np.array(alpha) * b) for b in b_combinations])

        # Filter unique positive q values
        unique_positive_q = np.array(sorted(set(q_values[q_values > 0])))
        # print(f"{unique_positive_q}, {alpha}")
        
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
            total_error = 2 * err(unique_positive_q[0], 0, np.inf, cache)

        if min_val > total_error:
            min_val = total_error
            selected_alpha = alpha
            # print(f"min_error: {min_val}, alpha: {alpha}")
        if k % (total_iters//1000) == 0:
            print(f"{100*k/total_iters:.2f}% Brute-force done | Alpha: {selected_alpha} | Min Err.: {min_val}")

# Example usage
M = 1 # Number of alpha variables
step = 0.0001  # Step size
max_value = 4 # Maximum value for alpha

generate_alpha_and_q(M, step, max_value)
