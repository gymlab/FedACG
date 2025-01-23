import itertools
import numpy as np
import math
from scipy import special
from scipy.stats import norm


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


def refined_error_search(q_values, refined_step, refined_range=-1):
    """
    Perform error minimization search based on provided q_values.

    Parameters:
        q_values (np.array): Array of q_values to optimize.
        refined_step (float): Step size for refined search.
        refinement_range (float): Range around q_values for refinement.

    Returns:
        optimized_q_values (np.array): Optimized q_values that minimize the error.
        min_error (float): Minimum error value achieved.
    """
    # Ensure q_values is sorted and unique
    q_values = np.array(sorted(set(q_values)))

    min_error = float('inf')
    optimized_q_values = q_values

    # Create ranges for refinement between consecutive q_values
    
    ranges = []
    if len(q_values) == 1:
        ranges = [(refined_step, 2.5)]
    elif refined_range > 0:
        for i, q in enumerate(q_values):
            start = max(0, q - refined_range)
            end = min(q + refined_range + refined_step, 2.5)
            ranges.append((start, end))
    else: 
        for i, _ in enumerate(q_values):
            if i == 0:
                start = refined_step
                end = q_values[i+1]
            elif i < len(q_values) - 1:
                start = q_values[i - 1] + refined_step    
                end = q_values[i + 1]
            else:
                start = q_values[i - 1] + refined_step
                end = 2.5
            ranges.append((start, end))
    
    # ranges[-1] = (1.8807, 1.8809)

    values_list = [
        np.arange(start, end, refined_step) for start, end in ranges
    ]

    # Generate all combinations of refined q_values
    q_combinations = itertools.product(*values_list)

    for k, q in enumerate(q_combinations):
        q = np.array(sorted(set(q)))
        q = np.insert(q, 0, 0)  # N+1

        q_minus = q[:-1] - q[1:]    # N 
        q_plus = q[:-1] + q[1:]     # N
        s = q_plus * 0.5            # N
        q_square_diff = q_minus * q_plus

        total_error = np.sqrt(2. / np.pi) * (np.sum(q_minus[1:] * np.exp(-0.5 * (s[1:] ** 2))) - q[1] * np.exp(-0.5 * (s[0] ** 2))) \
                        + 0.5 * (np.sum(q_square_diff * special.erf(s / np.sqrt(2))) + (q[-1] ** 2 + 1))
                        
        if total_error < min_error:
            min_error = total_error
            optimized_q_values = q

        if k % 10000 == 0:
            print(f"{k} iterations | q_values: {optimized_q_values} | Min Err.: {min_error}")

    print(f"Optimized | q_values: {optimized_q_values} | Min Err.: {min_error}")
    return optimized_q_values, min_error

# Example usage
q_values = [1.]  # Initial q_values
refined_step = 0.0001 # Step size for refined search
refined_range = 2.
optimized_q_values, min_error = refined_error_search(q_values, refined_step, refined_range)