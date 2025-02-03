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


def refined_error_search(q_values, q_ranges=None, steps=10, se=3.1):
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
    
    values_list = [
            np.linspace(start, end, steps) for start, end, in q_ranges
        ]

    # Generate all combinations of refined q_values
    q_combinations = itertools.product(*values_list)

    for k, q in enumerate(q_combinations):
        q = np.array(sorted(set(q)))
        q = np.insert(q, 0, 0)  # L+1   [0, q1, ..., qL]

        q_minus = q[:-1] - q[1:]    # L [-q1, q1-q2, ..., qL-1 - qL]
        q_plus = q[:-1] + q[1:]     # L [q1, q1+q2, ..., qL-1 + qL]
        s = q_plus * 0.5            # L [q1/2, (q1+q2)/2, ..., (qL-1 + qL)/2]
                                    # L [s1, s2, ..., sL]
        q_square_diff = q_minus * q_plus

        total_error = np.sqrt(2. / np.pi) * (np.sum(q_minus[1:] * np.exp(-0.5 * (s[1:] ** 2))) - q[1] * np.exp(-0.5 * (s[0] ** 2))) \
                        + 0.5 * (np.sum(q_square_diff[1:] * special.erf(s[1:] / np.sqrt(2))) 
                                 - (q[1]**2) * special.erf(s[0] / np.sqrt(2)) + (q[-1] ** 2 + 1))

        if total_error < min_error:
            min_error = total_error
            optimized_q_values = q

        if k % 10000 == 0:
            formatted_q_values = np.array2string(optimized_q_values, formatter={'float_kind': lambda x: f"{x:.4f}"})
            print(f"{k} iterations | q_values: {formatted_q_values} | Min Err.: {min_error}")

    formatted_q_values = np.array2string(optimized_q_values, formatter={'float_kind': lambda x: f"{x:.4f}"})
    print(f"Optimized | q_values: {formatted_q_values} | Min Err.: {min_error}")
    return optimized_q_values, min_error

# Example usage
q_values = [0.2686, 0.5439, 0.8337, 1.1490, 1.5080, 1.9735, 2.6536]  # Initial q_values
# q_values = [0.2303, 0.4648, 0.7081, 0.9663, 1.2481, 1.5676, 1.9676, 2.6488]  # Initial q_values

# q_ranges = [(0.0001, 0.3999), (0.2001, 0.5999),
#             (0.4001, 0.7999), (0.6001, 0.9999),
#             (0.8001, 1.1999), (1.0001, 1.3999),
#             (1.2001, 1.7999), (1.4001, 3.1)]
q_ranges = [(0.0001, 0.3499), (0.2001, 0.5499),
            (0.3501, 0.7499), (0.5501, 0.9999),
            (0.7501, 1.2999), (1.0001, 1.7999),
            (1.3001, 3.1)]

steps = 7
refined_range = 0.1/400

q_ranges = []
for i, q in enumerate(q_values):
    q_ranges.append((q-refined_range, q+refined_range))
# q_ranges[-1] = (2.5, 2.7)

optimized_q_values, min_error = refined_error_search(q_values, q_ranges, steps)