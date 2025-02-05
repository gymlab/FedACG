import itertools
import numpy as np
import math
from scipy import special
from scipy.stats import norm


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

        q_minus = q[:-1] - q[1:]    # L-1  [q1-q2, ..., qL-1 - qL]
        q_plus = q[:-1] + q[1:]     # L-1  [q1+q2, ..., qL-1 + qL]
        s = q_plus * 0.5            # L-1 [(q1+q2)/2, ..., (qL-1 + qL)/2]
        s = np.insert(s, 0, 0)      # L [0, (q1+q2)/2, ..., (qL-1 + qL)/2]
        q_square_diff = q_minus * q_plus

        total_error = np.sqrt(2. / np.pi) * (np.sum(q_minus * np.exp(-0.5 * (s[1:] ** 2))) - q[0]) \
                        + 0.5 * (np.sum(q_square_diff * special.erf(s[1:] / np.sqrt(2)))  + (q[-1] ** 2 + 1))

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
q_values = [0.1284, 0.3881, 0.6568, 0.9424, 1.2563, 1.6181, 2.0691, 2.327]  # Initial q_values
# q_values = [0.2303, 0.4648, 0.7081, 0.9663, 1.2481, 1.5676, 1.9676, 2.6488]  # Initial q_values

# q_ranges = [(0.005, 0.25), (0.2001, 0.4),
#             (0.4001, 0.7), (0.7, 1.0),
#             (1.0, 1.3), (1.3, 1.5),
#             (1.7, 1.9), (2.0, 3.1)]
# q_ranges = [(0.0001, 0.3499), (0.2001, 0.5499),
#             (0.3501, 0.7499), (0.5501, 0.9999),
#             (0.7501, 1.2999), (1.0001, 1.7999),
#             (1.3001, 3.1)]

steps = 5
refined_range = 0.0002

q_ranges = []
for i, q in enumerate(q_values):
    q_ranges.append((q-refined_range, q+refined_range))
# q_ranges[-1] = (2.5, 2.8)

optimized_q_values, min_error = refined_error_search(q_values, q_ranges, steps)