import numpy as np


def adjust_array(arr, some_var):
    # Step 1: Calculate the minimum threshold
    min_threshold = 1 / some_var

    # Step 2: Adjust elements below the threshold
    adjusted_arr = np.maximum(arr, min_threshold)

    # Step 3: Calculate the excess
    excess = np.sum(adjusted_arr) - 1

    # Step 4: Distribute the excess proportionally
    if excess > 0:
        # Find elements that were above the threshold
        above_threshold_indices = arr > min_threshold
        above_threshold_values = arr[above_threshold_indices]

        # Calculate the total sum of elements above the threshold
        total_above_threshold = np.sum(above_threshold_values)

        # Reduce the elements proportionally
        reduction_factors = above_threshold_values / total_above_threshold
        reduction_amounts = reduction_factors * excess

        # Apply the reductions
        adjusted_arr[above_threshold_indices] -= reduction_amounts

    return adjusted_arr


# Example usage
arr = np.array([0.001, 0.1, 0.699, 0.2])
some_var = 10
adjusted_arr = adjust_array(arr, some_var)
print(adjusted_arr)
print(np.sum(adjusted_arr))  # Should be 1