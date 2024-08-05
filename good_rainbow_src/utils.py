import numpy as np


def merge_dicts(dict1, dict2):
    """
    Merge two dictionaries, with dict2 taking precedence if both have the same key.
    This function handles nested dictionaries of arbitrary depth.
    """
    merged = dict1.copy()  # Start with a copy of the first dictionary

    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # If both values are dictionaries, merge them recursively
            merged[key] = merge_dicts(merged[key], value)
        else:
            # Otherwise, use the value from the second dictionary
            merged[key] = value

    return merged


def downsample_data(data, max_data_points):
    """
    Downsample a multi-series numpy array to a specified number of data points.

    Parameters:
    - data: numpy array of shape (num_points, num_series)
    - max_data_points: int, the desired number of data points after downsampling

    Returns:
    - downsampled_data: numpy array of shape (max_data_points, num_series)
    - downsampled_indices: numpy array of shape (max_data_points,)
    """
    num_points, num_series = data.shape

    if num_points <= max_data_points:
        return data, np.arange(num_points)

    # Calculate the bucket size
    bucket_size = num_points / max_data_points

    # Initialize the downsampled data array
    downsampled_data = np.zeros((max_data_points, num_series))
    downsampled_indices = np.zeros(max_data_points, dtype=int)

    for i in range(max_data_points):
        start_idx = int(i * bucket_size)
        end_idx = int((i + 1) * bucket_size)

        # Ensure the last bucket includes the last data point
        if i == max_data_points - 1:
            end_idx = num_points

        # Calculate the average for each series in the current bucket
        downsampled_data[i, :] = np.mean(data[start_idx:end_idx, :], axis=0)
        downsampled_indices[i] = start_idx

    return downsampled_data, downsampled_indices


def discounted_rewards(rewards, gamma):
    """
    Compute the sum of discounted rewards.

    Parameters:
    rewards (np.ndarray): Array of rewards.
    gamma (float): Discount factor.

    Returns:
    float: Sum of discounted rewards.
    """
    n = len(rewards)
    discounts = np.logspace(0, n-1, num=n, base=gamma)
    return np.sum(rewards * discounts)

def sum_rewards(rewards):
    """
    Compute the sum of rewards.

    Parameters:
    rewards (np.ndarray): Array of rewards.

    Returns:
    float: Sum of rewards.
    """
    return np.sum(rewards)
