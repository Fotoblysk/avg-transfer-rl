import csv
import matplotlib.pyplot as plt

# Path to your CSV file
#csv_file_path = 'results/FrozenLake-map_0/stats/train_ep_data.csv'
#csv_file_path = 'results/MiniGrid-Empty-5x5-v0/stats/train_ep_data.csv'
#csv_file_path = 'results/MiniGrid-KeyCorridorS3R1-v0/stats/train_ep_data.csv'
#csv_file_path = 'results/MiniGrid-Fetch-8x8-N3-v0_07-08-2024_01-09/stats/train_ep_data.csv'
#csv_file_path = 'results/MiniGrid-Fetch-8x8-N3-v0_07-08-2024_01-46/stats/train_ep_data.csv'
#csv_file_path = 'results/MiniGrid-Fetch-8x8-N3-v0_07-08-2024_02-22/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Fetch-8x8-N3-v0_07-08-2024_03-16/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Fetch-8x8-N3-v0_07-08-2024_10-08/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Fetch-8x8-N3-v0_07-08-2024_10-31/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Fetch-8x8-N3-v0_07-08-2024_13-53/stats/train_ep_data.csv'
#csv_file_path = 'results/MiniGrid-Empty-8x8-v0_07-08-2024_16-52/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Empty-Random-6x6-v0_07-08-2024_19-18/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Empty-8x8-v0_07-08-2024_16-52/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Fetch-5x5-N2-v0_08-08-2024_11-57/stats/train_ep_data.csv'
csv_file_path = 'results/lunar_lander_8_20_2_08-08-2024_14-42/stats/train_ep_data.csv'
csv_file_path = 'results/FrozenLake-map_2_08-08-2024_16-15/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Empty-5x5-v0_09-08-2024_21-37/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Empty-5x5-v0_09-08-2024_22-05/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Empty-16x16-v0_10-08-2024_01-41/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Empty-16x16-v0_10-08-2024_01-42/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Empty-16x16-v0_10-08-2024_01-43/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Empty-8x8-v0_10-08-2024_13-25/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Empty-8x8-v0_10-08-2024_13-48/stats/train_ep_data.csv'
csv_file_path = 'results/MiniGrid-Empty-8x8-v0_10-08-2024_13-49/stats/train_ep_data.csv'

#csv_file_path = 'results/lunar_lander_10_0_0/stats/train_ep_data.csv'
#csv_file_path = 'results/CartPole/stats/train_ep_data.csv'

import csv
import matplotlib.pyplot as plt
import numpy as np

# Path to your CSV file

# Initialize lists to store frame_idx and score
frame_idx = []
scores = []

# Read the CSV file
with open(csv_file_path, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        frame_idx.append(int(row['frame_idx']))
        scores.append(float(row['score']))


# Function to compute moving average with an odd window size
def moving_average(data, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number")

    half_window = window_size // 2
    moving_avg = []

    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        window_avg = np.mean(data[start_idx:end_idx])
        moving_avg.append(window_avg)

    return moving_avg


# Compute moving average with an odd window size
window_size = 51  # Example window size, must be odd
moving_avg_scores = moving_average(scores, window_size)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(frame_idx, scores,  label='Original Scores')
plt.plot(frame_idx, moving_avg_scores,  label='Moving Average')
plt.xlabel('Frame Index')
plt.ylabel('Score')
plt.title('Score vs Frame Index with Moving Average')
plt.legend()
plt.grid(True)
plt.show()