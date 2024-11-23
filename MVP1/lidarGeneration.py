import numpy as np
import pandas as pd

# Constants
max_lidar_distance = 50 # Maximum LiDAR reading in meters
max_distance = 100.0 # Length of lines
num_points_per_line = 25  # Points per line
num_scans = 1  # Number of scans

# Define scanning patterns
x_horizontal = np.linspace(-max_distance, max_distance, num_points_per_line)
y_horizontal = np.zeros(num_points_per_line)

x_diag1 = np.linspace(-max_distance, max_distance, num_points_per_line)
y_diag1 = -x_diag1

x_diag2 = np.linspace(-max_distance, max_distance, num_points_per_line)
y_diag2 = x_diag2

# Function to simulate a LiDAR reading
def simulate_lidar_reading():
    return np.random.uniform(10, max_lidar_distance)

# Initialize dataset lists
distances = []
x_positions = []
y_positions = []
scan_numbers = []
line_types = []

# Generate data
for scan_id in range(num_scans):
    # Horizontal line
    for x in x_horizontal:
        y = 200
        distances.append(simulate_lidar_reading())
        x_positions.append(x + 325)
        y_positions.append(y)
        scan_numbers.append(scan_id)
        line_types.append("horizontal")

    # Diagonal line (top-right to bottom-left)
    for x in x_diag1:
        y = -x + 200
        distances.append(simulate_lidar_reading())
        x_positions.append(x + 325)
        y_positions.append(y)
        scan_numbers.append(scan_id)
        line_types.append("diagonal1")

    # Diagonal line (top-left to bottom-right)
    for x in x_diag2:
        y = x + 200
        distances.append(simulate_lidar_reading())
        x_positions.append(x + 325)
        y_positions.append(y)
        scan_numbers.append(scan_id)
        line_types.append("diagonal2")

# Combine into a DataFrame
lidar_data = pd.DataFrame({
    "distance": distances,
    "x_position": x_positions,
    "y_position": y_positions,
    "scan_number": scan_numbers,
    "line_type": line_types,
})

# Save to CSV
output_file = "lidar_reading_dataset.csv"
lidar_data.to_csv(output_file, index=False)
print(f"Dataset saved to {output_file}")
