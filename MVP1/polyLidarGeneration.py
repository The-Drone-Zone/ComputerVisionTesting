import numpy as np
import pandas as pd

# Constants
max_lidar_distance = 50 # Maximum LiDAR reading in meters
line_length = 100.0  # Length of lines (vert FOV)
num_points_per_line = 25  # Points per line
num_lines = 2 # Number of vertical line scans
scan_width = 300 # Scan box (horz FOV), horizontal space that the scans distribute between
num_scans = 1  # Number of scans

# Function to simulate a LiDAR reading
def simulate_lidar_reading():
    return np.random.uniform(10, max_lidar_distance)

# Initialize dataset lists
distances = []
x_positions = []
y_positions = []
scan_numbers = []
line_num = []

# Generate data
for scan_id in range(num_scans):
    for line_id in range(num_lines):
        y_line = np.linspace(-line_length, line_length, num_points_per_line)
        # vertical line
        for y in y_line:
            x = line_id * (scan_width / num_lines)
            distances.append(simulate_lidar_reading())
            x_positions.append(x + 200)
            y_positions.append(y + 200)
            scan_numbers.append(scan_id)
            line_num.append("horizontal")

# Combine into a DataFrame
lidar_data = pd.DataFrame({
    "distance": distances,
    "x_position": x_positions,
    "y_position": y_positions,
    "scan_number": scan_numbers,
    "line_num": line_num,
})

# Save to CSV
output_file = "poly_lidar_reading_dataset.csv"
lidar_data.to_csv(output_file, index=False)
print(f"Dataset saved to {output_file}")
