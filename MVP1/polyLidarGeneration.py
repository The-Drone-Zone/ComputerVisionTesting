import numpy as np
import pandas as pd

# Constants
total_points = 500
max_lidar_distance = 50 # Maximum LiDAR reading in meters
line_length = 175  # Length of lines / 2 (horiz FOV)
num_lines = 12 # Number of horizontal line scans
scan_height = 175 # Scan box (vert FOV), horizontal space that the scans distribute between
num_scans = 1  # Number of scans

# Calculated Values
num_points_per_line = (int)(total_points / num_lines)  # Points per line
print('Num Point per Line: {}'.format(num_points_per_line))

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
        x_line = np.linspace(-line_length, line_length, num_points_per_line)
        # vertical line
        y_counter = 0
        for x in x_line:
            y = (line_id * (scan_height / num_lines)) + (y_counter * ((scan_height/num_lines)/num_points_per_line))
            distances.append(simulate_lidar_reading())
            x_positions.append(x + 300)
            y_positions.append(y + 150)
            scan_numbers.append(scan_id)
            line_num.append("horizontal")
            y_counter += 1

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
