import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to parse SDT files containing signature data
def parse_sdt_file(file_path):
    data = []  # Initialize a list to store all strokes
    current_stroke = []  # Initialize a list to store the current stroke
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line (usually header or metadata)
        for line in file:
            parts = line.strip().split()
            if parts[0] == '-1':  # Stroke delimiter
                if current_stroke:  # Check if there is data in the current stroke
                    data.append(current_stroke)
                    current_stroke = []  # Reset current stroke for the next one
            else:
                # Convert coordinates and other data to integers and store them
                x, y, pressure, direction, altitude, time = map(int, parts)
                current_stroke.append((x, y, pressure, direction, altitude, time))
    if current_stroke:
        data.append(current_stroke)
    return data

# Function to calculate distance between two signatures using linear matching
def calculate_distance(signature1, signature2):
    total_distance = 0
    N = len(signature1)  # Assume both signatures have the same number of strokes
    for i in range(N):
        stroke1 = signature1[i]
        stroke2 = signature2[i]
        stroke_distance = 0
        n = min(len(stroke1), len(stroke2))  # Use the smaller number of points to avoid index errors
        for j in range(n):
            x1, y1 = stroke1[j][:2]
            x2, y2 = stroke2[j][:2]
            stroke_distance += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        average_distance = stroke_distance / n
        total_distance += average_distance
    return total_distance

# Function to calculate distance using Dynamic Programming (DP) matching
def dp_matching(stroke1, stroke2):
    n = len(stroke1)
    m = len(stroke2)
    dp = np.full((n + 1, m + 1), float('inf'))  # Initialize DP table with infinite values
    dp[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt((stroke1[i - 1][0] - stroke2[j - 1][0]) ** 2 + (stroke1[i - 1][1] - stroke2[j - 1][1]) ** 2)
            dp[i, j] = min(dp[i - 1, j - 1] + cost, dp[i - 1, j] + cost, dp[i, j - 1] + cost)
    return dp[n, m]

# Function to calculate DP matching distance for two signatures
def calculate_dp_distance(signature1, signature2):
    total_distance = 0
    N = len(signature1)
    for i in range(N):
        stroke1 = signature1[i]
        stroke2 = signature2[i]
        stroke_distance = dp_matching(stroke1, stroke2)
        total_distance += stroke_distance
    return total_distance

# Function to plot the matching of two signatures
def plot_matching(signature1, signature2):
    fig, ax = plt.subplots(figsize=(10, 6))
    scale = 1.0  # Scaling factor for visualization
    for stroke1, stroke2 in zip(signature1, signature2):
        x1, y1 = zip(*[(x * scale, y * scale) for x, y, _, _, _, _ in stroke1])
        x2, y2 = zip(*[(x * scale, y * scale) for x, y, _, _, _, _ in stroke2])
        ax.plot(x1, y1, 'r-')  # Plot stroke1
        ax.plot(x2, y2, 'b-')  # Plot stroke2
        # Draw matching lines between points
        for (x1i, y1i), (x2i, y2i) in zip(zip(x1, y1), zip(x2, y2)):
            ax.plot([x1i, x2i], [y1i, y2i], 'grey', linestyle='--', linewidth=0.3)
    ax.set_title('Linear Matching of Two Signatures')
    ax.legend(['Signature 1', 'Signature 2', 'Matching'])
    plt.show()

# Main function
def main():
    signature_data1 = parse_sdt_file('001.001.000.sdt')
    signature_data2 = parse_sdt_file('001.001.001.sdt')
    distance = calculate_distance(signature_data1, signature_data2)
    dp_distance = calculate_dp_distance(signature_data1, signature_data2)
    print("Linear Matching Calculated Distance:", distance)
    print("DP Matching Calculated Distance:", dp_distance)
    plot_matching(signature_data1, signature_data2)

if __name__ == '__main__':
    main()
