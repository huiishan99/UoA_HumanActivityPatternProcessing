import numpy as np
import matplotlib.pyplot as plt

# Function to load data from a file and convert it to a NumPy array.
# Each line of the file is assumed to contain one float number.
def load_data(filename):
    with open(filename, 'r') as file:
        data = np.array([float(line.strip()) for line in file])
    return data

# Load data for Signal A and Signal B from text files.
data_a = load_data('data_a.txt')
data_b = load_data('data_b.txt')

# Plot and save the original signals A and B for initial visualization.
plt.figure(figsize=(10, 8))
plt.plot(data_a, label='Signal A')
plt.plot(data_b, label='Signal B')
plt.title('Original Signals')
plt.legend()
plt.savefig('original_signals.png')  # Save the figure to a file
plt.show()  # Display the figure

# Initialize the DP matrix with zeros and set the first element based on the first data points of A and B.
n, m = len(data_a), len(data_b)
dp = np.zeros((n, m))
dp[0, 0] = np.abs(data_a[0] - data_b[0])

# Fill the DP matrix with the minimum cost paths based on the recurrence relation defined.
for i in range(1, n):
    dp[i, 0] = dp[i-1, 0] + np.abs(data_a[i] - data_b[0])
for j in range(1, m):
    dp[0, j] = dp[0, j-1] + np.abs(data_a[0] - data_b[j])
for i in range(1, n):
    for j in range(1, m):
        cost = np.abs(data_a[i] - data_b[j])
        dp[i, j] = min(dp[i-1, j-1], dp[i, j-1], dp[i-1, j]) + cost

# Backtracking from the bottom-right corner of the matrix to find the optimal matching path.
i, j = n - 1, m - 1
path = []
while i > 0 and j > 0:
    path.append((i, j))
    if dp[i, j] == dp[i-1, j-1] + np.abs(data_a[i] - data_b[j]):
        i, j = i-1, j-1
    elif dp[i, j] == dp[i, j-1] + np.abs(data_a[i] - data_b[j]):
        j -= 1
    else:
        i -= 1
path.append((i, j))
path.reverse()  # Reverse the path to start from the beginning

# Adjust the time coordinate of Signal B based on the optimal path found during backtracking.
adjusted_b = np.empty(n)
for index, (i, j) in enumerate(path):
    adjusted_b[i] = data_b[j]

# Plot and save the adjusted signals where Signal B has been aligned to Signal A.
plt.figure(figsize=(10, 8))
plt.plot(data_a, label='Signal A')
plt.plot(adjusted_b, label='Adjusted Signal B')
plt.title('DP Matching Adjusted Signals')
plt.legend()
plt.savefig('adjusted_signals.png')  # Save the adjusted figure to a file
plt.show()  # Display the figure
