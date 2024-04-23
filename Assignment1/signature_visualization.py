import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Main window creation using tkinter
root = tk.Tk()
root.title("Signature Visualization")  # Window title

# Add a canvas widget to the main window
canvas = tk.Canvas(root, width=400, height=150)
canvas.pack()

# Function to draw the signature on the canvas
def draw_signature(canvas, signature_data):
    canvas.delete("all")  # Clear the canvas before drawing a new signature
    scale = 1.0  # Scale factor for the signature coordinates
    line_width = 2  # Thickness of the signature lines

    # Loop through each stroke in the signature data
    for stroke in signature_data:
        # Extract the x, y coordinates from the stroke data
        points = [(x * scale, y * scale) for x, y, _, _, _, _ in stroke]
        # Draw each line segment in the stroke
        for i in range(len(points) - 1):
            canvas.create_line(points[i], points[i + 1], fill='black', width=line_width)

# Function to draw graphs of the signature data
def draw_graphs(signature_data):
    if not signature_data:
        return  # Do nothing if there is no data

    # Create 5 subplots for different data attributes: pressure, direction, altitude, x and y coordinates
    fig, axs = plt.subplots(5, figsize=(10, 12))

    # Labels for the y-axis of each subplot
    labels = ['X coordinate', 'Y coordinate', 'Pressure', 'Direction', 'Altitude']
    # Indices corresponding to each attribute in the stroke data
    indices = [0, 1, 2, 3, 4]

    # Create each subplot
    for i, (label, index) in enumerate(zip(labels, indices)):
        axs[i].set_ylabel(label)  # Set y-axis label
        # Set x-axis label only on the last subplot
        if i == len(labels) - 1:
            axs[i].set_xlabel('Time')

        # Plot the data for each stroke
        for stroke in signature_data:
            times = [t for _, _, _, _, _, t in stroke]  # Extract time points
            values = [s[index] for s in stroke]  # Extract attribute values based on index
            axs[i].plot(times, values)  # Plot the attribute values against time

    plt.tight_layout()  # Adjust the layout to prevent overlapping of subplots

    # Embed the matplotlib figure into the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    # Save the figure to a file
    fig.savefig('C:/Users/HuiShan/Documents/HAPP/Assignment1/Capture/Data1.png')

# Function to parse the SDT file and return the data
def parse_sdt_file(file_path):
    data = []  # Initialize an empty list to hold all strokes
    current_stroke = []  # Initialize an empty list for the current stroke

    # Open the file and read the lines
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line as it's a data count
        # Iterate over each line in the file
        for line in file:
            parts = line.strip().split()  # Split the line into parts
            if parts[0] == '-1':  # Stroke separator detected
                if current_stroke:  # If the current stroke has data, add it to the list
                    data.append(current_stroke)
                    current_stroke = []  # Reset current stroke to start a new one
            else:
                # Parse each part of the line into respective variables
                x, y, pressure, direction, altitude, time = map(int, parts)
                current_stroke.append((x, y, pressure, direction, altitude, time))

    # Add the last stroke to the list if it's not empty
    if current_stroke:
        data.append(current_stroke)

    return data

# Load the SDT file data
signature_data = parse_sdt_file('C:/Users/HuiShan/Documents/HAPP/Assignment1/SignatureSampleData/001.001.000.sdt')

# Check if there is data to display and call the drawing functions
if signature_data:
    draw_signature(canvas, signature_data)  # Draw the signature on the canvas
    draw_graphs(signature_data)  # Draw the graphs for signature data
else:
    print("No data available to display.")  # Display message if no data is found

# Start the tkinter main event loop
root.mainloop()

