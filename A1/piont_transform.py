import tkinter as tk
import numpy as np

# Create an empty list to store the points
points = []

# Function to add a point to the canvas
def add_point(event):
    x, y = event.x, event.y
    x = round(x / grid_size) * grid_size  # Snap to the nearest grid position
    y = round(y / grid_size) * grid_size
    canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="blue")
    points.append((x, y))

# Function to transform the points
def transform_points():
    try:
        tx = float(tx_entry.get())
        ty = float(ty_entry.get())
        angle = np.radians(float(rotation_entry.get()))
        scaling = float(scaling_entry.get())

        transformation_matrix = np.array([
            [scaling * np.cos(angle), -scaling * np.sin(angle), tx],
            [scaling * np.sin(angle), scaling * np.cos(angle), ty],
            [0, 0, 1]
        ])

        transformed_points = []

        for point in points:
            point_vector = np.array([point[0], point[1], 1])
            transformed_point = np.dot(transformation_matrix, point_vector)
            transformed_points.append((transformed_point[0], transformed_point[1]))

        display_transformed_points(transformed_points)
    except ValueError:
        result_label.config(text="Invalid input")

# Function to display the transformed points on the canvas
def display_transformed_points(transformed_points):
    canvas.delete("points")
    for point in transformed_points:
        canvas.create_oval(point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2, fill="red", tags="points")

# Create the main window
root = tk.Tk()
root.title("Point Transformation")

# Define the grid size
grid_size = 10

# Left panel for displaying points
left_panel = tk.Frame(root)
left_panel.pack(side="left")

canvas = tk.Canvas(left_panel, width=300, height=300)
canvas.pack()
canvas.bind("<Button-1>", add_point)  # Bind click event to add points

# Create a grid on the canvas
for x in range(0, 300, grid_size):
    canvas.create_line(x, 0, x, 300, fill="gray", dash=(2, 2))

for y in range(0, 300, grid_size):
    canvas.create_line(0, y, 300, y, fill="gray", dash=(2, 2))

# Right panel for user input
right_panel = tk.Frame(root)
right_panel.pack(side="right")

# Labels and Entry widgets for transformation parameters
tx_label = tk.Label(right_panel, text="tx:")
tx_label.pack()
tx_entry = tk.Entry(right_panel)
tx_entry.pack()

ty_label = tk.Label(right_panel, text="ty:")
ty_label.pack()
ty_entry = tk.Entry(right_panel)
ty_entry.pack()

rotation_label = tk.Label(right_panel, text="Rotation Angle (degrees):")
rotation_label.pack()
rotation_entry = tk.Entry(right_panel)
rotation_entry.pack()

scaling_label = tk.Label(right_panel, text="Scaling Ratio:")
scaling_label.pack()
scaling_entry = tk.Entry(right_panel)
scaling_entry.pack()

transform_button = tk.Button(right_panel, text="Transform", command=transform_points)
transform_button.pack()

result_label = tk.Label(right_panel, text="")
result_label.pack()

root.mainloop()
