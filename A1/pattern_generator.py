# Assignment 1: NumPy Array Manipulation for 2D Pattern Generation

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# --------------- 1 ---------------
# Set canvas height in pixels
canvas_height = 500  
# Set canvas width equal to height to create a square canvas
canvas_width = canvas_height
# Create a 2D array of zeros with shape (canvas_height, canvas_width)
canvas = np.zeros((canvas_height, canvas_width))
# Create an explicit RGB canvas (H, W, 3)
canvas_rgb = np.zeros((canvas_height, canvas_width, 3))

# --------------- 2 ---------------
# Attractor Influence Section with Grid Pattern
num_attractors_per_row = 10 # Set the number of attractor points along the horizontal axis
num_attractors_per_col = 5 # Set the number of attractor points along the vertical axis
grid_x = np.linspace(0, canvas_width, num_attractors_per_row) # Create evenly spaced x-coordinates for attractor grid
grid_y = np.linspace(0, canvas_height, num_attractors_per_col) # Create evenly spaced y-coordinates for attractor grid
grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2) # Generate a grid of points from x and y coordinates and reshape to list of 2D points
jitter = (np.random.rand(len(grid_points), 2) - 1.0) * 100 # Generate random jitter for each attractor point in range [-100, 100] pixels

# Add jitter to grid points to create final attractor positions
attractors = grid_points + jitter


# --------------- 3 ---------------
# Define a function to calculate Euclidean distance from a point to all attractors
def distance(point1, point2):
    # Compute squared differences, sum along axis, take square root
    return np.sqrt(np.sum((point1 - point2) ** 2, axis=1))


# --------------- 4 ---------------
# Loop through each pixel in the canvas along the x-axis
for x in range(canvas_width):
    # Loop through each pixel in the canvas along the y-axis
    for y in range(canvas_height):
        # Calculate distances from current pixel to all attractor points
        dists = distance(np.array([x, y]), attractors)
        # Normalize pixel coordinates to range [0, 1]
        nx = x / canvas_width
        ny = y / canvas_height
        # Compute minimum and mean distances to attractors
        min_dist = np.min(dists)
        mean_dist = np.mean(dists)
        # Add gradient and noise components to create pattern
        gradient = np.sin(nx * np.pi * 2) * np.cos(ny * np.pi * 2)
        noise_field = np.sin(nx * 12.3 + ny * 4.7)
        # Combine components with specific weights to form final field value
        field = 0.5 * np.sin(min_dist * 0.4) + 0.3 * gradient + 0.2 * noise_field
        # Map field value to RGB channels using sine and cosine functions
        r = (np.sin(field * 2.0) + 1) * 0.5
        g = (np.cos(field * 1.5) + 1) * 0.5
        b = (np.sin(field * 1.2 + np.pi/3) + 1) * 0.5
        canvas_rgb[y, x, 0] = r
        canvas_rgb[y, x, 1] = g
        canvas_rgb[y, x, 2] = b



# --------------- 5 ---------------
# Add independent noise per RGB channel
noise_rgb = (np.random.rand(canvas_height, canvas_width, 3) - 0.5) * 0.5
canvas_rgb = canvas_rgb * 0.75 + noise_rgb * 0.25


# --------------- 6 ---------------
# Normalize canvas values to range [0, 1] to ensure valid color mapping
canvas_normalized = (canvas_rgb - canvas_rgb.min()) / (canvas_rgb.max() - canvas_rgb.min())

# --------------- 7 ---------------
# Display true RGB image (no colormap)
plt.imshow(canvas_normalized)
# Hide axis ticks and labels for cleaner visualization
plt.axis('off') 
# Show the generated image
plt.show()
# Save the image to the images folder with tight bounding box and no padding
# plt.savefig('images/attractor_grid_pattern.png', bbox_inches='tight', pad_inches=0)
