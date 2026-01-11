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
        # Determine pixel intensity using sine of the minimum distance (smooth gradient)
        color = np.sin(np.min(dists * 0.5)) 
        # Assign calculated color value to the canvas at position (y, x)
        canvas[y, x] = color


# --------------- 5 ---------------
# Adding Noise Section
# Generate random noise array with values in range [-2.5, 2.5]
noise = (np.random.rand(canvas_height, canvas_width) - 0.5) * 5.0 
# Blend the noise with the canvas values (75% original, 25% noise)
canvas = canvas * 0.75 + noise * 0.25


# --------------- 6 ---------------
# Normalize canvas values to range [0, 1] to ensure valid color mapping
canvas_normalized = (canvas - canvas.min()) / (canvas.max() - canvas.min())


# --------------- 7 ---------------
# Display the canvas as an image using the 'magma' colormap
plt.imshow(canvas_normalized, cmap='magma') 
# Hide axis ticks and labels for cleaner visualization
plt.axis('off') 
# Show the generated image
plt.show()
# Save the image to the images folder with tight bounding box and no padding
plt.savefig('images/attractor_grid_pattern.png', bbox_inches='tight', pad_inches=0)
