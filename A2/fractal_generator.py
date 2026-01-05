"""
Assignment 2: Fractal Generator

Author: Jesper Christensen Sørensen

Description:
This script generates fractal patterns using recursive functions and geometric transformations.
"""

# Import necessary libraries
import math
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.affinity import rotate, translate
import random

# Global list to store all line segments
line_list = []

random.seed(42)
# Global coherent wind blowing to the right
GLOBAL_WIND_STRENGTH = random.uniform(1.0, 7.0)


def generate_fractal(start_point, angle, length, depth, max_depth,angle_change, length_scaling_factor, wind_vector, min_distance=5):
    """
    Recursive function to generate fractal patterns.

    Parameters:
    - start_point: Tuple (x, y), starting coordinate.
    - angle: Float, current angle in degrees.
    - length: Float, length of the current line segment.
    - depth: Int, current recursion depth.
    - max_depth: Int, maximum recursion depth.
    - angle_change: Float, angle change at each recursion.
    - length_scaling_factor: Float, scaling factor for the length.
    """
    if depth >= max_depth or length < 1:
        return
    
    # Wind field influence
    # Stronger wind influence on smaller branches
    WIND_JITTER = random.uniform(-0.3, 0.3)

    # Normalize length influence (small branches move more)
    length_factor = max(0.2, 1.0 - (length / 100))

    WIND_STRENGTH = (GLOBAL_WIND_STRENGTH + WIND_JITTER) * length_factor

    # Apply wind as a vector force (correct lateral behaviour)
    dx = math.cos(math.radians(angle))
    dy = math.sin(math.radians(angle))

    # Wind pushes direction laterally
    dx += wind_vector[0] * WIND_STRENGTH * 0.01
    dy += wind_vector[1] * WIND_STRENGTH * 0.01

    # Recompute angle from resulting direction
    angle = math.degrees(math.atan2(dy, dx))

    # Compute end point
    end_x = start_point[0] + length * math.cos(math.radians(angle))
    end_y = start_point[1] + length * math.sin(math.radians(angle))
    end_point = (end_x, end_y)

    # Create a line segment using Shapely
    line = LineString([start_point, end_point]) 
    
    # Self-avoidance / intersection check
    for existing in line_list:
        if existing.touches(line):
            continue
        if line.distance(existing) < min_distance:
            return
        
    line_list.append(line)

    new_length = length * length_scaling_factor
    # Increment depth
    next_depth = depth + 1

    # Recursive calls for branches
    generate_fractal(end_point, angle + angle_change, new_length, next_depth, max_depth, angle_change, length_scaling_factor, wind_vector, min_distance)
    generate_fractal(end_point, angle - angle_change, new_length, next_depth, max_depth, angle_change, length_scaling_factor, wind_vector, min_distance)


# Main execution
if __name__ == "__main__":
    wind_vector = (5.0, 2.0) # wind vector blowing to the right and slightly upwards
    # Two trees with random separation
    separation = random.uniform(150, 350)
    start_points = [(-separation / 2, 0), (separation / 2, 0)]

    # Clear the line list
    line_list.clear()

    # Generate the fractal
    for sp in start_points:
        generate_fractal(
            start_point=sp,
            angle=90,
            length=100,
            depth=0,
            max_depth=20,
            angle_change=25,
            length_scaling_factor=0.7,
            wind_vector=wind_vector
        )


    # Visualization
    fig, ax = plt.subplots()
    for line in line_list:
        x, y = line.xy
        ax.plot(x, y, color='green', linewidth=1)

    # Optional: Customize the plot
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()

    # # Save the figure
    # fig.savefig('images/fractal_tree.png', dpi=300, bbox_inches='tight')

    # Repeat the process with different parameters for additional fractals
    # ...