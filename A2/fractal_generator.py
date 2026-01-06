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
# Attractor strength
ATTRACTOR_STRENGTH = 1.0   # values 1.0 – 6.0

def generate_fractal(start_point, angle, length, depth, max_depth,
                     angle_change, length_scaling_factor,
                     wind_vector, attractors, min_distance=5):
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
    - wind_vector: Tuple (wx, wy), wind direction vector.
    - attractors: List of tuples [(ax1, ay1), (ax2, ay2), ...], attractor points.
    """
    if depth >= max_depth or length < 1:
        return []
    
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

    # Attractor influence
    for ax, ay in attractors:
        # Vector from current point to attractor (pull towards)
        vec_x = ax - start_point[0]
        vec_y = ay - start_point[1]
        distance = math.hypot(vec_x, vec_y)
        if distance > 0:
            # Attraction strength decreases with distance
            influence = min((12 / distance) * ATTRACTOR_STRENGTH, 0.12 * ATTRACTOR_STRENGTH)
            angle_to_attractor = math.degrees(math.atan2(vec_y, vec_x))
            # Blend the angle toward the attractor
            angle = (1 - influence) * angle + influence * angle_to_attractor

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
            return []
        
    line_list.append(line)

    new_length = length * length_scaling_factor
    # Increment depth
    next_depth = depth + 1

    # Recursive calls for branches replaced with return of children
    return [
        (end_point, angle + angle_change, new_length, next_depth),
        (end_point, angle - angle_change, new_length, next_depth)
    ]


# Main execution
if __name__ == "__main__":
    wind_vector = (5.0, 2.0) # wind vector blowing to the right and slightly upwards
    # Random number of trees
    num_trees = random.randint(4, 10)

    # Horizontal spread for tree origins
    spread = random.uniform(500, 1000)

    # Generate random starting points along the ground
    start_points = [
        (random.uniform(-spread / 2, spread / 2), 0)
        for _ in range(num_trees)
    ]

    # Number of attractor points
    num_attractors = 4

    # Create attractor points
    attractors = [
        (random.uniform(-200, 800), random.uniform(150, 350)) # x, y coordinates for attractors
        for _ in range(num_attractors)
    ]

    # Clear the line list
    line_list.clear()

    # Initialize growth queue (interleaved growth)
    growth_queue = []

    for sp in start_points:
        growth_queue.append((sp, 90, 100, 0)) # start_point, angle, length, depth

    max_depth = 20

    while growth_queue:
        start_point, angle, length, depth = growth_queue.pop(0)

        children = generate_fractal(
            start_point=start_point,
            angle=angle,
            length=length,
            depth=depth,
            max_depth=max_depth,
            angle_change=30,
            length_scaling_factor=0.75,
            wind_vector=wind_vector,
            attractors=attractors
        )

        for child in children:
            growth_queue.append(child)


    # Visualization
    fig, ax = plt.subplots()
    for line in line_list:
        x, y = line.xy
        ax.plot(x, y, color='green', linewidth=1)

    # Visualize attractors
    ax.scatter(
        [a[0] for a in attractors],
        [a[1] for a in attractors],
        color='red',
        s=60,
        marker='.',
        zorder=5,
        label='Attractors'
    )

    # Optional: Customize the plot
    ax.set_aspect('equal')
    ax.legend(loc='lower left')
    plt.axis('off')
    plt.show()

    # Save the figure
    # fig.savefig('images/fractal_tree_V1.png', dpi=300, bbox_inches='tight')
    # Repeat the process with different parameters for additional fractals
    # ...