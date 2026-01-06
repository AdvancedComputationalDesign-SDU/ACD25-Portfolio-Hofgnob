"""
Assignment 2: Fractal Generator

Author: Jesper Christensen Sørensen

Description:
This script generates fractal patterns using recursive functions and geometric transformations.
"""

# Import necessary libraries
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import LineString
from shapely.affinity import rotate, translate
import random

# ============================================================
# CONFIGURATION PARAMETERS
# ============================================================

### Randomness ###
random_seed = 42                 # Seed for reproducible results

### Wind field ###
wind_vector = (5.0, 2.0)         # Global wind direction (x, y)
global_wind_strength_range = (1.0, 7.0)  # Range for global wind strength
wind_jitter_range = (-0.3, 0.3)  # Local turbulence per branch

### Attractor field ###
num_attractors = 4               # Number of attractor points
attractor_strength = 1.0         # Base pull toward attractors
attractor_radius = 100           # Radius for amplified attractor influence
attractor_boost = 2.0            # Strength multiplier inside attractor radius

### Tree population ###
num_trees_range = (4, 10)        # Min / max number of trees
tree_spread_range = (500, 1000)  # Horizontal spread of tree origins
min_start_distance = 150         # Minimum spacing between tree trunks

### Recursive growth ###
max_depth = 20                   # Maximum recursion depth
base_branch_angle = 30            # Branching angle in degrees
length_scaling_factor = 0.75     # Length reduction per generation
min_segment_length = 1.0          # Absolute cutoff for growth

### Canopy region ###
canopy_height = 225               # Height above which branching increases
canopy_branch_boost = 3           # Extra branches spawned in canopy

### Twig (secondary fractal) ###
twig_length_threshold = 18        # Branch length below which twigs appear
twig_branches = 3                 # Number of twig branches
twig_angle_spread = 40            # Angular spread of twig fractal
twig_length_scale = 0.45          # Length scale for twig segments

### Animation & drawing ###
steps_per_frame = 4               # Growth steps per animation frame
min_thickness = 0.3               # Minimum branch thickness
max_thickness = 3.0               # Maximum branch thickness
animation_interval = 120          # Milliseconds between frames


random.seed(random_seed)
global_wind_strength = random.uniform(*global_wind_strength_range)

# Store all generated line segments with depth for thickness control
line_list = []  # each entry: (LineString, depth)


def generate_fractal(start_point, angle, length, depth, max_depth,
                     angle_change, length_scaling_factor,
                     wind_vector, attractors, min_distance=5):

    if depth >= max_depth or length < min_segment_length: # Base case for recursion
        return []
    
    # Wind field: global flow with local turbulence (stronger on smaller branches)
    wind_jitter = random.uniform(*wind_jitter_range)

    # Normalize length influence (small branches move more)
    length_factor = max(0.2, 1.0 - (length / 100))

    wind_strength = (global_wind_strength + wind_jitter) * length_factor

    # Apply wind as a vector force (correct lateral behaviour)
    dx = math.cos(math.radians(angle))
    dy = math.sin(math.radians(angle))

    # Wind pushes direction laterally
    dx += wind_vector[0] * wind_strength * 0.01
    dy += wind_vector[1] * wind_strength * 0.01

    # Recompute angle from resulting direction
    angle = math.degrees(math.atan2(dy, dx))

    # Attractor field: bends growth direction toward spatial targets
    for ax, ay in attractors:
        # Vector from current point to attractor (pull towards)
        vec_x = ax - start_point[0]
        vec_y = ay - start_point[1]
        distance = math.hypot(vec_x, vec_y)
        if distance > 0:
            # Region-aware attractor amplification
            local_strength = attractor_strength
            if distance < attractor_radius:
                local_strength *= attractor_boost

            influence = min((12 / distance) * local_strength, 0.12 * local_strength)
            angle_to_attractor = math.degrees(math.atan2(vec_y, vec_x))
            # Blend the angle toward the attractor
            angle = (1 - influence) * angle + influence * angle_to_attractor

    # Compute end point
    end_x = start_point[0] + length * math.cos(math.radians(angle))
    end_y = start_point[1] + length * math.sin(math.radians(angle))
    end_point = (end_x, end_y)

    # Create a line segment using Shapely
    line = LineString([start_point, end_point]) 
    
    # Self-avoidance: prevents branches from intersecting existing geometry
    for existing, _ in line_list:
        if existing.touches(line):
            continue
        if line.distance(existing) < min_distance:
            return []
        
    line_list.append((line, depth))

    new_length = length * length_scaling_factor
    # Increment depth
    next_depth = depth + 1

    branches = [
        (end_point, angle + angle_change, new_length, next_depth),
        (end_point, angle - angle_change, new_length, next_depth)
    ]

    # Secondary twig fractal: fine-scale branching near tips
    if length < twig_length_threshold:
        for i in range(twig_branches):
            twig_angle = angle + random.uniform(
                -twig_angle_spread / 2,
                twig_angle_spread / 2
            )
            branches.append(
                (
                    end_point,
                    twig_angle,
                    new_length * twig_length_scale,
                    next_depth
                )
            )

    # Canopy rule: increase branching density above height threshold
    if end_point[1] > canopy_height:
        for _ in range(canopy_branch_boost):
            branches.append(
                (
                    end_point,
                    angle + random.uniform(-angle_change, angle_change),
                    new_length * random.uniform(0.7, 0.95),
                    next_depth
                )
        )

    return branches


# Main execution
if __name__ == "__main__":
    # Random number of trees
    num_trees = random.randint(*num_trees_range)

    # Horizontal spread for tree origins
    spread = random.uniform(*tree_spread_range)

    # Generate random starting points along the ground with minimum spacing
    start_points = []
    attempts = 0
    max_attempts = 1000

    while len(start_points) < num_trees and attempts < max_attempts:
        candidate = (random.uniform(-spread / 2, spread / 2), 0)
        if all(abs(candidate[0] - p[0]) >= min_start_distance for p in start_points):
            start_points.append(candidate)
        attempts += 1

    # Create attractor points
    attractors = [
        (random.uniform(-500, 500), random.uniform(150, 350)) # x, y coordinates for attractors
        for _ in range(num_attractors)
    ]

    # Clear the line list
    line_list.clear()

    # Initialize growth queue (interleaved growth)
    growth_queue = []

    for sp in start_points:
        growth_queue.append((sp, 90, 100, 0)) # start_point, angle, length, depth


    # Storage for incremental drawing
    drawn_lines = []

    def update(frame):
        # Perform a fixed number of growth steps per frame
        for _ in range(steps_per_frame):
            if not growth_queue:
                return

            start_point, angle, length, depth = growth_queue.pop(0)

            children = generate_fractal(
                start_point=start_point,
                angle=angle,
                length=length,
                depth=depth,
                max_depth=max_depth,
                angle_change=base_branch_angle,
                length_scaling_factor=length_scaling_factor,
                wind_vector=wind_vector,
                attractors=attractors
            )

            for child in children:
                growth_queue.append(child)

        # Draw only newly added lines
        while len(drawn_lines) < len(line_list):
            line, depth = line_list[len(drawn_lines)]
            x, y = line.xy

            seg_length = line.length
            linewidth = max(
                min_thickness,
                max_thickness * (seg_length / 100.0) ** 1.2
            )

            ax.plot(x, y, color='green', linewidth=linewidth)
            drawn_lines.append(line)

    # Visualization
    fig, ax = plt.subplots()

    ani = FuncAnimation(
        fig,
        update,
        interval=animation_interval,
        repeat=False
    )

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
