---
layout: default
title: Project Documentation
parent: "A1: NumPy Array Manipulation for 2D Pattern Generation"
nav_order: 2
nav_exclude: false
search_exclude: false
---

# Assignment 1: NumPy Array Manipulation for 2D Pattern Generation

[View on GitHub]({{ site.github.repository_url }})

![Example Image](images/perlin_moire.png)

## Project Overview

The project generates a full-color procedural image by combining distance-based attractor fields, spatial gradients, and structured noise.  
Rather than relying on predefined colormaps, color values are explicitly constructed and manipulated per RGB channel using NumPy arrays.

---

## Repository structure

```
A1/
├── index.md                           
├── README.md                          # Project documentation
├── BRIEF.md                           # Assignment brief
├── pattern_generator.py               # Code implementation
└── images/                            # Intermediary and final image outputs
    ├── perlin_moire.png               # Assignment brief image
    └── Figure_1_rgb_channels.png      # Visualization of explicit RGB channels
    └── attractor_grid_pattern.png     # Final generated image
```

---

# Documentation for Assignment 1

## Table of Contents

- [Pseudo-Code](#pseudo-code)
- [Technical Explanation](#technical-explanation)
- [Results](#results)
- [References](#references)

---

## Pseudo-Code

1. **Initialize Canvas and RGB Canvas**
   - Set canvas height and width (square canvas).
   `canvas_width = canvas_height`
   - Create a 2D array of zeros to represent pixel intensities.
   - Create an empty RGB image array (H, W, 3) where each pixel explicitly stores red, green, and blue values using numpy.
   `canvas_rgb`
   - Use this structure to enable direct manipulation of individual color channels.

2. **Define Attractor Points**
   - Create a grid of attractor points across the image domain to establish a base spatial order.
   `num_attracors_per_row`, `num_attractors_per_col` -> `grid_x`, `grid_y`
   - Apply random jitter to attractor positions to break symmetry and introduce organic variation.
   `jitter`
   - Use these attractors as sources that influence the surrounding pixel values.
   `attractors`

3. **Define Distance Function**
   - For each pixel position (x, y), calculate its Euclidean distance to all attractors.
   `distance`
   - Extract distance measures to describe local spatial relationships.
   - Treat these distances as continuous signals rather than discrete values.

4. **Assign Pixel Intensities Based on Attractors**
   - Loop over every pixel (x, y) to coder entire canvas.
   `for x in range(canvas_width)`
   `for y in range(canvas_height)`
   - Normalize pixel coordinates to create resolution independent spatial inputs.
   `nx`, `ny`
   - Generate a distance-driven oscillation field using trigonometric functions to create wave-like patterns.
   `dists`
   - Generate a global gradient field to introduce directional variation across the image.
   `gradient`
   - Generate a structured noise field using smooth trigonometric interference rather than random noise.
   `noise_field`
   - Combine all fields into a single procedural value that can be mapped into RGB space.
   `field`
   - Transform the combined procedural field into red, green, and blue values using phase-shifted sine and cosine functions.
   `r`, `g`, `b`
   - Store the RGB values explicitly in a (H, W, 3) NumPy array.
   `canvas_rgb`

5. **Channel-Specific Variation**
   - Generate independent noise for each RGB channel.
   `noise_rgb`
   - Blend noise with the existing color values to increase visual richness while preserving large-scale structure.
   `canvas_rgb`

6. **Normalize**
   - Normalize each RGB channel to ensure values remain within valid display ranges.

7. **Visualize and Save Image**
   - Use matplotlib.pyplot.imshow() to display normalized canvas.
   - Remove axes for cleaner visualization.
   - Save the generated image to the images/ folder.

---

## Technical Explanation

This program generates a full-color procedural pattern using explicit NumPy RGB array manipulation. The canvas is initialized as a three-channel array with shape (H, W, 3), where each pixel stores red, green, and blue values directly.

A grid of attractor points is created using evenly spaced coordinates and then spatially perturbed using random jitter. These attractors influence the pattern by generating distance fields, that for every pixel, the Euclidean distance to all attractors is computed.

To increase visual complexity, several spatial fields are layered together:
- A distance-based wave field using sine of the minimum distance
- A global spatial gradient using sinusoidal functions of normalized coordinates
- A structured noise field using trigonometric interference patterns

These components are combined into a single procedural field value. This field is then mapped into RGB space using sine and cosine functions, producing smooth, continuous color variation.

Independent noise is added per color channel to introduce texture while preserving global structure. Finally, the image is normalized and displayed directly as an RGB image using matplotlib.

This approach demonstrates explicit RGB channel construction, layered field composition, ans using different NumPy arrays.
---

## Results
![Visualization of explicit RGB channels](images/Figure_1_rgb_channels)
![Final generated image](images/attractor_grid_pattern)

---

## References

- NumPy Documentation: [Array Manipulation Routines](https://numpy.org/doc/stable/reference/routines.array-manipulation.html)
- Matplotlib Documentation: [Colormaps](https://matplotlib.org/stable/contents.html)

---