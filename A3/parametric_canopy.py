"""
Assignment 3: Parametric Structural Canopy — Pseudocode Scaffold

Author: Jesper Christensen Sørensen

"""

#r: numpy
import numpy as np
import rhinoscriptsyntax as rs
import random

# -------------------------------
# Helpers
# -------------------------------
# Inputs
# base_surface: Rhino Surface | None
# divU: int
# divV: int
# amplitude: int
# frequency: float
# phase: float

# Outputs
surface = [] #Rhino Surface
points = [] #list of list of 3D points
height = [] #2D array

def uv_grid(divU, divV):
    # Create uniform UV samples in [0,1]x[0,1].
    u = np.linspace(0.0, 1.0, divU)
    v = np.linspace(0.0, 1.0, divV)

    U, V = np.meshgrid(u, v, indexing="ij")

    return U, V


def bbox_corners(geo):
    # Get bounding box corners (returns 8 points)
    bbox = rs.BoundingBox(geo)
    if not bbox or len(bbox) < 8:
        raise ValueError("Invalid geometry or bounding box could not be computed")

    # Choose four ground-level corners as support anchors
    anchors = [bbox[0], bbox[1], bbox[2], bbox[3]]

    return anchors


# -------------------------------
# 1) Heightmap (placeholder)
# -------------------------------

def heightmap(U, V, amplitude=1.0, frequency=1.0, phase=0.0):
    # Base sinusoidal landscape
    H = (
        np.sin(2.0 * np.pi * frequency * U + phase) *
        np.cos(2.0 * np.pi * frequency * V + phase)
    )
    # Radial attractor centered in UV space
    cx, cy = 0.5, 0.5
    r = np.sqrt((U - cx) ** 2 + (V - cy) ** 2)
    attractor = np.exp(-4.0 * r)

    # Combine fields and scale
    H = amplitude * (0.7 * H + 0.3 * attractor)

    return H


# -------------------------------
# 2) Source point grid (planar OR sampled from surface)
# -------------------------------

def make_point_grid_xy(divU, divV, origin=(0.0, 0.0, 0.0), size=(100.0, 100.0)):
    ox, oy, oz = origin
    sx, sy = size

    U, V = uv_grid(divU, divV)

    point_grid = []
    for i in range(divU):
        row = []
        for j in range(divV):
            x = ox + U[i, j] * sx
            y = oy + V[i, j] * sy
            z = oz
            row.append(rs.CreatePoint(x, y, z))
        point_grid.append(row)

    return point_grid


def sample_point_grid_from_surface(base_surface, U, V):
    dom_u = rs.SurfaceDomain(base_surface, 0)
    dom_v = rs.SurfaceDomain(base_surface, 1)

    divU, divV = U.shape
    point_grid = []

    for i in range(divU):
        row = []
        for j in range(divV):
            u = dom_u[0] + U[i, j] * (dom_u[1] - dom_u[0])
            v = dom_v[0] + V[i, j] * (dom_v[1] - dom_v[0])
            pt = rs.EvaluateSurface(base_surface, u, v)
            row.append(pt)
        point_grid.append(row)

    return point_grid


# -------------------------------
# 3) Deform point grid (Z or surface normals)
# -------------------------------

def manipulate_points_z(point_grid, H):
    divU = len(point_grid)
    divV = len(point_grid[0])

    new_grid = []
    for i in range(divU):
        row = []
        for j in range(divV):
            x, y, z = point_grid[i][j]
            row.append(rs.CreatePoint(x, y, z + H[i, j]))
        new_grid.append(row)

    return new_grid


def manipulate_points_along_normals(point_grid, H, base_surface, U, V):
    dom_u = rs.SurfaceDomain(base_surface, 0)
    dom_v = rs.SurfaceDomain(base_surface, 1)

    divU, divV = U.shape
    new_grid = []

    for i in range(divU):
        row = []
        for j in range(divV):
            u = dom_u[0] + U[i, j] * (dom_u[1] - dom_u[0])
            v = dom_v[0] + V[i, j] * (dom_v[1] - dom_v[0])

            p = point_grid[i][j]
            n = rs.SurfaceNormal(base_surface, (u, v))

            if not n or rs.VectorLength(n) < 1e-6:
                row.append(p)
            else:
                n_unit = rs.VectorUnitize(n)
                offset = rs.VectorScale(n_unit, H[i, j])
                row.append(rs.PointAdd(p, offset))

        new_grid.append(row)

    return new_grid

# -------------------------------
# 4) Construct canopy surface from points
# -------------------------------

def surface_from_point_grid(point_grid):
    # Determine grid dimensions
    rows = len(point_grid)
    if rows == 0:
        return None
    cols = len(point_grid[0])

    # Flatten point grid row-wise (U-major order)
    flat_points = []
    for i in range(rows):
        for j in range(cols):
            flat_points.append(point_grid[i][j])

    # Create NURBS surface from point grid
    srf = rs.AddSrfPtGrid((cols, rows), flat_points)

    return srf


### pipeline execution ###

# 1. UV grid
U, V = uv_grid(divU, divV)

# 2. Heightmap
H = heightmap(U, V, amplitude, frequency, phase)

# 3. Point grid (choose ONE of the following)

if base_surface is not None:
    pts = sample_point_grid_from_surface(base_surface, U, V)
    pts_def = manipulate_points_along_normals(pts, H, base_surface, U, V)
else:
    pts = make_point_grid_xy(divU, divV)
    pts_def = manipulate_points_z(pts, H)

# 4. Surface construction
surf = surface_from_point_grid(pts_def)

# --- outputs ---
surface = surf
# Flatten the nested list so Grasshopper sees a single list of points
points = [pt for row in pts_def for pt in row] 
height = H.flatten().tolist()
