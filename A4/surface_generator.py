"""
Assignment 4: Agent-Based Model for Surface Panelization

Author: Jesper Christensen Sørensen
"""

# -----------------------------------------------------------------------------
# Imports (extend as needed)
# -----------------------------------------------------------------------------
import rhinoscriptsyntax as rs
import numpy as np
import random

# Outputs
surface = None          # Rhino Surface
points = []             # list of 3D points
height = []             # flattened height field
fields = {}             # dictionary of geometric fields


# Set random seeds for reproducible results
def seed_everything(seed):
    if seed is None:
        return
    try:
        random.seed(seed)
        np.random.seed(seed)
    except Exception as e:
        raise RuntimeError(f"Failed to set random seeds: {e}")
    
# Generate normalized UV parameter grid
def uv_grid(divU, divV):
    # Create uniform UV samples in [0,1]x[0,1] for surface parametrization
    # UV space standardizes sampling over any surface domain
    u = np.linspace(0.0, 1.0, divU)
    v = np.linspace(0.0, 1.0, divV)

    U, V = np.meshgrid(u, v, indexing="ij")

    return U, V

# -----------------------------------------------------------------------------
# 1. Heightmap Generation
# -----------------------------------------------------------------------------
# Generate scalar height field from sinusoidal + radial + noise components
def heightmap(
    U, V,
    amplitude=1.0,
    frequency=1.0,
    phase=0.0,
    radial_strength=0.3,
    radial_falloff=4.0,
    anisotropy=1.0,
    noise_strength=0.0
):
    # Base sinusoidal field (global wave structure)
    H_wave = (
        np.sin(2.0 * np.pi * frequency * U + phase) *
        np.cos(2.0 * np.pi * frequency * anisotropy * V + phase)
    )

    # Radial attractor (creates central lift or depression)
    cx, cy = 0.5, 0.5
    r = np.sqrt((U - cx) ** 2 + (V - cy) ** 2)
    H_radial = np.exp(-radial_falloff * r)

    # Noise field (for organic variation)
    if noise_strength > 0.0:
        noise = np.random.rand(*U.shape) - 0.5
    else:
        noise = 0.0

    # Weighted combination of fields
    H = (
        (1.0 - radial_strength) * H_wave +
        radial_strength * H_radial +
        noise_strength * noise
    )
    return amplitude * H

# -----------------------------------------------------------------------------
# Field generation
# -----------------------------------------------------------------------------

def compute_height_field(H):
    """Return height scalar field."""
    return H


def compute_slope_field(H, U, V):
    """
    Compute slope (gradient) field from heightmap.
    Returns a vector field (du, dv) at each UV sample.
    """
    du = np.gradient(H, axis=0)
    dv = np.gradient(H, axis=1)
    return du, dv

# -------------------------------
# 2) Source point grid (planar or surface-sampled)
# -------------------------------

# Generate planar XY grid of points
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


 # Sample point grid directly from a NURBS surface
def sample_point_grid_from_surface(base_surface, U, V):
    # Sample points from surface using UV parameters
    # Maps normalized UV to actual surface domain for accurate sampling
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

 # Deform point grid vertically using heightmap
def manipulate_points_z(point_grid, H):
    divU = len(point_grid)
    divV = len(point_grid[0])

    new_grid = []
    for i in range(divU):
        row = []
        for j in range(divV):
            x, y, z = point_grid[i][j]
            # Offset point vertically by heightmap value for simple deformation
            row.append(rs.CreatePoint(x, y, z + H[i, j]))
        new_grid.append(row)

    return new_grid


 # Deform point grid along surface normals using heightmap
def manipulate_points_along_normals(point_grid, H, base_surface, U, V):
    # Offset points along surface normals scaled by heightmap
    # Creates deformation respecting surface curvature and orientation
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
                # If normal invalid or zero length, keep original point
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

 # Construct NURBS surface from point grid
def surface_from_point_grid(point_grid):
    # Create NURBS surface from point grid points
    rows = len(point_grid)
    if rows == 0:
        return None
    cols = len(point_grid[0])

    flat_points = []
    for i in range(rows):
        for j in range(cols):
            flat_points.append(point_grid[i][j])

    # Rhino expects points in (u,v) order, cols x rows grid
    srf = rs.AddSrfPtGrid((cols, rows), flat_points)

    return srf


### pipeline execution ###
seed_everything(seed)

# 1. UV grid
U, V = uv_grid(divU, divV)

# 2. Heightmap
H = heightmap(U, V, amplitude, frequency, phase, radial_strength, radial_falloff, noise_strength)

# -----------------------------------------------------------------------------
# Field construction
# -----------------------------------------------------------------------------

height_field = compute_height_field(H)
slope_u, slope_v = compute_slope_field(H, U, V)

fields = {
    "height": height_field,
    "slope_u": slope_u,
    "slope_v": slope_v,
    "U": U,
    "V": V
}

# 3. Point grid (planar or surface sampled)
if base_surface is not None:
    pts = sample_point_grid_from_surface(base_surface, U, V)
    pts_def = manipulate_points_along_normals(pts, H, base_surface, U, V)
else:
    pts = make_point_grid_xy(divU, divV)
    pts_def = manipulate_points_z(pts, H)

# 4. Surface construction from deformed points
surf = surface_from_point_grid(pts_def)

# Return outputs
surface = surf
points = [pt for row in pts_def for pt in row]
height = H.flatten().tolist()
fields = fields