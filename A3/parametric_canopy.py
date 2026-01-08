"""
Assignment 3: Parametric Structural Canopy

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
# snap_tol: Number

# Outputs
surface = [] # Rhino Surface
points = [] # list of list of 3D points
height = [] # 2D array

def seed_everything(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def uv_grid(divU, divV):
    # Create uniform UV samples in [0,1]x[0,1] for surface parametrization
    # UV space standardizes sampling over any surface domain
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
# 1) Heightmap (sinusoidal + radial attractor)
# -------------------------------

def heightmap(U, V, amplitude=1.0, frequency=1.0, phase=0.0):
    # Generate combined sinusoidal and radial height field
    # Sinusoidal adds wave pattern, attractor creates central elevation focus
    H = (
        np.sin(2.0 * np.pi * frequency * U + phase) *
        np.cos(2.0 * np.pi * frequency * V + phase)
    )
    cx, cy = 0.5, 0.5
    r = np.sqrt((U - cx) ** 2 + (V - cy) ** 2)
    attractor = np.exp(-4.0 * r)

    H = amplitude * (0.7 * H + 0.3 * attractor)

    return H


# -------------------------------
# 2) Source point grid (planar or surface-sampled)
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


# -------------------------------
# 5) Uniform sampling + tessellation
# -------------------------------

def sample_surface_uniform(surface_id, divU, divV):
    # Uniformly sample surface points using UV grid
    dom_u = rs.SurfaceDomain(surface_id, 0)
    dom_v = rs.SurfaceDomain(surface_id, 1)

    U, V = uv_grid(divU, divV)

    point_grid = []
    for i in range(divU):
        row = []
        for j in range(divV):
            u = dom_u[0] + U[i, j] * (dom_u[1] - dom_u[0])
            v = dom_v[0] + V[i, j] * (dom_v[1] - dom_v[0])

            pt = rs.EvaluateSurface(surface_id, u, v)
            row.append(pt)
        point_grid.append(row)

    return point_grid


def tessellate_panels_from_grid(point_grid):
    # Create triangular panels from quad grid cells
    # Triangulation ensures planar panels for structural analysis
    panels = []

    rows = len(point_grid)
    cols = len(point_grid[0])

    for i in range(rows - 1):
        for j in range(cols - 1):
            a = point_grid[i][j]
            b = point_grid[i + 1][j]
            c = point_grid[i + 1][j + 1]
            d = point_grid[i][j + 1]

            srf1 = rs.AddSrfPt([a, b, c])
            srf2 = rs.AddSrfPt([a, c, d])

            if srf1: panels.append(srf1)
            if srf2: panels.append(srf2)

    return panels

# -------------------------------
# 6) Branching supports with snapping and recursion control
# -------------------------------

def generate_supports(
    roots,
    surface,
    tessellation_points,
    snap_tol=1.0,
    depth=3,
    length=6.0,
    length_reduction=0.7,
    n_children=3,
    angle=40.0,
    angle_variation=5.0,
    seed=None
):
    seed_everything(seed)
    curves = []
    used_points = set()

    def branch(pt, direction, curr_len, curr_depth, axis):
        # Recursion stop on depth or length limits to prevent infinite branching
        if curr_depth <= 0 or curr_len <= 0:
            return

        end_pt = rs.PointAdd(pt, rs.VectorScale(direction, curr_len))

        # Snap to nearest unused tessellation point within tolerance for structural connection
        surf_pt = rs.SurfaceClosestPoint(surface, end_pt)
        if surf_pt:
            closest_on_surf = rs.EvaluateSurface(surface, surf_pt[0], surf_pt[1])
            if rs.Distance(end_pt, closest_on_surf) <= snap_tol:

                # Filter tessellation points not yet used for snapping
                candidates = [
                    p for p in tessellation_points
                    if (p.X, p.Y, p.Z) not in used_points
                ]

                if candidates:
                    # Reserve the snapped point to avoid multiple branches snapping to same point
                    snap_pt = min(candidates, key=lambda p: rs.Distance(p, end_pt))
                    used_points.add((snap_pt.X, snap_pt.Y, snap_pt.Z))
                    crv = rs.AddLine(pt, snap_pt)
                    if crv:
                        curves.append(crv)
                return

        # Draw branch segment if no snapping occurred; continues branch growth
        crv = rs.AddLine(pt, end_pt)
        if crv:
            curves.append(crv)

        # Symmetrical branching with fixed angle steps for balanced structure
        if n_children == 1:
            angles = [0.0]
        else:
            step = angle
            start = -angle * (n_children - 1) / 2.0
            angles = [start + i * step for i in range(n_children)]

        # Rotate axis by 90 degrees around current direction to define new branching plane
        next_axis = rs.VectorRotate(axis, 90.0, direction)

        for a in angles:
            varied_angle = a + random.uniform(-angle_variation, angle_variation)
            new_dir = rs.VectorRotate(direction, varied_angle, axis)
            new_dir = rs.VectorUnitize(new_dir)

            branch(
                end_pt,
                new_dir,
                curr_len * length_reduction,
                curr_depth - 1,
                next_axis
            )

    # Start branches vertically from roots to simulate natural upward growth
    for r in roots:
        branch(r, (0, 0, 1), length, depth, (1, 0, 0))

    return curves


### pipeline execution ###
seed_everything(seed)

# 1. UV grid
U, V = uv_grid(divU, divV)

# 2. Heightmap
H = heightmap(U, V, amplitude, frequency, phase)

# 3. Point grid (planar or surface sampled)
if base_surface is not None:
    pts = sample_point_grid_from_surface(base_surface, U, V)
    pts_def = manipulate_points_along_normals(pts, H, base_surface, U, V)
else:
    pts = make_point_grid_xy(divU, divV)
    pts_def = manipulate_points_z(pts, H)

# 4. Surface construction from deformed points
surf = surface_from_point_grid(pts_def)

# 5. Uniform sampling + tessellation
sampled_pts = sample_surface_uniform(surf, divU, divV)
panels_out = tessellate_panels_from_grid(sampled_pts)

# Extract tessellation points from panels for support snapping
tessellation_pts = []

for srf in panels_out:
    pts = rs.SurfacePoints(srf)
    if pts:
        tessellation_pts.extend(pts)

# 6. Branching supports with random anchors and minimum spacing
roots_out = []
bbox = rs.BoundingBox(surf)
min_x = min(p[0] for p in bbox) + x_offset
max_x = max(p[0] for p in bbox) - x_offset
min_y = min(p[1] for p in bbox) + y_offset
max_y = max(p[1] for p in bbox) - y_offset

max_attempts = 1000
attempts = 0

while len(roots_out) < n_roots and attempts < max_attempts:
    attempts += 1

    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    candidate = rs.CreatePoint(x, y, anchor_z)

    # Check candidate against existing roots to enforce minimum spacing
    valid = True
    for pt in roots_out:
        # Enforce minimum distance between anchors (anchor_radius)
        if rs.Distance(pt, candidate) < anchor_radius:
            # Reject candidate if too close to existing root
            valid = False
            break

    if valid:
        roots_out.append(candidate)

if len(roots_out) < n_roots:
    # Warn if unable to place all roots due to spacing constraints
    print("Warning: Could not place all roots with the given anchor_radius.")

supports_out = generate_supports(
    roots_out,
    surf,
    tessellation_pts,
    snap_tol=snap_tol,
    depth=rec_depth,
    length=br_length,
    length_reduction=len_reduct,
    n_children=n_branches,
    angle=angle,
    angle_variation=angle_variation,
    seed=seed
)

### outputs ###
surface = surf
points = [pt for row in pts_def for pt in row] 
height = H.flatten().tolist()
panels = panels_out
supports = supports_out
roots = roots_out
tessellation_points = tessellation_pts
