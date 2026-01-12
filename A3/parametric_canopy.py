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

# -------------------------------
# Grasshopper inputs
# -------------------------------
# base_surface        : Rhino Surface or None — optional reference surface
# divU, divV          : int — resolution of UV grid and surface sampling
#
# amplitude           : float — global heightmap scale
# frequency           : float — sinusoidal wave frequency
# phase               : float — phase offset for sinusoidal heightmap
# radial_strength     : float — weight of radial attractor component
# radial_falloff      : float — radial decay controlling central influence
# noise_strength      : float — amplitude of stochastic height variation
#
# n_gon               : int — panel topology (3 = triangles, 4 = quads)
# panel_scale         : float — inset scale for panels (1.0 = no inset)
#
# rec_depth           : int — recursion depth of branching supports
# br_length           : float — initial branch length
# len_reduct          : float — branch length reduction per recursion
# n_branches          : int — number of child branches per node
# angle               : float — branching angle in degrees
# angle_variation     : float — random angular deviation per branch
#
# n_roots             : int — number of support root points
# z_roots             : float — Z-height of support roots
# anchor_radius       : float — minimum spacing between roots
# jitter_strength     : float — positional randomness of root placement
# x_offset, y_offset  : float — directional envelope scaling factors
# seed                : int or None — random seed for reproducibility

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

# -------------------------------
# 1) Heightmap generation
# -------------------------------

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


# -------------------------------
# 5) Uniform sampling + tessellation
# -------------------------------

 # Uniformly resample surface into a point grid
def sample_surface_uniform(surface_id, divU, divV):
    # Uniformly sample surface points using UV grid
    dom_u = rs.SurfaceDomain(surface_id, 0)
    dom_v = rs.SurfaceDomain(surface_id, 1)

    U, V = uv_grid(divU, divV)
    # Sample points from surface using UV parameters
    point_grid = []
    for i in range(divU):
        row = []
        for j in range(divV):
            u = dom_u[0] + U[i, j] * (dom_u[1] - dom_u[0])
            v = dom_v[0] + V[i, j] * (dom_v[1] - dom_v[0])
            # Evaluate surface at UV coordinates
            pt = rs.EvaluateSurface(surface_id, u, v)
            row.append(pt)
        point_grid.append(row)

    return point_grid


 # Generate connected panel tessellation and optional micro-panels
def tessellate_panels_from_grid(point_grid, base_surface, n_gon, panel_scale=1.0):
    # Connected tessellation from grid cells with meaningful n_gon modes
    # n_gon = 3  -> alternating triangle tiling
    # n_gon = 4  -> quad tiling
    # panel_scale insets panels about centroid

    panels = []
    global micro_panels
    micro_panels = []

    # Helper functions for insetting and micro-panel generation
    def inset_poly(pts, s):
        if s >= 1.0:
            return pts
        # Inset polygon pts towards centroid by scale s
        cx = sum(p[0] for p in pts) / float(len(pts))
        cy = sum(p[1] for p in pts) / float(len(pts))
        cz = sum(p[2] for p in pts) / float(len(pts))
        c = (cx, cy, cz)
        out = []
        # Inset each point
        for p in pts:
            v = rs.PointSubtract(p, c)
            v = rs.VectorScale(v, s)
            out.append(rs.PointAdd(c, v))
        return out


    def micro_panels_from_ring(original, inset):
        # Create triangular micro-panels filling the ring between original and inset polygons
        # Assumes same vertex count and ordering
        mps = []
        n = len(original)
        for k in range(n):
            a0 = original[k]
            a1 = original[(k + 1) % n]
            b0 = inset[k]
            b1 = inset[(k + 1) % n]

            # Two triangles forming a quad ring segment
            triA = [a0, a1, b1]
            triB = [a0, b1, b0]

            # Create polylines for micro-panels
            crvA = rs.AddPolyline(triA + [triA[0]])
            crvB = rs.AddPolyline(triB + [triB[0]])
            if crvA: mps.append(crvA)
            if crvB: mps.append(crvB)
        return mps

    # Triangles & Quads
    if n_gon in (3, 4):
        for i in range(len(point_grid) - 1):
            for j in range(len(point_grid[0]) - 1):
                p00 = point_grid[i][j]
                p10 = point_grid[i + 1][j]
                p11 = point_grid[i + 1][j + 1]
                p01 = point_grid[i][j + 1]
                
                # For quads
                if n_gon == 4:
                    orig = [p00, p10, p11, p01]
                    pts = inset_poly(orig, panel_scale)
                    poly = rs.AddPolyline(pts + [pts[0]])
                    if poly: panels.append(poly)

                    if panel_scale < 1.0:
                        micro_panels.extend(micro_panels_from_ring(orig, pts))
                # For triangles
                else:
                    if (i + j) % 2 == 0:
                        tri1 = [p00, p10, p11]
                        tri2 = [p00, p11, p01]
                    else:
                        tri1 = [p00, p10, p01]
                        tri2 = [p10, p11, p01]

                    # Inset triangles
                    orig1 = tri1
                    tri1 = inset_poly(tri1, panel_scale)
                    if panel_scale < 1.0:
                        micro_panels.extend(micro_panels_from_ring(orig1, tri1))
                    # Inset triangles
                    orig2 = tri2
                    tri2 = inset_poly(tri2, panel_scale)
                    if panel_scale < 1.0:
                        micro_panels.extend(micro_panels_from_ring(orig2, tri2))
                    # Create polylines for triangles
                    poly1 = rs.AddPolyline(tri1 + [tri1[0]])
                    poly2 = rs.AddPolyline(tri2 + [tri2[0]])
                    # Add to panels
                    if poly1: panels.append(poly1)
                    if poly2: panels.append(poly2)

        return panels


    # Fallback: quads
    for i in range(len(point_grid) - 1):
        for j in range(len(point_grid[0]) - 1):
            p00 = point_grid[i][j]
            p10 = point_grid[i + 1][j]
            p11 = point_grid[i + 1][j + 1]
            p01 = point_grid[i][j + 1]
            orig = [p00, p10, p11, p01]
            pts = inset_poly(orig, panel_scale)
            poly = rs.AddPolyline(pts + [pts[0]])
            if poly: panels.append(poly)

            if panel_scale < 1.0:
                micro_panels.extend(micro_panels_from_ring(orig, pts))

    return panels



# -------------------------------
# 6) Branching envelope
# -------------------------------

 # Estimate horizontal envelope reach of branching structure
def estimate_branching_envelope(depth, length, reduction, angle):
    # Squared horizontal contributions
    sq = 0.0
    # Current branch length
    curr = length
    # Sum horizontal contributions at each branching level
    for _ in range(depth):
        horiz = curr * np.sin(np.radians(angle))
        sq += horiz * horiz
        curr *= reduction
    return np.sqrt(sq)


# -------------------------------
# 7) Branching supports
# -------------------------------

 # Generate recursive branching support structure
def generate_supports(
    roots,
    target_surface,
    depth,
    length,
    length_reduction,
    n_children,
    angle,
    angle_variation,
    seed=None
):
    # Seed randomness for reproducibility
    seed_everything(seed)
    curves = []
    curves_with_depth = []
    used = set()
    # Recursive branching function
    def branch(pt, direction, curr_len, curr_depth, axis):
        if curr_depth <= 0 or curr_len <= 0:
            return
        # End point calculation
        end_pt = rs.PointAdd(pt, rs.VectorScale(direction, curr_len))

        # Snap only at terminal depth: project to surface and align to panel normal midpoint
        if curr_depth == 1:
            # Find closest UV on surface
            uv = rs.SurfaceClosestPoint(target_surface, end_pt)
            if not uv:
                return

            # Evaluate surface point and normal
            surf_pt = rs.EvaluateSurface(target_surface, uv[0], uv[1])
            normal = rs.SurfaceNormal(target_surface, (uv[0], uv[1]))
            if not normal or rs.VectorLength(normal) < 1e-6:
                return

            normal = rs.VectorUnitize(normal)

            # Optional small offset along normal to emphasize attachment
            snap = rs.PointAdd(surf_pt, rs.VectorScale(normal, 0.0))

            key = (snap.X, snap.Y, snap.Z)
            if key not in used:
                used.add(key)
                crv = rs.AddLine(pt, snap)
                curves.append(crv)
                curves_with_depth.append((crv, curr_depth))
            return
        # Create branch segment
        crv = rs.AddLine(pt, end_pt)
        curves.append(crv)
        curves_with_depth.append((crv, curr_depth))

        # Branch angles
        if n_children == 1:
            angles = [0.0]
        else:
            start = -angle * (n_children - 1) / 2.0
            angles = [start + i * angle for i in range(n_children)]
        # Recursively branch further 
        next_axis = rs.VectorRotate(axis, 90, direction)
        # Generate child branches
        for a in angles:
            a_var = a + random.uniform(-angle_variation, angle_variation)
            new_dir = rs.VectorUnitize(rs.VectorRotate(direction, a_var, axis))
            branch(
                end_pt,
                new_dir,
                curr_len * length_reduction,
                curr_depth - 1,
                next_axis
            )
    # Start branching from each root
    for r in roots:
        branch(r, (0, 0, 1), length, depth, (1, 0, 0))
    # Return branching curves
    return curves, curves_with_depth



### pipeline execution ###
seed_everything(seed)

# 1. UV grid
U, V = uv_grid(divU, divV)

# 2. Heightmap
H = heightmap(U, V, amplitude, frequency, phase, radial_strength, radial_falloff, noise_strength)

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
panels_out = tessellate_panels_from_grid(sampled_pts, surf, n_gon, panel_scale)

# Extract tessellation points from panels for support snapping
tessellation_pts = []
for crv in panels_out:
    pts = rs.CurvePoints(crv)
    if pts:
        tessellation_pts.extend(pts)


# 6. Root placement: tight grid + jitter + envelope + minimum spacing

bbox = rs.BoundingBox(surf)

# Estimate horizontal branching envelope
envelope = estimate_branching_envelope(
    rec_depth,
    br_length,
    len_reduct,
    angle
)

# Directional envelope scaling (acts as reduction factors)
env_x = envelope * max(0.0, x_offset)
env_y = envelope * max(0.0, y_offset)

# Define clamped bounding box for root placement
min_x = min(p[0] for p in bbox) + env_x
max_x = max(p[0] for p in bbox) - env_x
min_y = min(p[1] for p in bbox) + env_y
max_y = max(p[1] for p in bbox) - env_y

if min_x >= max_x or min_y >= max_y:
    raise ValueError("Branching envelope too large for canopy footprint.")

# Clamp jitter strength
jitter = max(0.0, min(1.0, jitter_strength))

# Compute tight grid resolution based on number of roots
grid_nx = int(np.ceil(np.sqrt(n_roots)))
grid_ny = int(np.ceil(float(n_roots) / grid_nx))

dx = (max_x - min_x) / grid_nx
dy = (max_y - min_y) / grid_ny

roots_out = []

for i in range(grid_nx):
    for j in range(grid_ny):
        if len(roots_out) >= n_roots:
            break

        # Grid cell center
        gx = min_x + (i + 0.5) * dx
        gy = min_y + (j + 0.5) * dy

        # Apply small jitter inside cell
        jx = random.uniform(-0.5 * jitter * dx, 0.5 * jitter * dx)
        jy = random.uniform(-0.5 * jitter * dy, 0.5 * jitter * dy)

        candidate = rs.CreatePoint(gx + jx, gy + jy, z_roots)

        # Enforce minimum spacing (anchor_radius)
        valid = True
        for pt in roots_out:
            if rs.Distance(pt, candidate) < anchor_radius:
                valid = False
                break

        if valid:
            roots_out.append(candidate)

# Warn if grid too dense for anchor_radius
if len(roots_out) < n_roots:
    print("Warning: Grid density too high for given anchor_radius; placed",
          len(roots_out), "of", n_roots)

supports_out, supports_with_depth = generate_supports(
    roots_out,
    surf,
    depth=rec_depth,
    length=br_length,
    length_reduction=len_reduct,
    n_children=n_branches,
    angle=angle,
    angle_variation=angle_variation,
    seed=seed
)

# Normalize branching depth for GH visualization (0–1)
support_depth_values = []
for crv, d in supports_with_depth:
    support_depth_values.append(float(d) / max(1.0, rec_depth))

# -------------------------------
# Grasshopper outputs
# -------------------------------
surface = surf                                  # NURBS Surface — final canopy surface geometry
points = [pt for row in pts_def for pt in row]  # list[Point3d] — deformed canopy point grid (flattened)
height = H.flatten().tolist()                   # list[float] — heightmap scalar values per grid point
panels = panels_out                             # list[Curve] — primary canopy panel polylines
micro_panels = micro_panels                     # list[Curve] — secondary inset micro-panel geometry
supports = supports_out                         # list[Curve] — recursive branching support curves
roots = roots_out                               # list[Point3d] — support root anchor points
tessellation_points = tessellation_pts          # list[Point3d] — panel corner points for snapping/reference
support_depth = support_depth_values            # list[float] — normalized recursion depth per support (0–1)