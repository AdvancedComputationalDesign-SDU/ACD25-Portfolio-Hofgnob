"""
Assignment 4: Agent-Based Model for Surface Panelization

Author: Jesper Christensen Sørensen

Agent Builder
"""
# -----------------------------------------------------------------------------
# Imports (extend as needed)
# -----------------------------------------------------------------------------

import rhinoscriptsyntax as rs
import random
import numpy as np
import scriptcontext as sc
import Grasshopper

# -----------------------------------------------------------------------------
# Utility functions (optional)
# -----------------------------------------------------------------------------
def seed_everything(seed):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

seed_everything(42)

### Inputs ###
# surface (surface)
# divU (int)
# divV (int)
# num_agents (int)
# max_age (int)
# reset (bool)

### Outputs ###
agents = []

# -----------------------------------------------------------------------------
# Field extraction helper (compute height, slope, and UV grids from the surface)
# -----------------------------------------------------------------------------

def compute_fields_from_surface(surface, divU, divV):
    # Compute a curvature-based flow field from a surface.
    # Returns two direction fields (flow_u, flow_v) in normalized UV space.

    divU = int(divU)
    divV = int(divV)

    # Coerce surface and get domains
    surf = rs.coercesurface(surface)
    dom_u = rs.SurfaceDomain(surf, 0)
    dom_v = rs.SurfaceDomain(surf, 1)

    # Normalized UV grids in [0,1]
    u_lin = np.linspace(0.0, 1.0, divU)
    v_lin = np.linspace(0.0, 1.0, divV)
    U, V = np.meshgrid(u_lin, v_lin, indexing="ij")

    # Build a direction field from surface normals (projected to UV plane)
    flow_u = np.zeros((divU, divV), dtype=float)
    flow_v = np.zeros((divU, divV), dtype=float)

    for i in range(divU):
        for j in range(divV):
            u = dom_u[0] + U[i, j] * (dom_u[1] - dom_u[0])
            v = dom_v[0] + V[i, j] * (dom_v[1] - dom_v[0])

            # Evaluate surface and normal
            pt = rs.EvaluateSurface(surf, u, v)
            if pt is None:
                continue

            normal = rs.SurfaceNormal(surf, (u, v))
            if normal is None:
                continue

            # Project normal onto XY plane to get a tangential flow direction
            nx, ny, nz = normal
            d = np.array([-nx, -ny], dtype=float)

            # Normalize direction
            length = np.linalg.norm(d)
            if length > 1e-6:
                d /= length

            flow_u[i, j] = d[0]
            flow_v[i, j] = d[1]

    return flow_u, flow_v

# -----------------------------------------------------------------------------
# Core agent class
# -----------------------------------------------------------------------------
class Agent(object):
# Agent moving on a surface influenced by environmental fields
    def __init__(self, surface, u, v, H, slope_u, slope_v, max_age):
        # Surface attributes
        self.surface = rs.coercesurface(surface)
        self.dom_u = rs.SurfaceDomain(self.surface, 0)
        self.dom_v = rs.SurfaceDomain(self.surface, 1)
        self.u = u
        self.v = v
        # Cached UV indices for field sampling
        self.u_idx = 0
        self.v_idx = 0
        # Environmental flow field (curvature-based)
        self.flow_u = H
        self.flow_v = slope_u

        # Shared coverage field
        self.coverage = slope_v

        # Position on surface (map normalized UV to real surface domain)
        u_real = self.dom_u[0] + self.u * (self.dom_u[1] - self.dom_u[0])
        v_real = self.dom_v[0] + self.v * (self.dom_v[1] - self.dom_v[0])
        self.position = rs.EvaluateSurface(self.surface, u_real, v_real)
        self.velocity = [
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01)
        ]
        # State attributes
        self.id = id(self)  # Unique identifier
        self.age = 0  # Initialize age
        self.max_age = max_age # Maximum age
        self.alive = True # Alive status
        # Path history
        self.path = [self.position]

    def sense(self):
        # Read curvature flow direction
        if self.flow_u is None or self.flow_v is None:
            return (0.0, 0.0)

        du = self.flow_u[self.u_idx, self.v_idx]
        dv = self.flow_v[self.u_idx, self.v_idx]

        # Push away from visited areas using coverage field
        if self.coverage is not None:
            rows, cols = self.coverage.shape
            u = self.u_idx
            v = self.v_idx

            # Sample local gradient of coverage field
            u_prev = max(0, u - 1)
            u_next = min(rows - 1, u + 1)
            v_prev = max(0, v - 1)
            v_next = min(cols - 1, v + 1)

            dc_u = self.coverage[u_next, v] - self.coverage[u_prev, v]
            dc_v = self.coverage[u, v_next] - self.coverage[u, v_prev]

            # Repel from high-density regions
            du -= dc_u * 0.05
            dv -= dc_v * 0.05

        return (du, dv)

    def decide(self, flow):
        du, dv = flow

        # Integration parameters
        damping = 0.9
        strength = 0.02
        max_speed = 0.03

        # Integrate velocity with damping
        self.velocity[0] = self.velocity[0] * damping + du * strength
        self.velocity[1] = self.velocity[1] * damping + dv * strength

        # Set speed limit
        speed = (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5
        if speed > max_speed:
            self.velocity[0] = self.velocity[0] / speed * max_speed
            self.velocity[1] = self.velocity[1] / speed * max_speed

        return self.velocity

    def move(self):
        # Move in normalized UV space
        self.u += self.velocity[0]
        self.v += self.velocity[1]

        # Bounce on U boundaries
        if self.u < 0.0:
            self.u = 0.0
            self.velocity[0] *= -1.0
        elif self.u > 1.0:
            self.u = 1.0
            self.velocity[0] *= -1.0

        # Bounce on V boundaries
        if self.v < 0.0:
            self.v = 0.0
            self.velocity[1] *= -1.0
        elif self.v > 1.0:
            self.v = 1.0
            self.velocity[1] *= -1.0

        # Update field indices
        self.update_field_indices()
        # Map normalized UV to real surface domain
        u_real = self.dom_u[0] + self.u * (self.dom_u[1] - self.dom_u[0])
        v_real = self.dom_v[0] + self.v * (self.dom_v[1] - self.dom_v[0])

        # Project back to surface
        self.position = rs.EvaluateSurface(self.surface, u_real, v_real)

        # Store path history
        self.path.append(self.position)

        return self.position

    def update_field_indices(self):
        # Map continuous UV [0,1] to discrete field indices
        if self.flow_u is None:
            return
        rows, cols = self.flow_u.shape
        self.u_idx = int(max(0, min(rows - 1, round(self.u * (rows - 1)))))
        self.v_idx = int(max(0, min(cols - 1, round(self.v * (cols - 1)))))

        # Deposit pheromone at current cell
        if self.coverage is not None:
            self.coverage[self.u_idx, self.v_idx] += 1.0

    def update(self, agents):
        # Update agent state
        if not self.alive:
            return
        # senses environment
        flow = self.sense()
        # decides on action based on sensed flow
        self.decide(flow)
        # moves according to decision
        self.move()
        # ages the agent
        self.age += 1
        # checks if agent is still alive
        if self.age >= self.max_age:
            self.alive = False
        return self

# -----------------------------------------------------------------------------
# Factory for creating agents
# -----------------------------------------------------------------------------
def build_agents(surface, divU, divV, num_agents, max_age):
    flow_u, flow_v = compute_fields_from_surface(surface, divU, divV)
    if flow_u is None or flow_v is None:
        return []

    # Coverage (pheromone) field: tracks visited density
    coverage = np.zeros_like(flow_u, dtype=float)

    agents = []
    for _ in range(int(num_agents)):
        u = random.uniform(0.0, 1.0)
        v = random.uniform(0.0, 1.0)
        agent = Agent(surface, u, v, flow_u, flow_v, coverage, max_age)
        # initialize field indices for first sense()
        agent.update_field_indices()
        agents.append(agent)
    return agents

# -----------------------------------------------------------------------------
# Grasshopper persistence (store agents on the component between recomputes)
# -----------------------------------------------------------------------------

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, surface, divU: int, divV: int, num_agents: int, max_age: int, reset: bool):
        if reset or not hasattr(self, "agents"):
            self.agents = build_agents(surface, divU, divV, num_agents, max_age)
        
        # Update all agents
        for agent in self.agents:
            agent.update(self.agents)
        
        return self.agents
