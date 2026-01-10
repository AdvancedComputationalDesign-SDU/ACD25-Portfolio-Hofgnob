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
# reset (bool)

### Outputs ###
agents = []

# -----------------------------------------------------------------------------
# Field extraction helper (compute height, slope, and UV grids from the surface)
# -----------------------------------------------------------------------------

def compute_fields_from_surface(surface, divU, divV):
    # Compute height field (Z) and slope (gradient in UV) from a surface.
    if surface is None or divU is None or divV is None:
        return None, None, None

    try:
        divU = int(divU)
        divV = int(divV)
        if divU < 2 or divV < 2:
            return None, None, None

        # Surface domains
        dom_u = rs.SurfaceDomain(surface, 0)
        dom_v = rs.SurfaceDomain(surface, 1)

        # Normalized UV grids in [0,1]
        u_lin = np.linspace(0.0, 1.0, divU)
        v_lin = np.linspace(0.0, 1.0, divV)
        U, V = np.meshgrid(u_lin, v_lin, indexing="ij")

        # Sample surface and build height field (use Z as height)
        H = np.zeros((divU, divV), dtype=float)
        for i in range(divU):
            for j in range(divV):
                u = dom_u[0] + U[i, j] * (dom_u[1] - dom_u[0])
                v = dom_v[0] + V[i, j] * (dom_v[1] - dom_v[0])
                pt = rs.EvaluateSurface(surface, u, v)
                if pt is None:
                    return None, None, None
                H[i, j] = pt[2]

        # Compute slope (gradient) in UV space
        slope_u = np.gradient(H, axis=0)
        slope_v = np.gradient(H, axis=1)

        return H, slope_u, slope_v
    except Exception:
        return None, None, None

# -----------------------------------------------------------------------------
# Core agent class
# -----------------------------------------------------------------------------
class Agent(object):
# Agent moving on a surface influenced by environmental fields
    def __init__(self, surface, u, v, H, slope_u, slope_v, max_age=100):
        # Surface attributes
        self.surface = rs.coercesurface(surface)
        self.u = u
        self.v = v
        # Cached UV indices for field sampling
        self.u_idx = 0
        self.v_idx = 0
        # Environmental fields
        self.H = H
        self.slope_u = slope_u
        self.slope_v = slope_v
        # Position on surface
        self.position = rs.EvaluateSurface(self.surface, self.u, self.v)
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
        # read height and slope at current position (grid sample)
        if self.H is None or self.slope_u is None or self.slope_v is None:
            return None, (0.0, 0.0)
        h = self.H[self.v_idx, self.u_idx]
        slope = (self.slope_u[self.v_idx, self.u_idx], self.slope_v[self.v_idx, self.u_idx])
        return h, slope

    def decide(self, h, slope):
        du, dv = slope
        # move downhill along slope
        self.velocity[0] += du * 0.5
        self.velocity[1] += dv * 0.5
        return self.velocity

    def move(self):
        # Move in UV space
        self.u += self.velocity[0]
        self.v += self.velocity[1]
        # Keep agents within surface
        self.u = max(0.0, min(1.0, self.u))
        self.v = max(0.0, min(1.0, self.v))
        # Update field indices
        self.update_field_indices()
        # Project back to surface
        self.position = rs.EvaluateSurface(self.surface, self.u, self.v)
        # Store path history
        self.path.append(self.position)
        return self.position

    def update_field_indices(self):
        # Map continuous UV [0,1] to discrete field indices
        if self.H is None:
            return
        rows, cols = self.H.shape
        self.u_idx = int(max(0, min(cols - 1, round(self.u * (cols - 1)))))
        self.v_idx = int(max(0, min(rows - 1, round(self.v * (rows - 1)))))

    def update(self, agents):
        # Update agent state
        if not self.alive:
            return
        # senses environment
        h, slope = self.sense()
        # decides on action based on sensed data
        self.decide(h, slope)
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
def build_agents(surface, divU, divV, num_agents):
    H, slope_u, slope_v = compute_fields_from_surface(surface, divU, divV)
    if H is None or slope_u is None or slope_v is None:
        return []
    agents = []
    for _ in range(int(num_agents)):
        u = random.uniform(0.0, 1.0)
        v = random.uniform(0.0, 1.0)
        agent = Agent(surface, u, v, H, slope_u, slope_v)
        # initialize field indices for first sense()
        agent.update_field_indices()
        agents.append(agent)
    return agents

# -----------------------------------------------------------------------------
# Grasshopper persistence (store agents on the component between recomputes)
# -----------------------------------------------------------------------------

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, surface, divU: int, divV: int, num_agents: int, reset: bool):
        if reset or not hasattr(self, "agents"):
            self.agents = build_agents(surface, divU, divV, num_agents)
        
        # Update all agents
        for agent in self.agents:
            agent.update(self.agents)
        
        return self.agents
