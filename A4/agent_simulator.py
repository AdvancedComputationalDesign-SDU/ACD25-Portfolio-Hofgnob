"""
Assignment 4: Agent-Based Model for Surface Panelization
Author: Jesper Christensen Sørensen

Agent Simulator

Description:
This component simulates the behavior of agents on a surface over time.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import rhinoscriptsyntax as rs

# -----------------------------------------------------------------------------
# Inputs (Grasshopper)
# -----------------------------------------------------------------------------
# agent_component  (reference to agent_builder component)
# draw_vectors     (bool)
# draw_paths       (bool)

# -----------------------------------------------------------------------------
# Retrieve agents from builder component
# -----------------------------------------------------------------------------

agents = agent_component if agent_component is not None else None
# -----------------------------------------------------------------------------
# Step simulation
# -----------------------------------------------------------------------------
if agents is not None:
    for agent in agents:
        agent.update(agents)

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
points = []  # agent positions
vectors = []  # velocity vectors
curves = []  # trails/curves

if agents is not None:
    for agent in agents:

        # draw position
        if agent.position is not None:
            points.append(rs.AddPoint(agent.position))

        # draw velocity vector on surface
        if draw_vectors:
            try:
                # real surface domains
                dom_u = rs.SurfaceDomain(agent.surface, 0)
                dom_v = rs.SurfaceDomain(agent.surface, 1)

                # start at the agent's true 3D position (already evaluated on surface)
                start = agent.position

                # take a small step in normalized UV along the velocity direction
                u2 = agent.u + agent.velocity[0]
                v2 = agent.v + agent.velocity[1]

                # clamp to normalized domain
                u2 = max(0.0, min(1.0, u2))
                v2 = max(0.0, min(1.0, v2))

                # map normalized UV to real surface UV
                u2_real = dom_u[0] + u2 * (dom_u[1] - dom_u[0])
                v2_real = dom_v[0] + v2 * (dom_v[1] - dom_v[0])

                # evaluate end point on surface
                end = rs.EvaluateSurface(agent.surface, u2_real, v2_real)

                if start and end:
                    vectors.append(rs.AddLine(start, end))
            except:
                pass

        # draw trail/curves
        if draw_paths and len(agent.path) > 1:
            try:
                curves.append(rs.AddPolyline(agent.path))
            except:
                pass