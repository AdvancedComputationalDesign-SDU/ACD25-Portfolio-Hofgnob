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
                end_u = agent.u + agent.velocity[0]
                end_v = agent.v + agent.velocity[1]
                end = rs.EvaluateSurface(agent.surface, end_u, end_v)
                if end:
                    vectors.append(rs.AddLine(agent.position, end))
            except:
                pass

        # draw trail/curves
        if draw_paths and len(agent.path) > 1:
            try:
                curves.append(rs.AddPolyline(agent.path))
            except:
                pass