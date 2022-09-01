import networkx as nx
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Triangulation import *


def triangulation_edges(triangulation, vertex):
    """Return a list of vertex-id pairs corresponding to the edges in the␣
    ↪triangulation."""
    return [(vertex[i], vertex[j]) for i, j in enumerate(triangulation) if i < j]


def triangulation_triangles(triangulation, vertex):
    """Return a list of vertex-id triples corresponding to the triangles in the␣
    ↪triangulation."""
    return [vertex[i:i + 3] for i in range(0, len(triangulation), 3)]


def plot_triangulation_3d(adj):
    """Display an attempt at embedding the triangulation in 3d."""
    num_vert, vertex = vertex_list(adj)
    edges = triangulation_edges(adj, vertex)
    triangles = triangulation_triangles(adj, vertex)
    # use the networkX 3d graph layout algorithm to find positions for the vertices
    pos = np.array(list(nx.spring_layout(nx.Graph(edges), dim=3).values()))
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    tris = Poly3DCollection(pos[triangles])
    tris.set_edgecolor('k')
    ax.add_collection3d(tris)
    ax.set_xlim3d(np.amin(pos[:, 0]), np.amax(pos[:, 0]))
    ax.set_ylim3d(np.amin(pos[:, 1]), np.amax(pos[:, 1]))
    ax.set_zlim3d(np.amin(pos[:, 2]), np.amax(pos[:, 2]))
    plt.show()