from __future__ import print_function
from __future__ import division

import sys

import numpy as np
import matplotlib.pyplot as plt

import cvxpy as cvx

import hypergraph
import hypergraph.core
from   hypergraph.core import Graph, Edge
from   hypergraph.convert.nx import networkx_export

import networkx

from scipy.stats import rankdata


def _bbox_default(bbox=None):
    # Default to placing on 11x11 area
    return np.array([10, 10]) if bbox is None else bbox

def random_circuit(n_vertices=100, seed=None, directed=True):
    rand = np.random.RandomState(seed)

    # Create the graph
    vertices = np.arange(n_vertices)
    graph = Graph(vertices=vertices, directed=directed)

    # Starting from the root vertex, create a random "circuit like" graph
    for i, v in enumerate(vertices):
        # Randomly sample vertices
        if i < n_vertices - 1:
            sample = rand.choice(vertices[i+1:], rand.randint(1, 4))
        else:
            sample = rand.choice(vertices[:i], 2)

        # Add the edge to the graph
        edge = Edge(np.hstack([v, sample]), head=v if directed else None)
        graph.add_edge(edge)

    return graph

def random_placement(graph, bbox=None, fixed_placement={}, seed=None, best_of=1):
    bbox = _bbox_default(bbox)
    rand = np.random.RandomState(seed)

    cost = float('inf')
    placement = None
    for _ in range(best_of):
        # The placement is a matrix with rows representing x, y of node
        new_placement = rand.random_sample((len(graph.vertices), 2)) * bbox

        # Fix the positions for the fixed nodes
        for v, pos in fixed_placement.viewitems():
            new_placement[v] = pos

        # check if we want to switch
        new_cost = cost_HPWL(graph, new_placement)
        if new_cost < cost:
            cost = new_cost
            placement = new_placement

    return placement

def random_fixed(graph, bbox=None, n_fixed=4, seed=None):
    """Randomly create a fixed placement for some nodes in the graph"""
    bbox = _bbox_default(bbox)
    rand = np.random.RandomState(seed)

    assert n_fixed >= 2, "Number of fixed nodes must be >= 2"

    # Number of nodes to choose from
    N = len(graph.vertices) - 1

    # fixed_placement is a dict from node -> [x, y]
    fixed_placement = {}

    # Pick a couple nodes from the center and fix them
    for _ in range(n_fixed - 2):
        fixed_placement[rand.randint(1, N)] = (
                [rand.randint(0, bbox[0] + 1), rand.randint(0, bbox[1] + 1)])

    # Always pick the first and last node and place them on the boundry
    fixed_placement[0] = [0,       rand.randint(bbox[1] + 1)]
    fixed_placement[N] = [bbox[0], rand.randint(bbox[1] + 1)]

    return fixed_placement

def convex_overlap_constraints(graph, x, y, bbox=None, fixed_placement={}):
    bbox = _bbox_default(bbox)

    # Non-overlap constraints
    z = cvx.Int(len(graph.vertices), 4 * len(graph.vertices))
    overlap_constraints = [1 >= z, z >= 0]
    for a in graph.vertices:
        for b in range(a+1, len(graph.vertices)):
            # if a == b, we don't need an overlap constraint
            if a == b:
                continue

            # if a and b are fixed, we don't need any more constraints
            if a in fixed_placement.keys() and b in fixed_placement.keys():
                print("skipping {} and {}".format(a, b))
                continue

            # Here, there are 4 possibilities between each node A and B
            #
            # 1) A left  of B => x[B] - x[A] + M * z'[A, 4 * B + 0] >= 1
            # 2) A right of B => x[A] - x[B] + M * z'[A, 4 * B + 1] >= 1
            # 4) A below    B => y[B] - y[A] + M * z'[A, 4 * B + 2] >= 1
            # 3) A above    B => y[A] - y[B] + M * z'[A, 4 * B + 3] >= 1
            #
            # Where M and N are values big enough to make the constraint
            # non-active when z is 1.
            overlap_constraints.extend(
                 [x[b] - x[a] + (bbox[0] + 1) * (1 - z[a, 4 * b + 0]) >= 1,
                  x[a] - x[b] + (bbox[0] + 1) * (1 - z[a, 4 * b + 1]) >= 1])
                  # y[b] - y[a] + (bbox[1] + 1) * (1 - z[a, 4 * b + 2]) >= 1,
                  # y[a] - y[b] + (bbox[1] + 1) * (1 - z[a, 4 * b + 3]) >= 1])

            print("x[{0}] - x[{1}] + {2} * (1 - z[{1}, {3}]) >= 1".format(
                b, a, bbox[0] + 1, 4 * b + 0))
            print("x[{1}] - x[{0}] + {2} * (1 - z[{1}, {3}]) >= 1".format(
                b, a, bbox[0] + 1, 4 * b + 1))

            # Note that z is zero for the active constraint and one otherwise.
            # i.e. only one of z[A, 4 * B + 0,1,2,3] may be active (zero).
            # To force this, we require that the sum is equal to one and that Z is a bool
            overlap_constraints.extend(
                [z[a, 4 * b + 0] +
                 z[a, 4 * b + 1] == 1])

            print("z[{0}, {1}] + z[{0}, {2}] == 1".format(a, 4 * b + 0, 4 * b + 1))
            # overlap_constraints.extend(
            #     [z[a, 4 * b + 0] +
            #      z[a, 4 * b + 1] +
            #      z[a, 4 * b + 2] +
            #      z[a, 4 * b + 3] == 1])

    return z, overlap_constraints

def convex_placement_problem(graph, bbox=None, fixed_placement={}, ranks=None):
    """Construct a convex problem for the given graph"""
    bbox = _bbox_default(bbox)

    # Weight of each edge (i.e. how important it is)
    w = cvx.Parameter(len(graph.edges), sign="Positive")

    # Set the weights to be the weights specified in the graph
    # We return w so that this can be overriden later by the caller
    w.value = np.array([graph.weights[edge] for edge in graph.edges])

    # Positions for each node
    x = cvx.Int(len(graph.vertices))
    y = cvx.Int(len(graph.vertices))

    # Iterate over each edge and calculate that edge's cost
    # Split the x and y costs since they can be optimized seperately
    xcost = 0
    ycost = 0
    cost = 0
    for i, edge in enumerate(graph.edges):
        # Convert the edge to a list of indices that correspond to the vertices
        # connected by the edge
        vs = list(edge)

        # Compute the HPWL cost for this edge
        xcost += w[i] * cvx.max_entries(x[vs]) - cvx.min_entries(x[vs])
        ycost += w[i] * cvx.max_entries(y[vs]) - cvx.min_entries(y[vs])

        # head = edge.head
        # for v in edge.tail:
        #     cost += w[i] * cvx.norm2(cvx.hstack(x[head], y[head]) - cvx.hstack(x[v], y[v])) ** 2

    cost = xcost + ycost

    # Constrain the coords to the specified bounding box
    bbox_constraints_x = [bbox[0] >= x, x >= 0]
    bbox_constraints_y = [bbox[1] >= y, y >= 0]

    # Constrain the coords to the specified fixed placement
    fixed_constraints_x = [x[v] == pos[0]
            for (v, pos) in fixed_placement.viewitems()]
    fixed_constraints_y = [y[v] == pos[1]
            for (v, pos) in fixed_placement.viewitems()]

    # Overlap constraints
    # z, overlap_constraints = convex_overlap_constraints(graph, x, y, bbox, fixed_placement)
    z = None; overlap_constraints = []
    if ranks is not None:
        for x_rank, x_rank_next in zip(ranks[0:], ranks[:-1]):
            overlap_constraints += [x[x_rank] <= x[x_rank_next]]

        for y_rank, y_rank_next in zip(ranks[0:], ranks[:-1]):
            overlap_constraints += [y[y_rank] <= y[y_rank_next]]

    problem = cvx.Problem(cvx.Minimize(cost),
            bbox_constraints_x + fixed_constraints_x
            + bbox_constraints_y + fixed_constraints_y
            + overlap_constraints)

    return w, x, y, z, problem

def heruistic_initializer(graph, fixed_placement={}):
    # Convert the graph to a networkx graph since hypergraph is shitty for this
    nx = networkx_export(graph)

    # Make it undirected
    nx = nx.to_undirected()

    # change the weights to be 1/w
    for _, _, d in nx.edges_iter(data=True):
        d['weight'] = 1 / d['weight']

    # Add edges between the fixed nodes
    for h in fixed_placement.keys():
        for t in fixed_placement.keys():
            if h != t:
                nx.add_edge(h, t, weight=np.sum(np.abs(
                    np.array(fixed_placement[h]) - np.array(fixed_placement[t]))))

    # Find the path lenght between all pairs
    paths = networkx.all_pairs_dijkstra_path_length(nx, weight='weight')

    # Convert the paths into a matrix of distances
    N = len(graph.vertices)
    dists = np.array([v for row in paths.values() for v in row.values()]).reshape(N, N)

    # Do least squares
    x = np.zeros(N)
    y = np.zeros(N)
    for n in range(N):
        A = 2 * np.array([[fixed_placement[0][0] - pos[0], fixed_placement[0][1] - pos[1]]
            for pos in fixed_placement.values()])[1:]
        b = np.array([pos[0] ** 2 + pos[1] ** 2
            - fixed_placement[0][0] ** 2 - fixed_placement[0][1] ** 2
            + dists[n][0] ** 2 - dists[n][v] ** 2
            for v, pos in fixed_placement.items()])[1:]

        (x[n], y[n]), _, _, _ = np.linalg.lstsq(A, b)

    # Round to the nearest 1
    x = np.round(x)
    y = np.round(y)

    # Rank in the x and y direction
    # Breaks ties randomly
    rank_x = argsort(x)
    rank_y = argsort(y)

    return x, y

def simulated_anneal_convex(graph, bbox=None, fixed_placement={}, initial_ranks=None):
    if initial_ranks is None:
        initial_ranks = np.vstack(heruistic_initializer(graph, fixed_placement))

    # Constant temp
    t = 1

    non_fixed = [v for v in graph.vertices if v not in fixed_placement.keys()]
    ranks = np.copy(initial_ranks)
    sample_cost = float('inf')
    placement = None
    for _ in range(1000):
        # randomly select two nodes
        n1 = np.random.choice(non_fixed)
        n2 = np.random.choice(non_fixed)

        # Select a direction
        d = np.random.randint(2)

        # Swap
        ranks[d][n1], ranks[d][n2] = ranks[d][n2], ranks[d][n1]

        # Solve
        sample_placement = convex_placement(graph, bbox, fixed_placement, ranks)

        # Check the cost of the placement
        sample_cost = cost_HPWL(graph, placement)

        # accept?
        if np.random.rand() < min(1, sample_cost / cost):
            cost = sample_cost
            placement = sample_placement

    return placement

def simulated_anneal(graph, bbox=None, fixed_placement={}, seed=None):
    # Constant temp
    t = 0.1

    non_fixed = [v for v in graph.vertices if v not in fixed_placement.keys()]
    placement = random_placement(graph, bbox=bbox, fixed_placement=fixed_placement, best_of=10)
    cost = cost_HPWL(graph, placement)
    for _ in range(1000):
        # Select a random node
        n = np.random.choice(non_fixed)

        # Generate a new graph with everthing but this node fixed
        fixed = dict(zip(np.arange(len(placement)), placement))
        fixed.pop(n)

        sample_placement = random_placement(graph, bbox=bbox, fixed_placement=fixed)

        # Accept?
        sample_cost = cost_HPWL(graph, placement)
        if np.random.rand() < min(1, sample_cost / cost):
            cost = sample_cost
            placement = sample_placement

    return placement

def convex_placement(graph, bbox=None, fixed_placement={}, ranks=None):
    """Place the graph using a convex optimization approach.
    :graph: Hypergraph that gives the nodes to be placed.
    :bbox: [x, y] of upper right corner of bounding box for placed nodes.
    :fixed_placement: Dict of nodes to fixed [x, y] coord that may not be
        changed during optimization.
    """

    # Construct the problem
    w, x, y, z, problem = (
        convex_placement_problem(graph, bbox, fixed_placement, ranks))

    # First, solve the default problem
    if problem.solve() == float('inf'):
        raise Exception("Could not solve problem!")

    # Convert the x and y postition vectors into a placement
    placement = np.hstack((np.array(x.value), np.array(y.value)))

    return placement

def cost_HPWL(graph, placement):
    """Compute placement cost using the Half-Perimeter approximation"""
    cost = 0
    for edge in graph.edges:
        # get the weight for the current edge
        w = graph.weights[edge]

        # compute the position of each vertex attached to the edge
        pos = np.array([placement[v] for v in edge])

        # max, min over all vertices, seperating x and y
        maxs = np.max(pos, axis=0)
        mins = np.min(pos, axis=0)

        # cost is w * sum(max - min), summing over both x and y
        cost += w * np.sum(maxs - mins)

    return cost

def offset_generator(r):
    """Produce concentric offsets up to radius r"""
    yield [0, 0]
    for r in range(1, r+1):
        for x in range(-r, r+1):
            yield [x, r]
        for y in range(r-1, -r-1, -1):
            yield [r, y]
        for x in range(r-1, -r-1, -1):
            yield [x, -r]
        for y in range(-r, r+1):
            yield [-r, y]

def legalize_fast(bbox, placement):
    """Attempt to legalize the graph through clamping"""
    # Place each node
    placement = np.round(placement)
    for i, pos in enumerate(placement):
        # Round the placement to an integer
        pos = np.round(pos)

        # Not a great way to do this...
        for offset in offset_generator(np.max(bbox)):
            compare_to = np.vstack((placement[:i, :], placement[i+1:, :]))
            new_pos = pos + offset
            if (not (new_pos == compare_to).all(axis=1).any()
                    and bbox[0] >= new_pos[0] >= 0
                    and bbox[1] >= new_pos[1] >= 0):
                pos = new_pos
                break
        else:
            raise Exception("Could not legalize placement")

        # update the placement
        placement[i] = pos

    return placement

def distribution():
    seed = 0
    bbox = [10, 10]

    rand_costs = []
    conv_costs = []
    for _ in range(100):
        graph = random_circuit(30, seed=seed)
        fixed = random_fixed(graph, bbox=bbox, n_fixed=4)
        rand_pos = simulated_anneal(graph, bbox=bbox, fixed_placement=fixed, seed=seed)
        conv_pos = simulated_anneal_convex(graph, bbox=bbox, fixed_placement=fixed)

        legal_rand_pos = legalize_fast(bbox, rand_pos)
        legal_conv_pos = legalize_fast(bbox, conv_pos)

        rand_costs += [cost_HPWL(graph, legal_rand_pos)]
        conv_costs += [cost_HPWL(graph, legal_conv_pos)]

    return rand_costs, conv_costs

def plot_placement(graph, placement, title=None, bbox=None, fixed_placement={}, ax=None):
    bbox = _bbox_default(bbox)

    # Default to using the current axis
    ax = plt.gca() if ax is None else ax

    # Convert the graph to a network
    nx_graph = networkx_export(graph)

    # Convert our placement format into a placement nx can use
    nx_placement = dict(enumerate(placement))

    # Plot using the specified axis
    networkx.draw_networkx_nodes(nx_graph, pos=nx_placement, ax=ax,
            nodelist=[v for v in graph.vertices if v not in fixed_placement.keys()],
            node_size=100, node_color='r', node_shape='o')

    # Use blue squares for the fixed placements
    networkx.draw_networkx_nodes(nx_graph, pos=nx_placement, ax=ax,
            nodelist=fixed_placement.keys(),
            node_size=100, node_color='b', node_shape='v')

    networkx.draw_networkx_edges(nx_graph, pos=nx_placement, ax=ax,
            width=0.5, arrows=False)

    # Set ticks so we can see the grid
    ax.set_xticks(np.linspace(0, bbox[0], bbox[0] + 1))
    ax.set_yticks(np.linspace(0, bbox[1], bbox[1] + 1))
    ax.grid('on', color='lightgrey', linewidth=0.1)

    # Set the title
    ax.set_title("{}, cost: {:.2f}".format(title, cost_HPWL(graph, placement)))

def main(args=None):
    seed = 0
    bbox = [10, 10]

    # Create a new graph
    graph = random_circuit(30, seed=seed)

    # Randomly constrain some of the nodes
    fixed = random_fixed(graph, bbox=bbox, n_fixed=4)

    # Simulated annealed placement
    rand_pos = simulated_anneal(graph, bbox=bbox, fixed_placement=fixed, seed=seed)

    # Placement using convex optimization
    conv_pos = simulated_anneal_convex(graph, bbox=bbox, fixed_placement=fixed)

    # Legalize
    legal_rand_pos = legalize_fast(bbox, rand_pos)
    legal_conv_pos = legalize_fast(bbox, conv_pos)

    print(rand_pos)
    print(conv_pos)

    # Plot
    f, axarr = plt.subplots(2, sharex=True, sharey=True)
    plot_placement(graph, legal_rand_pos, 'Random', bbox=bbox, fixed_placement=fixed, ax=axarr[0])
    plot_placement(graph, legal_conv_pos, 'Convex', bbox=bbox, fixed_placement=fixed, ax=axarr[1])
    f.show()

if __name__ == '__main__':
    main(sys.argv[1:])
