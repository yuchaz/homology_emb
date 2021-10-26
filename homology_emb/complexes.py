# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/yuchaz/homology_emb/blob/main/LICENSE

from __future__ import division, print_function, absolute_import
from scipy.spatial.distance import squareform, pdist
import numpy as np
from itertools import combinations
import gudhi as gd
from tqdm.auto import tqdm


class SimplicialComplexDim2(object):
    def __init__(self, distance_matrix, delta=None):
        self.distance_matrix = distance_matrix
        self.delta_default = delta
        self.fitted = False

    def get_n1(self, delta=None):
        delta = self._get_delta(delta)
        return np.sum(squareform(self.distance_matrix) < delta)

    def _get_delta(self, delta):
        if self.delta_default is None and delta is None:
            raise ValueError('You must provide delta to call this function')
        elif delta is None:
            delta = self.delta_default
        return delta

    def fit(self, delta=None, no_triangle=False):
        delta = self._get_delta(delta)
        vr_func = (compute_vr_edges_triangles if not no_triangle else
                   compute_vr_edges_no_triangles)
        self.nodes, self.edges, self.triangles = vr_func(
            self.distance_matrix, delta
        )
        self.edge_distance = np.array([
            self.distance_matrix[s, t] for s, t in self.edges
        ])
        self.fitted = True


def compute_vr_edges_triangles(distance_matrix, delta):
    vr_complex = gd.RipsComplex(distance_matrix=distance_matrix,
                                max_edge_length=delta)
    vr_spx_tree = vr_complex.create_simplex_tree(max_dimension=2)
    nodes_ = list(vr_spx_tree.get_skeleton(0))
    edges_ = list(vr_spx_tree.get_skeleton(1))
    triangles_ = list(vr_spx_tree.get_skeleton(2))
    nodes = np.array([v[0] for v in nodes_])
    edges_np = np.array([v[0] for v in edges_ if len(v[0]) == 2])
    edges_alpha = np.array([v[1] for v in edges_ if len(v[0]) == 2])

    triangles_np = np.array([v[0] for v in triangles_ if len(v[0]) == 3])
    triangles_alpha = np.array([v[1] for v in triangles_ if len(v[0]) == 3])
    del triangles_
    triangles = np.array([
        tri for tri, alpha in zip(triangles_np, triangles_alpha)
        if alpha < delta
    ])
    edges = np.array([
        edge for edge, alpha in zip(edges_np, edges_alpha) if alpha < delta
    ])
    return nodes, edges, triangles


def compute_vr_edges_no_triangles(distance_matrix, delta):
    n0 = distance_matrix.shape[0]
    nodes = np.arange(n0)
    n1_max = int(n0 * (n0 - 1) / 2)
    edges = [(i, j) for i, j in tqdm(combinations(nodes, 2), total=n1_max)
             if distance_matrix[i, j] < delta]
    return nodes, np.array(edges), np.array([])


class CubicalComplexDim2(object):
    def __init__(self, data):
        self.data = data
        distance_matrix = squareform(pdist(data))
        self.fitted = False
        self._scx = SimplicialComplexDim2(distance_matrix, 1.01)

    def get_n1(self):
        return self._scx.get_n1()

    def fit(self):
        self._scx.fit(no_triangle=True)
        self.nodes = self._scx.nodes
        self.edges = self._scx.edges
        self.rectangles_in_edge, self.rectangles_in_node = compute_rectangle(
            self.data, self.edges
        )
        self.fitted = True
        self.edge_distance = self._scx.edge_distance


def compute_rectangle(data, edges):
    '''
    Inputs
    ------
    data: n0 x 2 array
        data[:, 0] and data[:, 1] be the x and y coord of the image.

    edges2idx: dict
        edge to eid (edge id) mapping; edge is e = (nid1, nid2)
    '''
    nonzx, nonzy = data.T
    node_coord2idx = {
        (nx, ny): idx for idx, (nx, ny) in enumerate(zip(nonzx, nonzy))
    }
    edges2idx = {(ei, ej): idx for idx, (ei, ej) in enumerate(edges)}
    rectangles, rectangles_in_node = [], []
    for nx, ny in zip(tqdm(nonzx), nonzy):
        try:
            n1, n2, n3, n4 = map(
                lambda x: node_coord2idx[x],
                [(nx, ny), (nx+1, ny), (nx, ny+1), (nx+1, ny+1)])
        except KeyError:
            continue

        e1, e2, e3, e4 = map(lambda x: edges2idx[x],
                             [(n1, n2), (n1, n3), (n2, n4), (n3, n4)])
        rectangles.append((e1, e2, e3, e4))
        rectangles_in_node.append((n1, n2, n3, n4))

    rec_ix = np.array(sorted(range(len(rectangles)),
                             key=rectangles.__getitem__))
    rectangles = np.array(rectangles)[rec_ix]
    rectangles_in_node = np.array(rectangles_in_node)[rec_ix]

    return rectangles, rectangles_in_node
