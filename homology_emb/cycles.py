# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/yuchaz/homology_emb/blob/main/LICENSE

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from mne.preprocessing import infomax


class HomologyLoop(object):
    def __init__(self, harmonic_evects, scx, w1, precompute=True,
                 infomax_kwargs={}):
        if harmonic_evects.ndim == 1:
            harmonic_evects = harmonic_evects[:, None]
        self.harmonic_evects = harmonic_evects
        self.n1, self.ncomp = harmonic_evects.shape
        self.independent_harmonic_evects = None
        if not scx.fitted:
            scx.fit()
        self.scx = scx
        self.w1 = w1
        if precompute:
            self.max_info_rotation(**infomax_kwargs)

    def max_info_rotation(self, seed=42, **kwargs):
        if self.harmonic_evects.shape[1] == 1:
            print('Infomax is not run since beta_1 = 1')
            self.independent_harmonic_evects = self.harmonic_evects
        else:
            kwargs['random_state'] = seed
            self.unmix_mat = infomax(self.harmonic_evects, **kwargs)
            self.mix_mat = np.linalg.inv(self.unmix_mat)
            mix_mat_norm = self.mix_mat / np.linalg.norm(
                self.mix_mat, axis=0, keepdims=True
            )
            self.rot_mat = np.linalg.qr(mix_mat_norm)[0]
            self.independent_harmonic_evects = (
                self.rot_mat.T @ self.harmonic_evects.T
            ).T
        return self.independent_harmonic_evects

    def find_loops(self, use_independent=True, use_Alg_S1=False):
        used_flow = (
            self.independent_harmonic_evects if use_independent else
            self.harmonic_evects
        )
        used_flow = (
            used_flow * (np.ma.power(self.w1[:, None], -0.5).filled(0))
        )

        used_weights = [
            self.scx.edge_distance.copy() for _ in range(self.ncomp)
        ]
        return [
            homology_cycle_dijkstra_digraph(
                flow, self.scx.edges, weights=w, use_Alg_S1=use_Alg_S1,
                keep_top=(1 / self.ncomp)
            )
            for flow, w in zip(used_flow.T, used_weights)
        ]


def get_neighbors(A, v):
    return A.indices[A.indptr[v]:A.indptr[v+1]]


def find_path(predecessors, target):
    path = [target]
    si = target
    while predecessors[si] != -9999:
        path.append(predecessors[si])
        si = predecessors[si]
    return np.array(path)


def sort_and_parity(i, j):
    if i < j:
        return (i, j), 1
    else:
        return (j, i), -1


def find_next_edge(A, v, phi, edges, edges_to_ix, visited_nodes=None,
                   visited_edges=None):
    neigh_v = get_neighbors(A, v)

    def _cond(v, ni):
        if visited_nodes is None and visited_edges is None:
            return True
        cond = True
        if visited_nodes is not None:
            cond = cond and (ni not in visited_nodes)
        if visited_edges is not None:
            cond = cond and (tuple(sorted((v, ni))) not in visited_edges)
        return cond

    edges_neighs, edges_parity = map(np.array, zip(*[
        sort_and_parity(v, ni) for ni in neigh_v
        if _cond(v, ni)
    ]))

    edges_neigh_ix = [edges_to_ix[tuple(en)] for en in edges_neighs]
    comp_val = phi[edges_neigh_ix] * edges_parity
    next_edge = edges[edges_neigh_ix[comp_val.argmax()]]
    nv = next_edge[next_edge != v].item()
    return nv


def homology_cycle_dijkstra_digraph(
    phi, edges, weights, use_Alg_S1=False, keep_top=None,
):
    if use_Alg_S1:
        s0, __ = edges[phi.argmax()]
        return _homology_cycle_point(phi, edges, s0, weights)[0]
    else:
        return _homology_cycle_all(phi, edges, weights, keep_top=keep_top)[0]


def _homology_cycle_point(phi, edges, s0, weights=None):
    data, row_ind, col_ind = [], [], []
    if weights is None:
        weights = np.ones(edges.shape[0])
    for (i, j), phi_e, weight in zip(edges, phi, weights):
        s, t = (i, j) if phi_e > 0 else (j, i)
        row_ind.append(s)
        col_ind.append(t)
        data.append(weight)

    n0 = edges.max() + 1
    graph = csr_matrix((data, (row_ind, col_ind)), shape=(n0, n0))

    distances, predecessors = [], []
    neigh_s0 = get_neighbors(graph, s0)
    for s1 in neigh_s0:
        dist, precede = dijkstra(
            csgraph=graph, directed=True, indices=s1, return_predecessors=True
        )
        distances.append(dist)
        predecessors.append(precede)

    min_dist, min_ix = min([
        (dist[s0] + graph[s0, s1], ix)
        for ix, (dist, s1) in enumerate(zip(distances, neigh_s0))
    ])
    min_pred = predecessors[min_ix]
    cycle = np.append(find_path(min_pred, s0), s0)[::-1]
    return cycle, min_dist


def _homology_cycle_all(phi, edges, weights=None, keep_top=None):
    if keep_top is None:
        raise ValueError('Need to provide `keep_top`')
    data, row_ind, col_ind = [], [], []
    thres = np.percentile(abs(phi), (1-keep_top) * 100)
    if weights is None:
        weights = np.ones(edges.shape[0])

    oriented_edges = []
    for (i, j), phi_e, weight in zip(edges, phi, weights):
        if abs(phi_e) < thres:
            continue
        s, t = (i, j) if phi_e > 0 else (j, i)
        row_ind.append(s)
        col_ind.append(t)
        data.append(weight)
        oriented_edges.append((s, t))

    oriented_edges = np.array(oriented_edges)

    n0 = edges.max() + 1
    graph = csr_matrix((data, (row_ind, col_ind)), shape=(n0, n0))

    dist, precede = dijkstra(
        csgraph=graph, directed=True, return_predecessors=True
    )
    edge_geodesic = np.array([dist[j, i] for (i, j) in oriented_edges])
    s0, min_s = oriented_edges[edge_geodesic.argmin()]
    min_dist = edge_geodesic.min()
    min_pred = precede[min_s]
    cycle = np.append(find_path(min_pred, s0), s0)[::-1]
    del dist, precede
    return cycle, min_dist
