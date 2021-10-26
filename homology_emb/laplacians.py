# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/yuchaz/homology_emb/blob/main/LICENSE

from __future__ import division, print_function, absolute_import
import numpy as np
from numba import njit, prange
import math
from scipy.sparse import csr_matrix, diags, coo_matrix
from tqdm.auto import tqdm
from itertools import combinations
import warnings

import time
from datetime import timedelta


class HodgeLaplacian(object):
    def __init__(self, km1_simplices, k_simplices, kp1_simplices,
                 estimate_time=True):
        self.km1_simplices = km1_simplices
        self.k_simplices = k_simplices
        self.kp1_simplices = kp1_simplices

        self.Bk, self.Bkp1 = None, None
        self.need_calc_Bk, self.need_calc_Bkp1 = True, True

        # Pre-compile the boundary_map function!
        boundary_map(np.arange(4).reshape((2, 2)), np.arange(4))
        if estimate_time:
            self.estimate_bmap_calc_time()

    def k_laplacian(self):
        self._calc_Bk()
        self._calc_Bkp1()

        return self.Bk.T @ self.Bk + self.Bkp1 @ self.Bkp1.T

    def weighted_Laplacian(self, varepsilon, distance_matrix):
        self._calc_Bk()
        self._calc_Bkp1()
        w2, w1, w0 = calc_weights(self.Bkp1, self.Bk, self.kp1_simplices,
                                  distance_matrix, varepsilon)
        L1 = self._calc_Lkw(w2, w1, w0)
        return L1, w2, w1, w0

    def _calc_Lkw(self, wkp1, wk, wkm1):
        self._calc_Ak(wkm1, wk)
        self._calc_Akp1(wk, wkp1)
        Lkdown = self.Ak.T @ self.Ak
        Lkup = self.Akp1 @ self.Akp1.T
        Lkup.setdiag(np.ones(Lkup.shape[0]))

        return Lkdown + Lkup

    def _calc_Bk(self):
        if self.Bk is None or self.need_calc_Bk:
            self.need_calc_Bk = False
            self.Bk = boundary_map(self.k_simplices,
                                   self.km1_simplices).tocsr()

        return self.Bk

    def _calc_Bkp1(self):
        if self.Bkp1 is None or self.need_calc_Bkp1:
            self.need_calc_Bkp1 = False
            self.Bkp1 = boundary_map(self.kp1_simplices,
                                     self.k_simplices).tocsr()

        return self.Bkp1

    def _calc_Ak(self, wkm1, wk):
        self._calc_Bk()
        inv_wkm1 = np.ma.divide(1, wkm1).filled(0)
        self.Ak = diags(inv_wkm1 ** (1/2)) @ self.Bk @ diags(wk ** (1/2))

    def _calc_Akp1(self, wk, wkp1):
        self._calc_Bkp1()
        inv_wk = np.ma.divide(1, wk).filled(0)
        self.Akp1 = diags(inv_wk ** (1/2)) @ self.Bkp1 @ diags(wkp1 ** (1/2))

    def set_Bk(self, Bk):
        self.Bk = Bk
        self.need_calc_Bk = False

    def set_Bkp1(self, Bkp1):
        self.Bkp1 = Bkp1
        self.need_calc_Bkp1 = False

    def estimate_bmap_calc_time(self, till=50000):
        n2 = self.kp1_simplices.shape[0]
        if n2 < till:
            print('number of triangles too small, do not estimate!')
            return
        random_ix = np.random.choice(n2, till, replace=False)
        kp1_simplices = self.kp1_simplices[random_ix]
        start = time.perf_counter()
        boundary_map(kp1_simplices, self.k_simplices)
        end = time.perf_counter()
        elapsed_time = end - start
        est_time = math.ceil(self.kp1_simplices.shape[0] / till * elapsed_time)

        print('Need around %s to compute boundary map' %
              str(timedelta(seconds=int(est_time))))

    def num_disconnected_edges(self):
        self._calc_Bkp1()
        return np.sum(abs(self.Bkp1).sum(1) == 0)

    def num_disconnected_vertices(self):
        self._calc_Bk()
        return np.sum(abs(self.Bk).sum(1) == 0)


class HodgeCubicalLaplacianDim1(HodgeLaplacian):
    def __init__(self, km1_simplices=None, k_simplices=None,
                 kp1_simplices=None):
        HodgeLaplacian.__init__(self, km1_simplices, k_simplices,
                                kp1_simplices, False)

    def _calc_Bkp1(self):
        if self.k_simplices is None or self.kp1_simplices is None:
            raise ValueError('Should provide')
        if self.Bkp1 is None or self.need_calc_Bkp1:
            self.need_calc_Bkp1 = False

        dat = np.hstack([
            np.ones(self.kp1_simplices.shape[0]) * +1,
            np.ones(self.kp1_simplices.shape[0]) * -1,
            np.ones(self.kp1_simplices.shape[0]) * +1,
            np.ones(self.kp1_simplices.shape[0]) * -1,
        ])
        cols = self.kp1_simplices.T.flatten()
        rows = np.hstack([
            np.arange(self.kp1_simplices.shape[0]) for _ in range(4)
        ])
        n1, n2 = self.k_simplices.shape[0], self.kp1_simplices.shape[0]
        self.Bkp1 = coo_matrix((dat, (rows, cols)), shape=(n2, n1)).T.tocsr()
        return self.Bkp1

    def weighted_Laplacian(self):
        self._calc_Bk()
        self._calc_Bkp1()
        w2 = np.ones(self.kp1_simplices.shape[0])
        w1 = abs(self.Bkp1) @ w2
        w0 = abs(self.Bk) @ w1

        L1 = self._calc_Lkw(w2, w1, w0)
        return L1, w2, w1, w0


def boundary_map(simplices, faces, two_pointer=True, sequential=False):
    return coboundary_map(simplices, faces, sequential).T


def coboundary_map(simplices, faces, sequential=False):
    if isinstance(faces, int):
        faces = np.arange(faces)
    if faces.ndim == 1:
        faces = faces[:, None]
    dfaces = faces.shape[1]
    if dfaces > 2:
        warnings.warn('For B3 or up, switch to sequential algorithm due to the'
                      ' implementation constraint of numba and memory usage.')
        sequential = True

    if sequential:
        face_reverse_mapping = {
            tuple(face): idx for idx, face in enumerate(faces)
        }
    else:
        n = faces.max() + 1
        if dfaces == 1:
            face_reverse_mapping = np.arange(n, dtype=int)[:, None]
        elif dfaces == 2:
            face_reverse_mapping = np.zeros((n, n), dtype=int)
            for ix, (i, j) in enumerate(faces):
                face_reverse_mapping[i, j] = ix

    indptr = np.arange(simplices.shape[0]+1, dtype=int) * simplices.shape[1]
    data = np.zeros(simplices.shape, dtype=int)
    indices = np.zeros(simplices.shape, dtype=int)

    if not sequential:
        coboundary_map_parallel(
            simplices, data, indices, face_reverse_mapping)
    else:
        coboundary_map_sequential(
            simplices, data, indices, face_reverse_mapping)

    data, indices = map(lambda x: x.flatten(), [data, indices])

    return csr_matrix((data, indices, indptr), dtype=int,
                      shape=(simplices.shape[0], faces.shape[0]))


@njit(parallel=True, nogil=True)
def coboundary_map_parallel(simplices, data, indices, face_reverse_mapping):
    for ix in prange(simplices.shape[0]):
        permutation_per_simplex(
            simplices[ix], data[ix], indices[ix],
            face_reverse_mapping)


@njit
def permutation_per_simplex(simplex, data_row, indices_row,
                            face_reverse_mapping):
    dsimplex = len(simplex)
    for ixx, eid in enumerate(range(dsimplex-1, -1, -1)):
        face = [simplex[x] for x in range(dsimplex) if x != eid]
        if dsimplex == 3:
            indices_row[ixx] = face_reverse_mapping[face[0], face[1]]
        else:
            indices_row[ixx] = face_reverse_mapping[face[0], 0]
        data_row[ixx] = get_permutation(simplex, face)


def coboundary_map_sequential(simplices, data, indices, face_reverse_mapping):
    for ix in tqdm(range(simplices.shape[0])):
        d_simplex = simplices.shape[1]
        for ixx, slicing in enumerate(combinations(range(d_simplex),
                                                   d_simplex-1)):
            face = simplices[ix][list(slicing)]
            indices[ix, ixx] = face_reverse_mapping[tuple(face)]
            data[ix, ixx] = get_permutation(simplices[ix], face)


@njit(nogil=True, fastmath=True)
def get_permutation(simplex, face):
    nonzero, ix, iy = 0, 0, 0
    coface_idx = -1
    while ix < len(simplex) and iy < len(face):
        node_sim = simplex[ix]
        node_face = face[iy]
        if node_sim == node_face:
            nonzero += 1
            ix += 1
            iy += 1
        else:
            coface_idx = ix
            ix += 1

    if nonzero != len(face):
        return 0
    if coface_idx == -1:
        coface_idx = len(simplex) - 1
    return (-1) ** (coface_idx)


@njit(parallel=True, nogil=True)
def vr_weights(triangles, distance_matrix, varepsilon):
    n_tri = triangles.shape[0]
    weights = np.zeros(n_tri)
    for ix in prange(n_tri):
        tri = triangles[ix]
        a = distance_matrix[tri[0], tri[1]]
        b = distance_matrix[tri[0], tri[2]]
        c = distance_matrix[tri[1], tri[2]]
        weights[ix] = (np.exp(-a**2 / varepsilon**2) *
                       np.exp(-b**2 / varepsilon**2) *
                       np.exp(-c**2 / varepsilon**2))

    return weights


def calc_weights(Bkp1, Bk, triangles, distance_matrix, varepsilon):
    w2 = vr_weights(triangles, distance_matrix, varepsilon)
    w1 = abs(Bkp1).dot(w2)
    w0 = abs(Bk).dot(w1)
    return w2, w1, w0
