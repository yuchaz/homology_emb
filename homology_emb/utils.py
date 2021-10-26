# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/yuchaz/homology_emb/blob/main/LICENSE

from __future__ import division, print_function, absolute_import
import numpy as np
import gudhi as gd
from collections import Counter


def choose_n_farthest_points(points, nb_points, seed=None):
    rdn_state = np.random if seed is None else np.random.RandomState(seed=seed)
    starting_point = rdn_state.choice(points.shape[0])
    out_points = np.array(gd.choose_n_farthest_points(
        points=points, nb_points=nb_points, starting_point=starting_point))
    return get_data_indices(out_points, points)

def get_data_indices(datasubset, fulldata):
    npoint = datasubset.shape[0]
    output = np.ones(npoint)
    for i in range(npoint):
        b = Counter(np.where((fulldata == datasubset[i,:] ))[0])
        output[i] = (b.most_common(1)[0][0])
    output = np.asarray(output,dtype = int)
    return output


def cknn_distance(distance_matrix, k):
    kth_dist = np.array([dist_i[np.argpartition(dist_i, k)[k]]
                         for dist_i in distance_matrix])
    rho_x_rho_y = np.sqrt(np.outer(kth_dist, kth_dist))
    return distance_matrix / rho_x_rho_y
