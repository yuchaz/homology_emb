# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/yuchaz/homology_emb/blob/main/LICENSE

from configs import *
from homology_emb import *
import numpy as np
from datagen import load_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.spatial.distance import squareform, pdist
from megaman.utils.eigendecomp import eigen_decomposition
from tqdm.auto import tqdm
from plotter import pairplot
from plotter import savefig_function_gen
savefig = savefig_function_gen('./figs')
inline_mpl()
setup_color_palettes(fs=1.0)


def factorize_emb_and_find_shortest_loop(dat):
    dist_mat = squareform(pdist(dat.point_cloud))
    dat.distance_matrix = cknn_distance(dist_mat, 30)

    dat.scx = SimplicialComplexDim2(dat.distance_matrix, dat.delta)
    print(f'n1 = {dat.scx.get_n1()}')

    dat.scx.fit()
    dat.hodge_lapla = HodgeLaplacian(dat.scx.nodes, dat.scx.edges,
                                     dat.scx.triangles)

    dat.varepsilon = dat.delta ** (2/3) / 3
    dat.L1w, dat.w2, dat.w1, dat.w0 = dat.hodge_lapla.weighted_Laplacian(
        dat.varepsilon, dat.distance_matrix
    )

    dat.evalues1, dat.evects1 = eigen_decomposition(
        dat.L1w, 20, eigen_solver='lobpcg', drop_first=False, largest=False,
        solver_kwds=dict(maxiter=500, tol=1e-6, verbosityLevel=0)
    )

    dat.hloop = HomologyLoop(dat.evects1[:, :dat.beta1], dat.scx, dat.w1)
    dat.cycles_indep = dat.hloop.find_loops()
    dat.cycles_orig = dat.hloop.find_loops(False)
    return dat

# Generate all synthetic manifolds and run our framework
alias_all = [
    'PUNCTPLANE', 'TORUS', '3-TORUS', 'GENUS-2', 'TORI-CONCAT'
]
print('Generating data...')
dat_all = [load_data(alias) for alias in tqdm(alias_all)]
print('Compute L1, Y, Z, and the shortest loops...')
dat_all = [factorize_emb_and_find_shortest_loop(dat)
            for dat in tqdm(dat_all)]
punctplane, torus, three_torus, genus_two, tori_concat = dat_all


def punctplane_emb(ax, indep):
    used_emb = (punctplane.hloop.independent_harmonic_evects if indep else
                punctplane.hloop.harmonic_evects)
    ax.scatter(*used_emb.T, s=1, edgecolors='none', rasterized=True)
    label = 'z' if indep else 'y'
    ax.set_xlabel(r'$%s_1$' % label)
    ax.set_ylabel(r'$%s_2$' % label)
    ax.grid(False)
    sns.despine(ax=ax)
    ax.tick_params(bottom=True, left=True)
    ax.axis('equal')


def punctplane_loops(ax, indep):
    used_emb = (punctplane.hloop.independent_harmonic_evects if indep else
                punctplane.hloop.harmonic_evects)
    ax.scatter(*punctplane.point_cloud.T[[0, 1]], s=1, edgecolors='none')
    cycles = punctplane.cycles_indep if indep else punctplane.cycles_orig
    for ix, cycle in enumerate(cycles):
        ax.scatter(*punctplane.point_cloud[cycle].T[[0, 1]], s=10, c=color_hex(1+ix))

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.grid(False)
    sns.despine(ax=ax)
    ax.tick_params(bottom=True, left=True)
    ax.axis('equal')


def torus_emb(ax, indep):
    used_emb = (torus.hloop.independent_harmonic_evects if indep else
                torus.hloop.harmonic_evects)
    ax.scatter(*used_emb.T, s=1, edgecolors='none', rasterized=True)
    label = 'z' if indep else 'y'
    ax.set_xlabel(r'$%s_1$' % label)
    ax.set_ylabel(r'$%s_2$' % label)
    ax.grid(False)
    sns.despine(ax=ax)
    ax.tick_params(bottom=True, left=True)
    ax.axis('equal')


def torus_loops(ax, indep):
    ax.scatter(*torus.point_cloud.T[:3], s=1)
    cycles = torus.cycles_indep if indep else torus.cycles_orig

    for cycle in cycles:
        ax.scatter(*torus.point_cloud[cycle].T[:3], s=50)

    ax.view_init(49, -59)
    label = 'z' if indep else 'y'
    ax.set_xlabel(r'$%s_1$' % label)
    ax.set_ylabel(r'$%s_2$' % label)
    ax.set_zlabel(r'$%s_3$' % label)


def three_torus_loops(ax, indep, beautify=True):
    ax.scatter(*three_torus.intrinsic_coord.T, s=.1)
    cycles = three_torus.cycles_indep if indep else three_torus.cycles_orig

    for ix, cycle in enumerate(cycles):
        toplot = three_torus.intrinsic_coord[cycle]
        # Move 0 -> 2pi to make the extracted loops more interpretable
        if beautify and ix == 1:
            ixx = toplot[:, 0] < np.pi
            toplot[ixx, 0] = toplot[ixx, 0] + np.pi*2
        if beautify and ix == 2:
            ixx = toplot[:, 1] < np.pi
            toplot[ixx, 1] = toplot[ixx, 1] + np.pi*2

        ax.scatter(*toplot.T[:3], s=50)

    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_yticks([0, np.pi, 2*np.pi])
    ax.set_zticks([0, np.pi, 2*np.pi])

    ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'], fontsize=15)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'], fontsize=15)
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zticklabels([r'$0$', r'$\pi$', r'$2\pi$'], fontsize=15)
    ax.set_zlabel(r'$\theta_3$', labelpad=.5)

    ax.tick_params(axis='x', pad=-5)
    ax.tick_params(axis='y', pad=-5)
    ax.tick_params(axis='z', pad=0)


def three_torus_emb(ax, indep):
    used_emb = (three_torus.hloop.independent_harmonic_evects if indep else
                three_torus.hloop.harmonic_evects)

    ax.scatter(*used_emb.T, s=.1, rasterized=True)
    label = 'z' if indep else 'y'
    ax.set_xlabel(r'$%s_1$' % label)
    ax.set_ylabel(r'$%s_2$' % label)
    ax.set_zlabel(r'$%s_3$' % label)
    ax.tick_params(axis='z', pad=0)


def genus_two_emb(ax, indep):
    used_emb = (genus_two.hloop.independent_harmonic_evects if indep else
                genus_two.hloop.harmonic_evects)

    ax.scatter(*used_emb.T, s=.1, rasterized=True)
    label = 'z' if indep else 'y'
    ax.set_xlabel(r'$%s_1$' % label)
    ax.set_ylabel(r'$%s_2$' % label)
    ax.set_zlabel(r'$%s_3$' % label)
    ax.tick_params(axis='z', pad=0)


def genus_two_loops(ax, indep):
    ax.scatter(*genus_two.point_cloud.T, s=1, edgecolors=None, alpha=.5)
    cycles = genus_two.cycles_indep if indep else genus_two.cycles_orig

    for cycle in cycles:
        ax.scatter(*genus_two.point_cloud[cycle].T, s=50)
    ax.set_zlim([-.3, .3])
    ax.view_init(47, -104)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$x_3$')


def tori_concat_loops(ax, indep):
    cycles = tori_concat.cycles_indep if indep else tori_concat.cycles_orig
    ax.scatter(*tori_concat.point_cloud.T[:2], s=1, edgecolors='none')
    for cycle in cycles:
        ax.scatter(*tori_concat.point_cloud[cycle].T[:2], s=10)
    ax.axis('equal')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.grid(False)
    sns.despine(ax=ax)
    ax.tick_params(bottom=True, left=True)
    ax.axis('equal')


def make_blank_plot(ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def main():
    print('Plot results for all synthetic manifolds...')
    data_strings = ['punctplane', 'torus', 'three_torus', 'genus_two']
    used_functions = []
    is_3d = []
    for ix, ds in enumerate(data_strings):
        if ix == 0:
            is_3d.append([False for _ in range(4)])
        elif ix == 1:
            is_3d.append([False, True, False, True])
        else:
            is_3d.append([True for _ in range(4)])

        used_functions.append(list(map(
            lambda x: eval(f'lambda ax: {x[0]}(ax, {x[1]})'),
            zip([f'{ds}_emb', f'{ds}_loops', f'{ds}_emb', f'{ds}_loops'],
                [True, True, False, False])
        )))
    with sns.plotting_context('paper', 2.0):
        fig = plt.figure(figsize=(20, 20), constrained_layout=True)
        gs = fig.add_gridspec(ncols=5, nrows=6, height_ratios=(.1, 1, 1, 1, 1, 1),
                            width_ratios=(.1, 1, 1, 1, 1))

        axZ = fig.add_subplot(gs[0, 1:3])
        make_blank_plot(axZ)
        axZ.set_xlabel(r'Indep. homology embedding $\vec{Z}$')

        axY = fig.add_subplot(gs[0, 3:])
        make_blank_plot(axY)
        axY.set_xlabel(r'Homology embedding $\vec{Y}$')

        for i in range(5):
            ax = fig.add_subplot(gs[i+1, 0])
            make_blank_plot(ax)
            ax.set_ylabel(r'\texttt{%s}' % alias_all[i])

        for r in range(4):
            for c in range(4):
                if is_3d[r][c]:
                    ax = fig.add_subplot(gs[r+1, c+1], projection='3d')
                else:
                    ax = fig.add_subplot(gs[r+1, c+1])

                used_functions[r][c](ax)


        ax = fig.add_subplot(gs[5, 1:3])
        tori_concat_loops(ax, True)

        ax = fig.add_subplot(gs[5, 3:])
        tori_concat_loops(ax, False)
        savefig('all_exp')
        plt.close()

    print('Generate pairwise scatter plots for GENUS-2 (Figure S1)...')
    pairplot(genus_two.hloop)
    savefig('pairplot_genus_two')
    plt.close()

    print('Generate pairwise scatter plots for TORI-CONCAT (Figure S2)...')
    pairplot(tori_concat.hloop)
    savefig('pairplot_tori_concat')
    plt.close()

    print('Finished!')


if __name__ == '__main__':
    main()
