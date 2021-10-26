# The decomposition of the higher-order homology embedding constructed from the $k$-Laplacian
- Author: Yu-Chia Chen <yuchaz@uw.edu>
- LICENSE: [Simplified BSD](https://github.com/yuchaz/homology_emb/blob/main/LICENSE)

The source code includes example for *synthetic* manifolds only. Please follow the links in the `neurips21-supp.pdf` to download the real datasets.

## Dependencies
- python 3.7 (fully tested)
- Scientific computing: `numpy`, `scipy`, `numba`, `mne`
- Plotting: `matplotlib`, `seaborn`
- Manifold learning:  `gudhi`, [`megaman`](https://github.com/mmp2/megaman) (download and install from github)
- Utilities: `tqdm`

## Experiments

Run the following command to generate figures

```python
python run_synthetic_manifolds.py
```

It will take around 5min to finish.

## Links
- [arXiv paper](https://arxiv.org/abs/2107.10970)
- [Project page](https://yuchaz.github.io/publication/2021-harmonic-emb)
- [Slides](https://yuchaz.github.io/files/2021-harmonic-emb-slides.pdf)
- [Source code](https://github.com/yuchaz/homology_emb) (this repo)
