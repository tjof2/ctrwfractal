[![Build Status](https://travis-ci.org/tjof2/ctrwfractal.svg?branch=master)](https://travis-ci.org/tjof2/ctrwfractal)
[![Coverage Status](https://coveralls.io/repos/github/tjof2/ctrwfractal/badge.svg?branch=master)](https://coveralls.io/github/tjof2/ctrwfractal?branch=master)
[![DOI](https://zenodo.org/badge/58554121.svg)](https://zenodo.org/badge/latestdoi/58554121)

# ctrwfractal

**Modelling continuous-time random walks on fractal percolation clusters**

This code implements the hybrid model for spatial and temporal disorder on graphene oxide, as described in [our paper](https://dx.doi.org/10.1016/j.cplett.2017.04.071) in *Chemical Physics Letters*.

In site percolation on a regular 2D lattice, each site is occupied with probability _p_. Spatial disorder is incorporated as intermediate values of _p_, where clusters of accessible sites will exhibit fractal-like characteristics. Particles undertaking random walks on such clusters will exhibit _anomalous_ diffusion. Our hybrid model extends this diffusive behaviour using the theory of continuous-time random walks (CTRW), where the waiting times between jumps are drawn from a power-law distribution.

If you use this code in a publication, please cite our work:

> T. Furnival, R. K. Leary, E. C. Tyo, S. Vajda, Q. M. Ramasse, J. M. Thomas, P. D. Bristowe and P. A. Midgley, "Anomalous diffusion of single metal atoms on a graphene oxide support", *Chem. Phys. Lett.*, vol. 683, pp. 370–374, 2017. DOI:[10.1016/j.cplett.2017.04.07](https://dx.doi.org/10.1016/j.cplett.2017.04.071).

ctrwfractal is released free of charge under the GNU General Public License ([GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html)).

## Installation

<!-- The easiest way to install the package is with `pip`:

```bash
$ pip install -U ctrwfractal
$ pip install git+https://github.com/tjof2/ctrwfractal.git
``` -->

#### Building from source

This library makes use of the **[Armadillo](http://arma.sourceforge.net)** C++ linear algebra library, which needs to be installed first. It is recommended that you use a high-speed replacement for LAPACK and BLAS such as OpenBLAS, MKL or ACML; more information can be found in the [Armadillo
FAQs](http://arma.sourceforge.net/faq.html#dependencies).

```bash
$ sudo apt-get install libarmadillo-dev
# Alternatively, download and build Armadillo from source
$ sh ./install-dependencies.sh
```

To build the Python package from source:

```bash
$ git clone https://github.com/tjof2/ctrwfractal.git
$ cd ctrwfractal
$ pip install -e .
```

## Usage

```python

from ctrwfractal import CTRWfractal

est = CTRWfractal(
   grid_size=64,
   lattice_type="square",
   n_walks=1,
   n_steps=100,
)
est.run()

# Attributes:
#   est.lattice_
#   est.clusters_
#   est.walks_
#   est.analysis_
#   est.occupied_fraction_
```

Both square and honeycomb (i.e. graphene) lattices are supported. The percolation clusters are generated using the periodic algorithm described in *[A fast Monte Carlo algorithm for site or bond percolation](http://aps.arxiv.org/abs/cond-mat/0101295/), M. E. J. Newman and R. M. Ziff, Phys. Rev. E 64, 016706 (2001).*

Copyright (C) 2016-2020 Tom Furnival.
