[![Build Status](https://travis-ci.org/tjof2/ctrwfractal.svg?branch=master)](https://travis-ci.org/tjof2/ctrwfractal)
[![Coverage Status](https://coveralls.io/repos/github/tjof2/ctrwfractal/badge.svg?branch=master)](https://coveralls.io/github/tjof2/ctrwfractal?branch=master)
[![DOI](https://zenodo.org/badge/58554121.svg)](https://zenodo.org/badge/latestdoi/58554121)

# ctrwfractal

**Modelling continuous-time random walks on fractal percolation clusters**

The percolation clusters are generated using a C++ port of code by [Mark Newman](http://www-personal.umich.edu/~mejn/percolation/). The original C code supplements the paper: [A fast Monte Carlo algorithm for site or bond percolation](http://aps.arxiv.org/abs/cond-mat/0101295/), M. E. J. Newman and R. M. Ziff, Phys. Rev. E 64, 016706 (2001). This code has been adapted to work on both square and honeycomb (i.e. graphene) lattices.

## Installation

**Dependencies**

This library makes use of the **[Armadillo](http://arma.sourceforge.net)** C++ linear algebra library, which needs to be installed first. It is recommended that you use a high-speed replacement for LAPACK and BLAS such as OpenBLAS, MKL or ACML; more information can be found in the [Armadillo
FAQs](http://arma.sourceforge.net/faq.html#dependencies).

**Building from source**

To build the library, unpack the source and `cd` into the unpacked directory, then type `make`:

```bash
$ tar -xzf ctrwfractal.tar.gz
$ cd ctrwfractal
$ make
```

## Usage

Example usage:
```
./fractalwalk -d 64 -f 0.5 -l Square
```

Percolation options:
```
-d, -dim
   Dimension of lattice
-f, -fraction
   Percolation fraction
-l, -lattice
   Lattice type ("Square" or "Honeycomb")
-o, -output
   Filename for output
-s, -seed
   Random seed
```
The following two options are for random walks on the cluster. If
the `-w` flag is `> 0`, then the program will simulate a number of random walks
on the percolation cluster.

```
-n, -nsteps ARG
   Length of random walks
-w, -walks ARG
   Simulate w random walks on this lattice
```

Copyright (C) 2016-2020 Tom Furnival. ctrwfractal is released free of charge under the GNU General Public License (GPLv3).
