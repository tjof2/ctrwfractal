# CTRWfractal

**Modelling continuous-time random walks on fractals**

The percolation clusters are generated using a C++ port of code by [Mark Newman](http://www-personal.umich.edu/~mejn/percolation/). The original C code supplements the paper: [A fast Monte Carlo algorithm for site or bond percolation](http://aps.arxiv.org/abs/cond-mat/0101295/), M. E. J. Newman and R. M. Ziff, Phys. Rev. E 64, 016706 (2001). This code has been adapted to work on both square and honeycomb (i.e. graphene) lattices.

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

_Copyright (C) 2016-2019 Tom Furnival._
