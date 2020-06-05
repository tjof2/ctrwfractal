# Copyright 2016-2020 Tom Furnival
#
# This file is part of ctrwfractal.
#
# ctrwfractal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ctrwfractal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ctrwfractal.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from ._ctrwfractal import ctrw_fractal


class CTRWfractal:
    """Continuous-time random walks on 2D percolation clusters.

    The percolation clusters are generated using the periodic algorithm
    described in [New2001]_.

    Parameters
    ----------
    grid_size : int, default=32
        Dimensions of the 2D lattice. The total number of sites in the
        lattice will depend on the ``lattice_type``; for example, a
        square lattice will have ``grid_size ** 2`` sites in total.
    lattice_type : str {"square", "honeycomb"}, default="square"
        - If "square", then a 2D square lattice is generated.
        - If "honeycomb", then a 2D honeycomb (or graphene) lattice
          is generated.
    threshold : None or float, default=None
        The fraction of occupied sites on the lattice. If None,
        then the critical percolation threshold for the given
        ``lattice_type`` is used.
    walk_type : str {"all", "largest"}, default="all"
        - If "all", then the random walks can occur on any of the
          clusters on the 2D lattice.
        - If "largest", then the random walks will only occur on
          the largest cluster on the 2D lattice.
    n_walks : None or int, default=None
        Simulate ``n_walks`` random walks on the 2D lattice.
    n_steps : None or int, default=None
        Length of random walks on the 2D lattice.
    beta : None or float, default=None

    tau0 : None or float, default=None

    noise : None or float, default=None
        If not None, add zero-mean Gaussian noise to the random walks
        with standard deviation = ``noise``.
    random_seed : None or int, default=None

    n_jobs : None or int, default=None
        The number of threads to use for the random walk analysis, which
        is performed in parallel over ``n_walks``. None means using a
        single thread, while -1 means using all threads dependent on the
        available hardware.

    Attributes
    ----------
    clusters_ : array-like, shape (n_sites,)

    lattice_ : array-like, shape (2, n_sites)

    walks_ : None or array-like, shape (n_walks, n_steps, 2)
        If ``n_walks``
    analysis_ : None or pandas.DataFrame
        If ``n_walks``

    Notes
    -----
    See [Jac2014]_ for details on critical thresholds for percolation clusters.

    References
    ----------
    .. [New2001] M. E. J. Newman and R. M. Ziff, "A fast Monte Carlo algorithm
                 for site or bond percolation", Phys. Rev. E 64, 016706 (2001).
    .. [Jac2014] J. L. Jacobsen, "High-precision percolation thresholds and
                 Potts-model critical manifolds from graph polynomials",
                 J. Phys. A 47(13), 135001, (2014).

    """

    def __init__(
        self,
        grid_size=32,
        lattice_type="square",
        threshold=None,
        walk_type="all",
        n_walks=None,
        n_steps=None,
        beta=None,
        tau0=None,
        noise=None,
        random_seed=None,
        n_jobs=None,
    ):
        self.grid_size = grid_size
        self.lattice_type = lattice_type
        self.threshold = threshold
        self.walk_type = walk_type
        self.n_walks = n_walks
        self.n_steps = n_steps
        self.beta = beta
        self.tau0 = tau0
        self.noise = noise
        self.random_seed = random_seed
        self.n_jobs = n_jobs

        self._has_run = False

    def _analysis_to_df(self, analysis):
        columns = ["EnsembleMSD", "EnsembleTimeAveragedMSD", "ErgodicityBreaking"]
        columns.extend([f"TimeAveragedMSD_Walk{i}" for i in range(self.n_walks_)])

        return pd.DataFrame(analysis, columns=columns)

    def run(self):
        """

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        lattice_types = {"square": 0, "honeycomb": 1}
        lattice_thresholds = {"square": 0.592746, "honeycomb": 0.697040230}
        walk_types = {"all": 0, "largest": 1}

        self.lattice_type_ = lattice_types.get(self.lattice_type, None)
        self.walk_type_ = walk_types.get(self.walk_type, None)

        # If no threshold given, use the critical values
        self.threshold_ = (
            lattice_thresholds.get(self.lattice_type, 0.0)
            if self.threshold is None
            else self.threshold
        )

        # C++ uses numerical values instead of None for defaults
        self.n_walks_ = 0 if self.n_walks is None else self.n_walks
        self.n_steps_ = 0 if self.n_steps is None else self.n_steps
        self.beta_ = 0.0 if self.beta is None else self.beta
        self.tau0_ = 1.0 if self.tau0 is None else self.tau0
        self.noise_ = 0.0 if self.noise is None else self.noise
        self.random_seed_ = -1 if self.random_seed is None else self.random_seed
        self.n_jobs_ = 0 if self.n_jobs is None else self.n_jobs

        # Check lattice & walk types are supported
        if self.lattice_type_ is None:
            raise ValueError(
                f"Invalid lattice_type parameter: got '{self.lattice_type}' "
                f"instead of one of {lattice_types.keys()}"
            )

        if self.walk_type_ is None:
            raise ValueError(
                f"Invalid walk_type parameter: got '{self.walk_type}' "
                f"instead of one of {walk_types.keys()}"
            )

        # Check parameter ranges
        if self.threshold_ < 0.0 or self.threshold_ > 1.0:
            raise ValueError(
                f"Invalid threshold parameter: got '{self.threshold_}' "
                f"instead of a float between 0.0 and 1.0"
            )

        if self.beta_ < 0.0:
            raise ValueError(
                f"Invalid beta parameter: got '{self.beta_}' "
                f"instead of a float >= 0.0"
            )

        if self.tau0_ < 0.0:
            raise ValueError(
                f"Invalid tau0 parameter: got '{self.tau0_}' "
                f"instead of a float >= 0.0"
            )

        if self.noise_ < 0.0:
            raise ValueError(
                f"Invalid noise parameter: got '{self.noise_}' "
                f"instead of a float >= 0.0"
            )

        # Now we can safely call the C++ function
        res = ctrw_fractal(
            grid_size=self.grid_size,
            n_walks=self.n_walks_,
            n_steps=self.n_steps_,
            threshold=self.threshold_,
            beta=self.beta_,
            tau0=self.tau0_,
            noise=self.noise_,
            lattice_type=self.lattice_type_,
            walk_type=self.walk_type_,
            random_seed=self.random_seed_,
            n_jobs=self.n_jobs_,
        )

        self.clusters_ = res[0]
        self.lattice_ = res[1]

        if self.n_walks_ > 0 and self.n_steps_ > 0:
            self.walks_ = res[2]
            self.analysis_ = self._analysis_to_df(res[3])
        else:
            self.walks_ = None
            self.analysis_ = None

        self._has_run = True

        return self

    def plot_lattice(self, ax=None):
        if not self._has_run:
            self.run()

    def plot_walks(self, ax=None):
        if not self._has_run:
            self.run()
