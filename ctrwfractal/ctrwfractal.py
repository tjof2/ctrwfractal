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

from ._ctrwfractal import ctrw_fractal


class CTRWfractal:
    """

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(
        self,
        grid_size=64,
        lattice_type="square",
        threshold=None,
        walk_type="all",
        n_walks=0,
        n_steps=0,
        beta=None,
        tau0=None,
        noise=None,
        random_seed=None,
        n_jobs=-1,
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

    def run(self):
        """

        Parameters
        ----------
        None

        Returns
        -------

        Notes
        -----
        See http://dx.doi.org/10.1088/1751-8113/47/13/135001
        for details on thresholds for percolation

        """
        lattice_types = {"square": 0, "honeycomb": 1}
        lattice_thresholds = {"square": 0.592746, "honeycomb": 0.697040230}
        walk_types = {"all": 0, "largest": 1}

        self.lattice_type_ = lattice_types.get(self.lattice_type, None)

        if self.lattice_type_ is None:
            raise ValueError(
                f"Invalid lattice type: got '{self.lattice_type}' "
                f"instead of one of {lattice_types.keys()}"
            )

        self.walk_type_ = walk_types.get(self.walk_type, None)

        if self.walk_type_ is None:
            raise ValueError(
                f"Invalid walk type: got '{self.walk_type}' "
                f"instead of one of {walk_types.keys()}"
            )

        # If no threshold given, use the critical values
        # for the given lattice type
        self.threshold_ = (
            lattice_thresholds.get(self.lattice_type, 0.0)
            if self.threshold is None
            else self.threshold
        )

        # C++ uses numerical values instead of None
        self.beta_ = 0.0 if self.beta is None else self.beta
        self.tau0_ = 1.0 if self.tau0 is None else self.tau0
        self.noise_ = 0.0 if self.noise is None else self.noise
        self.random_seed_ = -1 if self.random_seed is None else self.random_seed

        res = ctrw_fractal(
            grid_size=self.grid_size,
            n_walks=self.n_walks,
            n_steps=self.n_steps,
            threshold=self.threshold_,
            beta=self.beta_,
            tau0=self.tau0_,
            noise=self.noise_,
            lattice_type=self.lattice_type_,
            walk_type=self.walk_type_,
            random_seed=self.random_seed_,
            n_jobs=self.n_jobs,
        )

        self.clusters_ = res[0]
        self.lattice_ = res[1]
        self.walks_ = res[2]
        self.analysis_ = res[3]

        return self
