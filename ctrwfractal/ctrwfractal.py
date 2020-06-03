# -*- coding: utf-8 -*-
# Copyright 2016-2020 Tom Furnival
#
# This file is part of CTRWfractal.
#
# CTRWfractal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CTRWfractal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CTRWfractal.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from ._ctrwfractal import ctrw_fractal_double


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
        n_walks=1,
        n_steps=10,
        threshold=None,
        beta=1.0,
        tau0=1.0,
        noise=0.0,
        lattice_type=0,
        walk_type=0,
        random_seed=1,
        n_jobs=-1,
    ):
        self.grid_size = grid_size
        self.n_walks = n_walks
        self.n_steps = n_steps
        self.threshold = threshold
        self.beta = beta
        self.tau0 = tau0
        self.noise = noise
        self.lattice_type = lattice_type
        self.walk_type = walk_type
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
        for details on thresholds for percolation:
            - Square:     0.592746
            - Honeycomb:  0.697040230

        """

        if self.threshold is None:
            if self.lattice_type == 1:
                self.threshold_ = 0.697040230
            elif self.lattice_type == 0:
                self.threshold_ = 0.592746
        else:
            self.threshold_ = self.threshold

        res = ctrw_fractal_double(
            grid_size=self.grid_size,
            n_walks=self.n_walks,
            n_steps=self.n_steps,
            threshold=self.threshold_,
            beta=self.beta,
            tau0=self.tau0,
            noise=self.noise,
            lattice_type=self.lattice_type,
            walk_type=self.walk_type,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
        )
        self.lattice_ = res[0]
        self.analysis_ = res[1]
        self.walks_ = res[2]

        return self
