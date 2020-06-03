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

from ._ctrwfractal import ctrw_fractal


class CTRWfractal:
    def __init__(
        self,
        gridSize=64,
        nWalks=1,
        walkLength=10,
        threshold=None,
        beta=1.0,
        tau0=1.0,
        noise=0.0,
        latticeMode=0,
        walkMode=0,
        randomSeed=1,
        nJobs=-1,
    ):
        self.gridSize = gridSize
        self.nWalks = nWalks
        self.walkLength = walkLength
        self.threshold = threshold
        self.beta = beta
        self.tau0 = tau0
        self.noise = noise
        self.latticeMode = latticeMode
        self.walkMode = walkMode
        self.randomSeed = randomSeed
        self.nJobs = nJobs

    def run(self):
        # See http://dx.doi.org/10.1088/1751-8113/47/13/135001
        # for details on thresholds for percolation:
        #   - Square:     0.592746
        #   - Honeycomb:  0.697040230

        if self.threshold is None:
            if self.latticeMode == 1:
                fraction = 0.697040230
            elif self.latticeMode == 0:
                fraction = 0.592746
        else:
            fraction = self.threshold

        lattice, analysis, walks, result = ctrw_fractal(
            gridSize=self.gridSize,
            nWalks=self.nWalks,
            walkLength=self.walkLength,
            threshold=fraction,
            beta=self.beta,
            tau0=self.tau0,
            noise=self.noise,
            latticeMode=self.latticeMode,
            walkMode=self.walkMode,
            randomSeed=self.randomSeed,
            nJobs=self.nJobs,
        )

        return lattice, analysis, walks, result
