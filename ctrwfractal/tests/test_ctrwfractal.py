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
import pytest

from ctrwfractal import CTRWfractal


class TestCTRWFractal:
    def setup_method(self, method):
        self.seed = 123
        self.size = 32

    def test_square(self):
        est = CTRWfractal(grid_size=self.size, lattice_type=0, random_seed=self.seed)
        est.run()

        for attr in ["lattice_", "clusters_", "analysis_", "walks_"]:
            assert hasattr(est, attr)

        assert est.clusters_.shape == (self.size * self.size,)

    def test_honeycomb(self):
        est = CTRWfractal(grid_size=self.size, lattice_type=1, random_seed=self.seed)
        est.run()

        for attr in ["lattice_", "clusters_", "analysis_", "walks_"]:
            assert hasattr(est, attr)

        assert est.clusters_.shape == (4 * self.size * self.size,)
