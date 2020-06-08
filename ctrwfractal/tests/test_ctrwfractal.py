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

import numpy as np
import pandas as pd
import pytest

from ctrwfractal import CTRWfractal


class TestSquare:
    def setup_method(self, method):
        self.seed = 123
        self.grid_size = 32

    @pytest.mark.parametrize("threshold", [None, 0.55, 0.65])
    def test_square_no_walks(self, threshold):
        est = CTRWfractal(
            grid_size=self.grid_size,
            lattice_type="square",
            threshold=threshold,
            random_seed=self.seed,
        )
        est.run()

        for attr in ["lattice_", "clusters_", "analysis_", "walks_"]:
            assert hasattr(est, attr)

        assert est.clusters_.shape == (self.grid_size * self.grid_size,)
        assert est.walks_ is None
        assert est.analysis_ is None

        # Check occupied fraction
        expected_threshold = 0.592746 if threshold is None else threshold
        np.testing.assert_allclose(
            expected_threshold, est.occupied_fraction_, atol=5e-3,
        )

    @pytest.mark.parametrize("walk_type", ["all", "largest"])
    @pytest.mark.parametrize("n_walks", [1, 2])
    @pytest.mark.parametrize("n_steps", [10, 25])
    def test_square_with_walks(self, walk_type, n_walks, n_steps):
        est = CTRWfractal(
            grid_size=self.grid_size,
            lattice_type="square",
            walk_type=walk_type,
            n_walks=n_walks,
            n_steps=n_steps,
            random_seed=self.seed,
        )
        est.run()

        for attr in ["lattice_", "clusters_", "analysis_", "walks_"]:
            assert hasattr(est, attr)

        assert est.clusters_.shape == (self.grid_size * self.grid_size,)
        assert isinstance(est.walks_, np.ndarray)
        assert isinstance(est.analysis_, pd.DataFrame)


class TestHoneycomb:
    def setup_method(self, method):
        self.seed = 123
        self.grid_size = 32

    @pytest.mark.parametrize("threshold", [None, 0.65, 0.75])
    def test_honeycomb_no_walks(self, threshold):
        est = CTRWfractal(
            grid_size=self.grid_size,
            lattice_type="honeycomb",
            threshold=threshold,
            random_seed=self.seed,
        )
        est.run()

        for attr in ["lattice_", "clusters_", "analysis_", "walks_"]:
            assert hasattr(est, attr)

        assert est.clusters_.shape == (4 * self.grid_size * self.grid_size,)
        assert est.walks_ is None
        assert est.analysis_ is None

        # Check occupied fraction
        expected_threshold = 0.697040230 if threshold is None else threshold
        np.testing.assert_allclose(
            expected_threshold, est.occupied_fraction_, atol=5e-3,
        )

    @pytest.mark.parametrize("walk_type", ["all", "largest"])
    @pytest.mark.parametrize("n_walks", [1, 2])
    @pytest.mark.parametrize("n_steps", [10, 25])
    def test_honeycomb_with_walks(self, walk_type, n_walks, n_steps):
        est = CTRWfractal(
            grid_size=self.grid_size,
            lattice_type="honeycomb",
            walk_type=walk_type,
            n_walks=n_walks,
            n_steps=n_steps,
            random_seed=self.seed,
        )
        est.run()

        for attr in ["lattice_", "clusters_", "analysis_", "walks_"]:
            assert hasattr(est, attr)

        assert est.clusters_.shape == (4 * self.grid_size * self.grid_size,)
        assert isinstance(est.walks_, np.ndarray)
        assert isinstance(est.analysis_, pd.DataFrame)


class TestErrors:
    def setup_method(self, method):
        self.seed = 123
        self.grid_size = 32

    def test_lattice_type_error(self):
        est = CTRWfractal(grid_size=self.grid_size, lattice_type=None)
        with pytest.raises(ValueError, match="Invalid lattice_type parameter"):
            est.run()

        est = CTRWfractal(grid_size=self.grid_size, lattice_type="triangle")
        with pytest.raises(ValueError, match="Invalid lattice_type parameter"):
            est.run()

    def test_walk_type_error(self):
        est = CTRWfractal(grid_size=self.grid_size, walk_type=None)
        with pytest.raises(ValueError, match="Invalid walk_type parameter"):
            est.run()

        est = CTRWfractal(grid_size=self.grid_size, walk_type="smallest")
        with pytest.raises(ValueError, match="Invalid walk_type parameter"):
            est.run()

    def test_threshold_error(self):
        est = CTRWfractal(grid_size=self.grid_size, threshold=-0.2)
        with pytest.raises(ValueError, match="Invalid threshold parameter"):
            est.run()

    def test_beta_error(self):
        est = CTRWfractal(grid_size=self.grid_size, beta=-0.2)
        with pytest.raises(ValueError, match="Invalid beta parameter"):
            est.run()

    def test_tau0_error(self):
        est = CTRWfractal(grid_size=self.grid_size, tau0=-0.2)
        with pytest.raises(ValueError, match="Invalid tau0 parameter"):
            est.run()

    def test_noise_error(self):
        est = CTRWfractal(grid_size=self.grid_size, noise=-0.2)
        with pytest.raises(ValueError, match="Invalid noise parameter"):
            est.run()
