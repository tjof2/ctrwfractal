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

import hashlib

import numpy as np
import pandas as pd
import pytest

from ctrwfractal import CTRWfractal


def _hash_ndarray(arr, n_char=-1):
    """Simple function to hash a np.ndarray object."""
    return hashlib.sha256(arr.data.tobytes()).hexdigest()[:n_char]


class TestSquare:
    def setup_method(self, method):
        self.seed = 123
        self.grid_size = 32

    @pytest.mark.parametrize(
        "threshold, expected_threshold, expected_clusters_hash",
        [
            (None, 0.592746, "a77f4546"),
            (0.55, 0.55, "079d302b"),
            (0.65, 0.65, "e16909db"),
        ],
    )
    def test_square_no_walks(
        self, threshold, expected_threshold, expected_clusters_hash
    ):
        s = CTRWfractal(
            grid_size=self.grid_size,
            lattice_type="square",
            threshold=threshold,
            random_seed=self.seed,
        )
        s.run()

        for attr in [
            "lattice_",
            "clusters_",
            "analysis_",
            "walks_",
            "occupied_fraction_",
        ]:
            assert hasattr(s, attr)

        assert s.clusters_.shape == (self.grid_size * self.grid_size,)
        assert s.walks_ is None
        assert s.analysis_ is None

        # Check occupied fraction
        np.testing.assert_allclose(
            expected_threshold, s.occupied_fraction_, atol=5e-3,
        )

        # Check clusters hash (first 8 characters)
        assert _hash_ndarray(s.clusters_, 8) == expected_clusters_hash

    @pytest.mark.parametrize("walk_type", ["all", "largest"])
    @pytest.mark.parametrize("n_walks", [1, 2])
    @pytest.mark.parametrize("n_steps", [10, 25])
    def test_square_with_walks(self, walk_type, n_walks, n_steps):
        s = CTRWfractal(
            grid_size=self.grid_size,
            lattice_type="square",
            walk_type=walk_type,
            n_walks=n_walks,
            n_steps=n_steps,
            random_seed=self.seed,
        )
        s.run()

        assert isinstance(s.walks_, np.ndarray)
        assert isinstance(s.analysis_, pd.DataFrame)

        assert s.walks_.shape == (n_walks, n_steps, 2)
        assert s.analysis_.shape == (n_steps - 1, n_walks + 3)


class TestHoneycomb:
    def setup_method(self, method):
        self.seed = 123
        self.grid_size = 32

    @pytest.mark.parametrize(
        "threshold, expected_threshold, expected_clusters_hash",
        [
            (None, 0.697040230, "c24af7ff"),
            (0.65, 0.65, "9c13bf6b"),
            (0.75, 0.75, "7d1c6d3d"),
        ],
    )
    def test_honeycomb_no_walks(
        self, threshold, expected_threshold, expected_clusters_hash
    ):
        s = CTRWfractal(
            grid_size=self.grid_size,
            lattice_type="honeycomb",
            threshold=threshold,
            random_seed=self.seed,
        )
        s.run()

        for attr in [
            "lattice_",
            "clusters_",
            "analysis_",
            "walks_",
            "occupied_fraction_",
        ]:
            assert hasattr(s, attr)

        assert s.clusters_.shape == (4 * self.grid_size * self.grid_size,)
        assert s.walks_ is None
        assert s.analysis_ is None

        # Check occupied fraction
        np.testing.assert_allclose(
            expected_threshold, s.occupied_fraction_, atol=5e-3,
        )

        # Check clusters hash (first 8 characters)
        assert _hash_ndarray(s.clusters_, 8) == expected_clusters_hash

    @pytest.mark.parametrize("walk_type", ["all", "largest"])
    @pytest.mark.parametrize("n_walks", [1, 2])
    @pytest.mark.parametrize("n_steps", [10, 25])
    def test_honeycomb_with_walks(self, walk_type, n_walks, n_steps):
        s = CTRWfractal(
            grid_size=self.grid_size,
            lattice_type="honeycomb",
            walk_type=walk_type,
            n_walks=n_walks,
            n_steps=n_steps,
            random_seed=self.seed,
        )
        s.run()

        assert isinstance(s.walks_, np.ndarray)
        assert isinstance(s.analysis_, pd.DataFrame)

        assert s.walks_.shape == (n_walks, n_steps, 2)
        assert s.analysis_.shape == (n_steps - 1, n_walks + 3)


class TestErrors:
    def setup_method(self, method):
        self.seed = 123
        self.grid_size = 32

    def test_lattice_type_error(self):
        s = CTRWfractal(grid_size=self.grid_size, lattice_type=None)
        with pytest.raises(ValueError, match="Invalid lattice_type parameter"):
            s.run()

        s = CTRWfractal(grid_size=self.grid_size, lattice_type="triangle")
        with pytest.raises(ValueError, match="Invalid lattice_type parameter"):
            s.run()

    def test_walk_type_error(self):
        s = CTRWfractal(grid_size=self.grid_size, walk_type=None)
        with pytest.raises(ValueError, match="Invalid walk_type parameter"):
            s.run()

        s = CTRWfractal(grid_size=self.grid_size, walk_type="smallest")
        with pytest.raises(ValueError, match="Invalid walk_type parameter"):
            s.run()

    def test_threshold_error(self):
        s = CTRWfractal(grid_size=self.grid_size, threshold=-0.2)
        with pytest.raises(ValueError, match="Invalid threshold parameter"):
            s.run()

    def test_beta_error(self):
        s = CTRWfractal(grid_size=self.grid_size, beta=-0.2)
        with pytest.raises(ValueError, match="Invalid beta parameter"):
            s.run()

    def test_tau0_error(self):
        s = CTRWfractal(grid_size=self.grid_size, tau0=-0.2)
        with pytest.raises(ValueError, match="Invalid tau0 parameter"):
            s.run()

    def test_noise_error(self):
        s = CTRWfractal(grid_size=self.grid_size, noise=-0.2)
        with pytest.raises(ValueError, match="Invalid noise parameter"):
            s.run()
