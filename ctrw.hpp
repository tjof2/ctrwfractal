/***************************************************************************

  Copyright 2016-2020 Tom Furnival

  This file is part of CTRWfractal.

  CTRWfractal is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  CTRWfractal is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with CTRWfractal.  If not, see <http://www.gnu.org/licenses/>.

  Percolation clusters developed from C code by Mark Newman.
  http://www-personal.umich.edu/~mejn/percolation/
  "A fast Monte Carlo algorithm for site or bond percolation"
  M. E. J. Newman and R. M. Ziff, Phys. Rev. E 64, 016706 (2001).

***************************************************************************/

#ifndef CTRW_HPP
#define CTRW_HPP

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <armadillo>
#include <omp.h>

#include "pcg/pcg_random.hpp"

template <class T>
class CTRWfractal
{
public:
  CTRWfractal(){};
  ~CTRWfractal(){};

  void Initialize(int size, double pc, int rngseed, std::string type,
                  int nwalks, int nsteps, double power_beta, double power_tau,
                  double walk_noise, int walk_type)
  {

#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(4);
#endif

    threshold = pc;
    n_walks = nwalks;
    walk_length = nsteps;
    beta = power_beta;
    tau0 = power_tau;
    sim_length = (tau0 < 1.) ? static_cast<int>(walk_length / tau0) : walk_length;
    noise = walk_noise;
    walk_mode = walk_type;

    L = size; // Get dimensions

    auto tStart = std::chrono::high_resolution_clock::now();
    std::cout << "Searching neighbours...    ";

    if (type.compare("Honeycomb") == 0)
    {
      lattice_mode = 1;
      nearest = 3;
      N = L * L * 4;
      nn.set_size(nearest, N);
      first_row.set_size(2 * L);
      last_row.set_size(2 * L);
      for (int i = 1; i <= 2 * L; i++)
      {
        first_row(i - 1) = 1 - (3 * L) / 2. + (std::pow(-1, i) * L) / 2. + 2 * i * L - 1;
        last_row(i - 1) = L / 2 * (4 * i + std::pow(-1, i + 1) - 1) - 1;
      }
      BoundariesHoneycomb();
    }
    else if (type.compare("Square") == 0)
    {
      lattice_mode = 0;
      nearest = 4;
      N = L * L;
      nn.set_size(nearest, N);
      BoundariesSquare();
    }
    else
    {
      std::cerr << "ERROR: " << type.c_str() << " must be either 'Square' or 'Honeycomb'" << std::endl;
      return;
    }

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tElapsed = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    std::cout << std::setprecision(6) << tElapsed.count() * 1E-6 << " s" << std::endl;

    EMPTY = (-N - 1);    // Define empty index
    lattice.set_size(N); // Set array sizes
    occupation.set_size(N);
    lattice_coords.set_size(3, N);
    walks.set_size(sim_length);
    ctrw_times.set_size(sim_length);
    true_walks.set_size(walk_length);
    walks_coords.set_size(2, walk_length, n_walks);
    eaMSD.set_size(walk_length);
    eaMSD_all.set_size(walk_length - 1, n_walks);
    taMSD.set_size(walk_length - 1, n_walks);
    eataMSD.set_size(walk_length - 1);
    eataMSD_all.set_size(walk_length - 1, n_walks);
    ergodicity.set_size(walk_length - 1);
    analysis.set_size(walk_length - 1, n_walks + 3);

    RNG = SeedRNG(rngseed); // Seed the generator

    return;
  }

  void Run()
  {

    std::cout << "Randomizing occupations... ";
    auto tStart = std::chrono::high_resolution_clock::now();
    Permutation(); // Randomize the order in which the sites are occupied
    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tElapsed = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    std::cout << std::setprecision(6) << tElapsed.count() * 1E-6 << " s" << std::endl;

    std::cout << "Running percolation...     ";
    tStart = std::chrono::high_resolution_clock::now();
    Percolate(); // Now run the percolation algorithm
    tEnd = std::chrono::high_resolution_clock::now();
    tElapsed = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    std::cout << std::setprecision(6) << tElapsed.count() * 1E-6 << " s" << std::endl;

    std::cout << "Building lattice...        ";
    tStart = std::chrono::high_resolution_clock::now();
    BuildLattice(); // Now build the lattice coordinates
    tEnd = std::chrono::high_resolution_clock::now();
    tElapsed = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    std::cout << std::setprecision(6) << tElapsed.count() * 1E-6 << " s" << std::endl;

    if (n_walks > 0) // Now run the random walks and analyse
    {
      std::cout << std::endl;
      std::cout << "Simulating random walks... ";
      tStart = std::chrono::high_resolution_clock::now();
      RandomWalks();
      tEnd = std::chrono::high_resolution_clock::now();
      tElapsed = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
      std::cout << std::setprecision(6) << tElapsed.count() * 1E-6 << " s" << std::endl;

      // Add noise to walk
      if (noise > 0.)
      {
        std::cout << "Adding noise...            ";
        tStart = std::chrono::high_resolution_clock::now();
        AddNoise();
        tEnd = std::chrono::high_resolution_clock::now();
        tElapsed = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
        std::cout << std::setprecision(6) << tElapsed.count() * 1E-6 << " s" << std::endl;
      }

      std::cout << "Analysing random walks...  ";
      tStart = std::chrono::high_resolution_clock::now();
      AnalyseWalks();
      tEnd = std::chrono::high_resolution_clock::now();
      tElapsed = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
      std::cout << std::setprecision(6) << tElapsed.count() * 1E-6 << " s" << std::endl;
    }
    return;
  }

  void Save(std::string filename)
  {
    std::cout << std::endl
              << "Saving files: " << std::endl;
    lattice_coords.save(filename + ".cluster", arma::raw_binary);
    std::cout << "   Cluster saved to:    " << filename << ".cluster"
              << std::endl;
    walks_coords.save(filename + ".walks", arma::raw_binary);
    std::cout << "   Walks saved to:      " << filename << ".walks"
              << std::endl;
    analysis.save(filename + ".data", arma::raw_binary);
    std::cout << "   Analysis saved to:   " << filename << ".data" << std::endl;
    return;
  }

private:
  int L, N, EMPTY, lattice_mode, n_walks, walk_length, sim_length;
  int nearest, walk_mode;

  arma::Col<T> lattice, occupation, walks, true_walks, first_row, last_row;
  arma::Mat<T> nn;
  arma::vec unit_cell, ctrw_times, eaMSD, eataMSD, ergodicity;
  arma::mat lattice_coords, eaMSD_all, eataMSD_all, taMSD;
  arma::mat analysis;
  arma::cube walks_coords;

  double threshold, beta, tau0, tElapsed, noise;
  const double sqrt3 = 1.7320508075688772;

  pcg64 RNG;
  std::uniform_int_distribution<uint32_t> UniformDistribution{0, 4294967294};

  void AnalyseWalks()
  {
    // Zero the placeholders
    eaMSD.zeros();
    eaMSD_all.zeros();
    taMSD.zeros();
    eataMSD.zeros();
    eataMSD_all.zeros();
    ergodicity.zeros();

// Parallelize over n_walks
#pragma omp parallel for
    for (int i = 0; i < n_walks; i++)
    {
      arma::vec2 walk_origin, walk_step;
      walk_origin = walks_coords.slice(i).col(0);
      for (int j = 1; j < walk_length; j++)
      {
        // Ensemble-average MSD
        walk_step = walks_coords.slice(i).col(j);
        eaMSD_all(j - 1, i) = std::pow(walk_step(0) - walk_origin(0), 2) +
                              std::pow(walk_step(1) - walk_origin(1), 2);
        // Time-average MSD
        taMSD(j - 1, i) = TAMSD(walks_coords.slice(i), walk_length, j);

        // Ensemble-time-average MSD
        eataMSD_all(j - 1, i) = TAMSD(walks_coords.slice(i), j, 1);
      }
    }
    // Check for NaNs
    eaMSD.elem(arma::find_nonfinite(eaMSD)).zeros();
    taMSD.elem(arma::find_nonfinite(taMSD)).zeros();
    eaMSD_all.elem(arma::find_nonfinite(eataMSD_all)).zeros();

    // Take means
    eaMSD = arma::mean(eaMSD_all, 1);
    eataMSD = arma::mean(eataMSD_all, 1);

    // Another check for NaNs
    eataMSD.elem(arma::find_nonfinite(eataMSD)).zeros();

    // Ergodicity breaking over s
    arma::mat mean_taMSD = arma::square(arma::mean(taMSD, 1));
    arma::mat mean_taMSD2 = arma::mean(arma::square(taMSD), 1);
    ergodicity = (mean_taMSD2 - mean_taMSD) / mean_taMSD;
    ergodicity.elem(arma::find_nonfinite(ergodicity)).zeros();
    ergodicity /= arma::regspace<arma::vec>(1, walk_length - 1);
    ergodicity.elem(arma::find_nonfinite(ergodicity)).zeros();

    analysis.col(0) = eaMSD;
    analysis.col(1) = eataMSD;
    analysis.col(2) = ergodicity;
    analysis.cols(3, n_walks + 2) = taMSD;

    return;
  }

  double TAMSD(const arma::mat &walk, int t, int delta)
  {
    double integral = 0.;
    int diff = t - delta;
    for (int i = 0; i < diff; i++)
    {
      integral += std::pow(walk(0, i + delta) - walk(0, i), 2) +
                  std::pow(walk(1, i + delta) - walk(1, i), 2);
    }
    return integral / diff;
  }

  void AddNoise()
  {
    // Add noise to walk
    arma::cube noise_cube(size(walks_coords));
    std::normal_distribution<double> NormalDistribution(0, noise);
    noise_cube.imbue([&]() { return NormalDistribution(RNG); });
    walks_coords += noise_cube;
    return;
  }

  void RandomWalks()
  {
    arma::Col<T> latticeones;

    // Set up selection of random start point
    //  - on largest cluster, or
    //  - on ALL clusters
    if (walk_mode == 1)
    {
      T lattice_min = lattice.elem(find(lattice > EMPTY)).min();
      arma::uvec index_min = arma::find(lattice == lattice_min);
      arma::uvec biggest_cluster = arma::find(lattice == index_min(0));
      int size_biggest_cluster = biggest_cluster.n_elem;
      size_biggest_cluster++;
      biggest_cluster.resize(size_biggest_cluster);
      biggest_cluster(size_biggest_cluster - 1) = index_min(0);
      latticeones = arma::regspace<arma::Col<T>>(0, N - 1);
      latticeones = latticeones.elem(biggest_cluster);
    }
    else
    {
      latticeones = arma::regspace<arma::Col<T>>(0, N - 1);
      latticeones = latticeones.elem(find(lattice != EMPTY));
    }
    std::uniform_int_distribution<T> RandSample(
        0, static_cast<int>(latticeones.n_elem) - 1);

    arma::uvec boundary_detect(sim_length);
    arma::uvec true_boundary(sim_length);

    for (int i = 0; i < n_walks; i++) // Simulate a random walk on the lattice
    {
      bool ok_start = false;
      int pos;
      int count_loop = 0;
      int count_max = (N > 1E6) ? N : 1E6;

      do // Search for a random start position
      {
        pos = latticeones(RandSample(RNG));
        // Check start position has >= 1 occupied nearest neighbours
        arma::Col<T> neighbours = GetOccupiedNeighbours(pos);
        if (neighbours.n_elem > 0 || count_loop >= count_max)
        {
          ok_start = true;
        }
        else
        {
          count_loop++;
        }
      } while (!ok_start);

      // If stuck on a site with no nearest neighbours,
      // set the whole walk to that site
      if (count_loop == count_max)
      {
        walks = pos * arma::ones<arma::Col<T>>(sim_length);
        boundary_detect.zeros();
      }
      else
      {
        walks(0) = pos;
        boundary_detect(0) = 0;
        for (int j = 1; j < sim_length; j++)
        {
          arma::Col<T> neighbours = GetOccupiedNeighbours(pos);
          std::uniform_int_distribution<T> RandChoice(0, static_cast<int>(neighbours.n_elem) - 1);
          pos = neighbours(RandChoice(RNG));
          walks(j) = pos;

          if (arma::any(first_row == walks(j - 1)) &&
              arma::any(last_row == pos)) // Walks that hit the top boundary
          {
            boundary_detect(j) = 1;
          }

          else if (arma::any(last_row == walks(j - 1)) &&
                   arma::any(first_row == pos)) // Walks that hit the bottom boundary
          {
            boundary_detect(j) = 2;
          }
          else if (walks(j - 1) >= (N - L) && pos < L) // Walks that hit the RHS
          {
            boundary_detect(j) = 3;
          }
          else if (walks(j - 1) < L && pos >= (N - L)) // Walks that hit the LHS
          {
            boundary_detect(j) = 4;
          }

          else
          {
            boundary_detect(j) = 0;
          }
        }
      }

      ctrw_times.set_size(sim_length);
      if (beta > 0.)
      {
        // Draw CTRW variates from exponential distribution
        std::exponential_distribution<double> ExponentialDistribution(beta);
        ctrw_times.imbue([&]() { return ExponentialDistribution(RNG); });

        // Transform to Pareto distribution and accumulate
        ctrw_times = arma::cumsum(tau0 * arma::exp(ctrw_times));
      }
      else
      {
        ctrw_times = arma::linspace<arma::vec>(1, sim_length, sim_length);
      }

      // Only keep times within range [0, walk_length]
      arma::uvec temp_time_boundary = arma::find(ctrw_times >= walk_length, 1, "first");
      int time_boundary = temp_time_boundary(0);
      ctrw_times = ctrw_times(arma::span(0, time_boundary));
      ctrw_times(time_boundary) = walk_length;

      int counter = 0;
      true_boundary.zeros();
      for (int j = 0; j < walk_length; j++) // Subordinate fractal walk with CTRW
      {
        if (j > ctrw_times(counter))
        {
          counter++;
          true_boundary(j) = boundary_detect(counter);
        }
        true_walks(j) = walks(counter);
      }

      int nx_cell = 0;
      int ny_cell = 0;
      for (int nstep = 0; nstep < walk_length; nstep++) // Convert the walk to the coordinate system
      {
        switch (true_boundary(nstep))
        {
        case 1:
          ny_cell++;
          break;
        case 2:
          ny_cell--;
          break;
        case 3:
          nx_cell++;
          break;
        case 4:
          nx_cell--;
          break;
        case 0:
        default:
          break;
        }
        walks_coords(0, nstep, i) =
            lattice_coords(0, true_walks(nstep)) + nx_cell * unit_cell(0);
        walks_coords(1, nstep, i) =
            lattice_coords(1, true_walks(nstep)) + ny_cell * unit_cell(1);
      }
    }
    return;
  }

  void BuildLattice()
  {
    if (lattice_mode == 1) // Populate the honeycomb lattice coordinates
    {
      double xx, yy;
      int count = 0;
      int cur_col = 0;
      for (int i = 0; i < 4 * L; i++)
      {
        for (int j = L - 1; j >= 0; j--)
        {
          cur_col = i % 4;
          switch (cur_col)
          {
          case 0:
          default:
            xx = i / 4 * 3;
            yy = j * sqrt3 + sqrt3 / 2;
            break;
          case 1:
            xx = i / 4 * 3 + 1. / 2;
            yy = j * sqrt3;
            break;
          case 2:
            xx = i / 4 * 3 + 3. / 2;
            yy = j * sqrt3;
            break;
          case 3:
            xx = i / 4 * 3 + 2.;
            yy = j * sqrt3 + sqrt3 / 2;
            break;
          }
          lattice_coords(0, count) = xx;
          lattice_coords(1, count) = yy;
          lattice_coords(2, count) =
              (lattice(count) == EMPTY) ? 0 : lattice(count);
          count++;
        }
      }

      unit_cell = arma::max(lattice_coords, 1); // Get unit cell size
      unit_cell(0) += 3 / 2;
      unit_cell(1) += sqrt3 / 2;
    }

    else if (lattice_mode == 0) // Populate the square lattice coordinates
    {
      int count = 0;
      for (int i = 0; i < L; i++)
      {
        for (int j = 0; j < L; j++)
        {
          lattice_coords(0, count) = i;
          lattice_coords(1, count) = j;
          lattice_coords(2, count) =
              (lattice(count) == EMPTY) ? 0 : lattice(count);
          count++;
        }
      }

      unit_cell = arma::max(lattice_coords, 1); // Get unit cell size
      unit_cell(0) += 1;
      unit_cell(1) += 1;
    }
    return;
  }

  arma::Col<T> GetOccupiedNeighbours(int pos)
  {
    arma::Col<T> neighbours = nn.col(pos);
    arma::Col<T> neighbour_check(3);
    for (int k = 0; k < nearest; k++)
    {
      neighbour_check(k) = (lattice(neighbours(k)) == EMPTY) ? 0 : 1;
    }
    neighbours = neighbours.elem(find(neighbour_check == 1));
    return neighbours;
  }

  void Permutation()
  {
    T j;
    T temp;

    for (int i = 0; i < N; i++)
    {
      occupation(i) = i;
    }
    for (int i = 0; i < N; i++)
    {
      j = i + (N - i) * 2.3283064e-10 * UniformDistribution(RNG);
      temp = occupation(i);
      occupation(i) = occupation(j);
      occupation(j) = temp;
    }
    return;
  }

  int FindRoot(int i)
  {
    if (lattice(i) < 0)
    {
      return i;
    }
    return lattice(i) = FindRoot(lattice(i));
  }

  void Percolate()
  {
    int s1, s2;
    int r1, r2;
    T big = 0;

    for (int i = 0; i < N; i++)
    {
      lattice(i) = EMPTY;
    }
    for (int i = 0; i < (threshold * N) - 1; i++)
    {
      r1 = s1 = occupation[i];
      lattice(s1) = -1;
      for (int j = 0; j < nearest; j++)
      {
        s2 = nn(j, s1);
        if (lattice(s2) != EMPTY)
        {
          r2 = FindRoot(s2);
          if (r2 != r1)
          {
            if (lattice(r1) > lattice(r2))
            {
              lattice(r2) += lattice(r1);
              lattice(r1) = r2;
              r1 = r2;
            }
            else
            {
              lattice(r1) += lattice(r2);
              lattice(r2) = r1;
            }
            if (-lattice(r1) > big)
            {
              big = -lattice(r1);
            }
          }
        }
      }
    }
    return;
  }

  // Nearest neighbours of a honeycomb lattice with
  // periodic boundary conditions
  void BoundariesHoneycomb()
  {
    int cur_col = 0;
    int count = 0;
    for (int i = 0; i < N; i++)
    {
      if (i == 0) // First site
      {
        nn(0, i) = i + L;
        nn(1, i) = i + 2 * L - 1;
        nn(2, i) = i + N - L;
      }
      else if (i == N - L) // Top right-hand corner
      {
        nn(0, i) = i - 1;
        nn(1, i) = i - L;
        nn(2, i) = i - N + L;
      }
      else if (i == N - L - 1) // Bottom right-hand corner
      {
        nn(0, i) = i - L;
        nn(1, i) = i + L;
        nn(2, i) = i + 1;
      }
      else if (i < L) // First column
      {
        nn(0, i) = i + L - 1;
        nn(1, i) = i + L;
        nn(2, i) = i + N - L;
      }
      else if (i > (N - L)) // Last column
      {
        nn(0, i) = i - L - 1;
        nn(1, i) = i - L;
        nn(2, i) = i - N + L;
      }
      else // Run through the rest of the tests
      {
        switch (cur_col)
        {
        case 0:
          if (arma::any(first_row == i)) // First row
          {
            nn(0, i) = i - L;
            nn(1, i) = i + L;
            nn(2, i) = i + 2 * L - 1;
          }
          else
          {
            nn(0, i) = i - L;
            nn(1, i) = i + L - 1;
            nn(2, i) = i + L;
          }
          break;
        case 1:
          if (arma::any(last_row == i)) // Last row
          {
            nn(0, i) = i - L;
            nn(1, i) = i + L;
            nn(2, i) = i - 2 * L + 1;
          }
          else
          {
            nn(0, i) = i - L;
            nn(1, i) = i - L + 1;
            nn(2, i) = i + L;
          }
          break;
        case 2:
          if (arma::any(last_row == i)) // Last row
          {
            nn(0, i) = i - L;
            nn(1, i) = i + L;
            nn(2, i) = i + 1;
          }
          else
          {
            nn(0, i) = i - L;
            nn(1, i) = i + L;
            nn(2, i) = i + L + 1;
          }
          break;
        case 3:

          if (arma::any(first_row == i)) // First row
          {
            nn(0, i) = i - 1;
            nn(1, i) = i - L;
            nn(2, i) = i + L;
          }
          else
          {
            nn(0, i) = i - L - 1;
            nn(1, i) = i - L;
            nn(2, i) = i + L;
          }
          break;
        }
      }

      if ((i + 1) % L == 0) // Update current column
      {
        count++;
        cur_col = count % 4;
      }
    }
    return;
  }

  // Nearest neighbours of a square lattice
  // with periodic boundary conditions
  void BoundariesSquare()
  {
    for (int i = 0; i < N; i++)
    {
      nn(0, i) = (i + 1) % N;
      nn(1, i) = (i + N - 1) % N;
      nn(2, i) = (i + L) % N;
      nn(3, i) = (i + N - L) % N;
      if (i % L == 0)
      {
        nn(1, i) = i + L - 1;
      }
      if ((i + 1) % L == 0)
      {
        nn(0, i) = i - L + 1;
      }
    }
    return;
  }

  pcg64 SeedRNG(int seed)
  {
    if (seed > 0) // Check for user-defined seed
    {
      return pcg64(seed);
    }
    else // Initialize random seed
    {
      pcg_extras::seed_seq_from<std::random_device> seed_source;
      return pcg64(seed_source);
    }
  }
};

#endif
