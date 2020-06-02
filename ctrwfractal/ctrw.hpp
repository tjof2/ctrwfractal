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
#include <thread>
#include <vector>

#include "pcg/pcg_random.hpp"
#include "utils.hpp"

template <typename T1, typename T2>
class CTRWfractal
{
public:
  CTRWfractal(
      arma::Mat<T2> lattice,
      arma::Cube<T2> walks,
      arma::Mat<T2> analysis,
      const uint32_t L,
      const uint32_t nWalks,
      const uint32_t walkLength,
      const double threshold,
      const double beta,
      const double tau0,
      const double noise,
      const uint8_t latticeMode,
      const uint8_t walkMode,
      const int nJobs) : latticeCoords(lattice),
                         walksCoords(walks),
                         analysis(analysis),
                         L(L),
                         nWalks(nWalks),
                         walkLength(walkLength),
                         threshold(threshold),
                         beta(beta),
                         tau0(tau0),
                         noise(noise),
                         latticeMode(latticeMode),
                         walkMode(walkMode),
                         nJobs(nJobs)
  {
    simLength = (tau0 < 1.) ? static_cast<uint32_t>(walkLength / tau0) : walkLength;

    walks.set_size(simLength); // Set array sizes
    ctrwTimes.set_size(simLength);
    trueWalks.set_size(walkLength);
    walksCoords.set_size(2, walkLength, nWalks);
    eaMSD.set_size(walkLength);
    eaMSDall.set_size(walkLength - 1, nWalks);
    taMSD.set_size(walkLength - 1, nWalks);
    eataMSD.set_size(walkLength - 1);
    eataMSDall.set_size(walkLength - 1, nWalks);
    ergodicity.set_size(walkLength - 1);
    analysis.set_size(walkLength - 1, nWalks + 3);
  };

  ~CTRWfractal(){};

  void Initialize(const uint64_t rngseed)
  {
    RNG = SeedRNG(rngseed); // Seed the generator

    auto tStart = std::chrono::high_resolution_clock::now();
    std::cout << "Searching neighbours...    ";

    if (latticeMode == 1)
    {
      neighbourCount = 3;
      N = L * L * 4;
      nn.set_size(neighbourCount, N);
      firstRow.set_size(2 * L);
      lastRow.set_size(2 * L);
      for (size_t i = 1; i <= 2 * L; i++)
      {
        firstRow(i - 1) = 1 - 0.5 * (3 * L) + 0.5 * (std::pow(-1, i) * L) + 2 * i * L - 1;
        lastRow(i - 1) = 0.5 * L * (4 * i + std::pow(-1, i + 1) - 1) - 1;
      }
      BoundariesHoneycomb();
    }
    else if (latticeMode == 0)
    {
      neighbourCount = 4;
      N = L * L;
      nn.set_size(neighbourCount, N);
      BoundariesSquare();
    }

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tElapsed = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    std::cout << std::setprecision(6) << tElapsed.count() * 1E-6 << " s" << std::endl;

    EMPTY = (-N - 1);    // Define empty index
    lattice.set_size(N); // Set array sizes
    occupation.set_size(N);
    latticeCoords.set_size(3, N);

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

    if (nWalks > 0) // Now run the random walks and analyse
    {
      std::cout << "Simulating random walks... ";
      tStart = std::chrono::high_resolution_clock::now();
      RandomWalks();
      tEnd = std::chrono::high_resolution_clock::now();
      tElapsed = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
      std::cout << std::setprecision(6) << tElapsed.count() * 1E-6 << " s" << std::endl;

      if (noise > 0.) // Add noise to walk
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

private:
  arma::Mat<T2> latticeCoords, analysis;
  arma::Cube<T2> walksCoords;

  uint32_t L, nWalks, walkLength;
  double threshold, beta, tau0, noise;
  uint8_t latticeMode, walkMode;
  int nJobs;

  uint32_t N, simLength;
  int64_t EMPTY;
  uint8_t neighbourCount;
  const double sqrt3 = 1.7320508075688772;
  const double sqrt3o2 = 0.8660254037844386;

  arma::Col<T1> lattice, occupation, walks, trueWalks, firstRow, lastRow;
  arma::Mat<T1> nn;
  arma::Col<T2> unitCell, ctrwTimes, eaMSD, eataMSD, ergodicity;
  arma::Mat<T2> eaMSDall, eataMSDall, taMSD;

  pcg64 RNG;
  std::uniform_int_distribution<uint32_t> UniformDistribution{0, 4294967294};

  inline double SquaredDist(const double &x1, const double &x2, const double &y1, const double &y2)
  {
    double a = (x1 - x2);
    double b = (y1 - y2);
    return a * a + b * b;
  }

  void AnalyseWalks()
  {
    eaMSD.zeros(); // Zero the placeholders
    eaMSDall.zeros();
    taMSD.zeros();
    eataMSD.zeros();
    eataMSDall.zeros();
    ergodicity.zeros();

    auto &&func = [&](uint32_t i) {
      arma::Col<double>::fixed<2> walkOrigin, walkStep;
      walkOrigin = walksCoords.slice(i).col(0);
      for (size_t j = 1; j < walkLength; j++)
      {
        walkStep = walksCoords.slice(i).col(j);
        eaMSDall(j - 1, i) = SquaredDist(walkStep(0), walkOrigin(0),
                                         walkStep(1), walkOrigin(1)); // Ensemble-average MSD
        taMSD(j - 1, i) = TAMSD(walksCoords.slice(i), walkLength, j); // Time-average MSD
        eataMSDall(j - 1, i) = TAMSD(walksCoords.slice(i), j, 1);     // Ensemble-time-average MSD
      }
    };
    parallel(func, static_cast<uint32_t>(0), static_cast<uint32_t>(nWalks), nJobs);

    // Check for NaNs
    eaMSD.elem(arma::find_nonfinite(eaMSD)).zeros();
    taMSD.elem(arma::find_nonfinite(taMSD)).zeros();
    eaMSDall.elem(arma::find_nonfinite(eataMSDall)).zeros();

    // Take means
    eaMSD = arma::mean(eaMSDall, 1);
    eataMSD = arma::mean(eataMSDall, 1);

    // Another check for NaNs
    eataMSD.elem(arma::find_nonfinite(eataMSD)).zeros();

    // Ergodicity breaking over s
    arma::Mat<T2> meanTAMSD = arma::square(arma::mean(taMSD, 1));
    arma::Mat<T2> meanTAMSD2 = arma::mean(arma::square(taMSD), 1);
    ergodicity = (meanTAMSD2 - meanTAMSD) / meanTAMSD;
    ergodicity.elem(arma::find_nonfinite(ergodicity)).zeros();
    ergodicity /= arma::regspace<arma::Col<T2>>(1, walkLength - 1);
    ergodicity.elem(arma::find_nonfinite(ergodicity)).zeros();

    analysis.col(0) = eaMSD;
    analysis.col(1) = eataMSD;
    analysis.col(2) = ergodicity;
    analysis.cols(3, nWalks + 2) = taMSD;

    return;
  }

  double TAMSD(const arma::Mat<T2> &walk, const int t, const int delta)
  {
    double integral = 0.;
    int diff = t - delta;
    for (int i = 0; i < diff; i++)
    {
      integral += SquaredDist(walk(0, i + delta), walk(0, i), walk(1, i + delta), walk(1, i));
    }
    return integral / diff;
  }

  void AddNoise()
  {
    arma::Cube<T2> noiseCube(size(walksCoords));
    std::normal_distribution<T2> NormalDistribution(0, noise);
    noiseCube.imbue([&]() { return NormalDistribution(RNG); });
    walksCoords += noiseCube;
    return;
  }

  void RandomWalks()
  {
    arma::Col<T1> latticeOnes;

    // Set up selection of random start point
    //  - on largest cluster, or
    //  - on ALL clusters
    if (walkMode == 1)
    {
      T1 latticeMin = lattice.elem(find(lattice > EMPTY)).min();
      arma::uvec idxMin = arma::find(lattice == latticeMin);
      arma::uvec largestCluster = arma::find(lattice == idxMin(0));
      int largestClusterSize = largestCluster.n_elem;
      largestClusterSize++;
      largestCluster.resize(largestClusterSize);
      largestCluster(largestClusterSize - 1) = idxMin(0);
      latticeOnes = arma::regspace<arma::Col<T1>>(0, N - 1);
      latticeOnes = latticeOnes.elem(largestCluster);
    }
    else
    {
      latticeOnes = arma::regspace<arma::Col<T1>>(0, N - 1);
      latticeOnes = latticeOnes.elem(find(lattice != EMPTY));
    }
    std::uniform_int_distribution<T1> RandSample(0, static_cast<uint32_t>(latticeOnes.n_elem) - 1);

    arma::uvec boundaryDetect(simLength);
    arma::uvec boundaryTrue(simLength);
    int64_t bound1 = static_cast<int64_t>(L);
    int64_t bound2 = static_cast<int64_t>(N) - bound1;

    for (size_t i = 0; i < nWalks; i++) // Simulate a random walk on the lattice
    {
      bool okStart = false;
      int pos;
      int countLoop = 0;
      int countMax = (N > 1E6) ? N : 1E6;

      do // Search for a random start position
      {
        pos = latticeOnes(RandSample(RNG));
        // Check start position has >= 1 occupied nearest neighbours
        arma::Col<T1> neighbours = GetOccupiedNeighbours(pos);
        if (neighbours.n_elem > 0 || countLoop >= countMax)
        {
          okStart = true;
        }
        else
        {
          countLoop++;
        }
      } while (!okStart);

      // If stuck on a site with no nearest neighbours,
      // set the whole walk to that site
      if (countLoop == countMax)
      {
        walks = pos * arma::ones<arma::Col<T1>>(simLength);
        boundaryDetect.zeros();
      }
      else
      {
        walks(0) = pos;
        boundaryDetect(0) = 0;
        for (size_t j = 1; j < simLength; j++)
        {
          arma::Col<T1> neighbours = GetOccupiedNeighbours(pos);
          std::uniform_int_distribution<T1> RandChoice(0, static_cast<uint32_t>(neighbours.n_elem) - 1);
          pos = neighbours(RandChoice(RNG));
          walks(j) = pos;

          if (arma::any(firstRow == walks(j - 1)) && arma::any(lastRow == pos)) // Walks that hit the top boundary
          {
            boundaryDetect(j) = 1;
          }
          else if (arma::any(lastRow == walks(j - 1)) && arma::any(firstRow == pos)) // Walks that hit the bottom boundary
          {
            boundaryDetect(j) = 2;
          }
          else if (walks(j - 1) >= bound2 && pos < bound1) // Walks that hit the RHS
          {
            boundaryDetect(j) = 3;
          }
          else if (walks(j - 1) < bound1 && pos >= bound2) // Walks that hit the LHS
          {
            boundaryDetect(j) = 4;
          }
          else
          {
            boundaryDetect(j) = 0;
          }
        }
      }

      ctrwTimes.set_size(simLength);
      if (beta > 0.)
      {
        // Draw CTRW variates from exponential distribution
        std::exponential_distribution<T2> ExponentialDistribution(beta);
        ctrwTimes.imbue([&]() { return ExponentialDistribution(RNG); });

        // Transform to Pareto distribution and accumulate
        ctrwTimes = arma::cumsum(tau0 * arma::exp(ctrwTimes));
      }
      else
      {
        ctrwTimes = arma::linspace<arma::Col<T2>>(1, simLength, simLength);
      }

      // Only keep times within range [0, walkLength]
      arma::uvec boundaryTime_ = arma::find(ctrwTimes >= walkLength, 1, "first");
      int boundaryTime = boundaryTime_(0);
      ctrwTimes = ctrwTimes(arma::span(0, boundaryTime));
      ctrwTimes(boundaryTime) = walkLength;

      int counter = 0;
      boundaryTrue.zeros();
      for (size_t j = 0; j < walkLength; j++) // Subordinate fractal walk with CTRW
      {
        if (j > ctrwTimes(counter))
        {
          counter++;
          boundaryTrue(j) = boundaryDetect(counter);
        }
        trueWalks(j) = walks(counter);
      }

      int nxCell = 0;
      int nyCell = 0;
      for (size_t nstep = 0; nstep < walkLength; nstep++) // Convert the walk to the coordinate system
      {
        switch (boundaryTrue(nstep))
        {
        case 1:
          nyCell++;
          break;
        case 2:
          nyCell--;
          break;
        case 3:
          nxCell++;
          break;
        case 4:
          nxCell--;
          break;
        case 0:
        default:
          break;
        }
        walksCoords(0, nstep, i) = latticeCoords(0, trueWalks(nstep)) + nxCell * unitCell(0);
        walksCoords(1, nstep, i) = latticeCoords(1, trueWalks(nstep)) + nyCell * unitCell(1);
      }
    }
    return;
  }

  void BuildLattice()
  {
    if (latticeMode == 1) // Populate the honeycomb lattice coordinates
    {
      double xx, yy;
      int count = 0;
      int curCol = 0;
      for (size_t i = 0; i < 4 * L; i++)
      {
        for (size_t j = L - 1; j >= 0; j--)
        {
          curCol = i % 4;
          switch (curCol)
          {
          case 0:
          default:
            xx = 0.25 * i * 3;
            yy = j * sqrt3 + sqrt3o2;
            break;
          case 1:
            xx = 0.25 * i * 3 + 0.5;
            yy = j * sqrt3;
            break;
          case 2:
            xx = 0.25 * i * 3 + 1.5;
            yy = j * sqrt3;
            break;
          case 3:
            xx = 0.25 * i * 3 + 2.0;
            yy = j * sqrt3 + sqrt3o2;
            break;
          }
          latticeCoords(0, count) = xx;
          latticeCoords(1, count) = yy;
          latticeCoords(2, count) = (lattice(count) == EMPTY) ? 0 : lattice(count);
          count++;
        }
      }

      unitCell = arma::max(latticeCoords, 1); // Get unit cell size
      unitCell(0) += 1.5;
      unitCell(1) += sqrt3o2;
    }
    else if (latticeMode == 0) // Populate the square lattice coordinates
    {
      int count = 0;
      for (size_t i = 0; i < L; i++)
      {
        for (size_t j = 0; j < L; j++)
        {
          latticeCoords(0, count) = i;
          latticeCoords(1, count) = j;
          latticeCoords(2, count) = (lattice(count) == EMPTY) ? 0 : lattice(count);
          count++;
        }
      }

      unitCell = arma::max(latticeCoords, 1); // Get unit cell size
      unitCell(0) += 1;
      unitCell(1) += 1;
    }
    return;
  }

  arma::Col<T1> GetOccupiedNeighbours(const int pos)
  {
    arma::Col<uint8_t> checkNeighbour(neighbourCount, arma::fill::zeros);
    arma::Col<T1> neighbours = nn.col(pos);
    for (size_t k = 0; k < neighbourCount; k++)
    {
      checkNeighbour(k) = (lattice(neighbours(k)) == EMPTY) ? 0 : 1;
    }
    neighbours = neighbours.elem(find(checkNeighbour == 1));
    return neighbours;
  }

  void Permutation()
  {
    T1 j;
    T1 temp;

    for (size_t i = 0; i < N; i++)
    {
      occupation(i) = i;
    }
    for (size_t i = 0; i < N; i++)
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
    T1 big = 0;

    for (size_t i = 0; i < N; i++)
    {
      lattice(i) = EMPTY;
    }
    for (int i = 0; i < (threshold * N) - 1; i++)
    {
      r1 = s1 = occupation[i];
      lattice(s1) = -1;
      for (size_t j = 0; j < neighbourCount; j++)
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
    int curCol = 0;
    int count = 0;
    for (size_t i = 0; i < N; i++)
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
        switch (curCol)
        {
        case 0:
          if (arma::any(firstRow == i)) // First row
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
          if (arma::any(lastRow == i)) // Last row
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
          if (arma::any(lastRow == i)) // Last row
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

          if (arma::any(firstRow == i)) // First row
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
        curCol = count % 4;
      }
    }
    return;
  }

  // Nearest neighbours of a square lattice
  // with periodic boundary conditions
  void BoundariesSquare()
  {
    for (size_t i = 0; i < N; i++)
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

  pcg64 SeedRNG(const uint64_t seed)
  {
    if (seed > 0) // User-defined seed
    {
      return pcg64(seed);
    }
    else // Initialize random seed
    {
      pcg_extras::seed_seq_from<std::random_device> seedSource;
      return pcg64(seedSource);
    }
  }
};

template <typename T>
uint32_t CTRWwrapper(
    arma::Mat<T> &lattice,
    arma::Cube<T> &walks,
    arma::Mat<T> &analysis,
    const uint32_t gridSize,
    const uint32_t nWalks,
    const uint32_t walkLength,
    const double threshold,
    const double beta,
    const double tau0,
    const double noise,
    const uint8_t latticeMode,
    const uint8_t walkMode,
    const int randomSeed,
    const int nJobs)
{
  CTRWfractal<int32_t, T> *sim = new CTRWfractal<int32_t, T>(
      lattice,
      walks,
      analysis,
      gridSize,
      nWalks,
      walkLength,
      threshold,
      beta,
      tau0,
      noise,
      latticeMode,
      walkMode,
      nJobs);

  sim->Initialize(randomSeed);
  sim->Run();

  return 0;
};

#endif
