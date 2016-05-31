/***************************************************************************

    Copyright (C) 2016 Tom Furnival
    Email: tjof2@cam.ac.uk

    This file is part of Percolation.

    Developed from C code by Mark Newman.
    http://www-personal.umich.edu/~mejn/percolation/
    "A fast Monte Carlo algorithm for site or bond percolation"
    M. E. J. Newman and R. M. Ziff, Phys. Rev. E 64, 016706 (2001).

***************************************************************************/

#ifndef MAIN_H
#define MAIN_H

// C++ headers
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <random>

// Armadillo
#include <armadillo>

// PCG RNG
#include "pcg/pcg_random.hpp"

template <class T>
class Percolation {
public:
  Percolation() {};
  ~Percolation() {};

  void Initialize(int size,
                  double pc,
                  int rngseed,
                  std::string type,
                  int numwalks,
                  int nsteps,
                  double userbeta,
                  int truelength) {
    // Initialize threshold
    threshold = pc;

    // Number of random walks (>0)
    N_walks = numwalks;
    walk_length = nsteps;
    beta = userbeta;
    true_length = truelength;

    // Get dimensions
    L = size;

    // Check mode
    std::cout<<"Searching neighbours...    ";
    if (type.compare("Honeycomb") == 0) {
      latticemode = 1;
      nearest = 3;
      N = L * L * 4;
      nn.set_size(nearest, N);
      firstrow.set_size(2 * L);
      lastrow.set_size(2 * L);
      for (int i = 1; i <= 2 * L; i++) {
        firstrow(i-1) = 1 - (3 * L) / 2. + (std::pow(-1, i) * L) / 2. + 2 * i * L - 1;
        lastrow(i-1) = L/2 * (4*i + std::pow(-1, i + 1) - 1) - 1;
      }
      time_start = GetTime();
      BoundariesHoneycomb();
      time_end = GetTime();
      run_time = (std::chrono::duration_cast<std::chrono::microseconds>(
        time_end - time_start).count()/1E6);
      std::cout<<std::setprecision(6)<<run_time<<" s"<<std::endl;
    }
    else if (type.compare("Square") == 0) {
      latticemode = 0;
      nearest = 4;
      N = L * L;
      nn.set_size(nearest, N);
      time_start = GetTime();
      BoundariesSquare();
    	time_end = GetTime();
  		run_time = (std::chrono::duration_cast<std::chrono::microseconds>(
        time_end - time_start).count()/1E6);
  		std::cout<<std::setprecision(6)<<run_time<<" s"<<std::endl;
    }
    else {
      std::cout << "!!! WARNING: "
                << type.c_str()
                << " must be either 'Square' or 'Honeycomb' !!!"
                << std::endl;
    }

    // Define empty index
    EMPTY = (-N - 1);

    // Set array sizes
    lattice.set_size(N);
    occupation.set_size(N);
    lattice_coordinates.set_size(3, N);
    walks.set_size(walk_length);
    true_walks.set_size(true_length);
    walks_coordinates.set_size(2, true_length, N_walks);
    ctrwTimes.set_size(walk_length);

    // Seed the generator
    RNG = SeedRNG(rngseed);
    return;
  }

  void Run() {
    // First randomise the order in which the
    // sites are occupied
    std::cout<<"Randomising occupations... ";
		time_start = GetTime();
    Permutation();
  	time_end = GetTime();
		run_time = (std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start).count()/1E6);
		std::cout<<std::setprecision(6)<<run_time<<" s"<<std::endl;

    // Now run the percolation algorithm
    std::cout<<"Running percolation...     ";
    time_start = GetTime();
    Percolate();
  	time_end = GetTime();
		run_time = (std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start).count()/1E6);
		std::cout<<std::setprecision(6)<<run_time<<" s"<<std::endl;

    // Now build the lattice coordinates
    std::cout<<"Building lattice...        ";
    time_start = GetTime();
    BuildLattice();
  	time_end = GetTime();
		run_time = (std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start).count()/1E6);
		std::cout<<std::setprecision(6)<<run_time<<" s"<<std::endl;

    // Now run the random walks
    if (N_walks > 0) {
      std::cout<<std::endl;
      std::cout<<"Simulating random walks... ";
      time_start = GetTime();
      RandomWalks();
      time_end = GetTime();
      run_time = (std::chrono::duration_cast<std::chrono::microseconds>(
        time_end - time_start).count()/1E6);
      std::cout<<std::setprecision(6)<<run_time<<" s"<<std::endl;
    }

    return;
  }

  void Save(std::string filename) {
    std::cout<<std::endl<<"Saving files: "<<std::endl;
    lattice_coordinates.save(filename + ".cluster", arma::raw_binary);
    std::cout<<"   Cluster saved to: "<<filename<<".cluster"<<std::endl;
    walks_coordinates.save(filename + ".walks", arma::raw_binary);
    std::cout<<"   Walks saved to:   "<<filename<<".walks"<<std::endl;
    return;
  }

private:
  // Dimensions
  int L, N, EMPTY, latticemode, N_walks, walk_length, true_length;

  // Number of nearest neighbours
  int nearest;

  // Arrays to hold information
  arma::Col<T> lattice, occupation, walks, true_walks;
  arma::Mat<T> nn;
  arma::vec ctrwTimes;
  arma::mat lattice_coordinates;
  arma::cube walks_coordinates;

  arma::colvec unit_cell;

  // Honeycomb lattice only - check for first or last row
  arma::Col<T> firstrow, lastrow;

  // Percolation threshold
  double threshold;

  // Power-law beta
  double beta;

  const double sqrt3 = 1.7320508075688772;

  // Timing
  double run_time;
	#if __cplusplus <= 199711L
	 std::chrono::time_point<std::chrono::monotonic_clock> time_start, time_end;
   std::chrono::time_point<std::chrono::monotonic_clock> GetTime() {
     return std::chrono::monotonic_clock::now();
   }
	#else
	 std::chrono::time_point<std::chrono::steady_clock> time_start, time_end;
   std::chrono::time_point<std::chrono::steady_clock> GetTime() {
     return std::chrono::steady_clock::now();
   }
	#endif

  // Random numbers
  pcg64 RNG;
  std::uniform_int_distribution<uint32_t> UniformDistribution {0, 4294967294};

  pcg64 SeedRNG(int seed) {
    // Check for user-defined seed
    if(seed > 0) {
      return pcg64(seed);
    }
    else {
      // Initialize random seed
      pcg_extras::seed_seq_from<std::random_device> seed_source;
      return pcg64(seed_source);
    }
  }

  void RandomWalks() {
    // Set up selection of random start point
    arma::Col<T> latticeones = arma::regspace<arma::Col<T>>(0, N - 1);
    latticeones = latticeones.elem( find(lattice != EMPTY) );
    std::uniform_int_distribution<T> RandSample(0, static_cast<int>(latticeones.n_elem) - 1);
    std::exponential_distribution<double> ExponentialDistribution(beta);

    arma::uvec boundarydetect(walk_length);

    // Simulate a random walk on the lattice
    for (int i = 0; i < N_walks; i++) {
      bool ok_start = false;
      int pos;
      int count_loop = 0;
      int count_max = (N > 1E6) ? N : 1E6;
      do {
        pos = latticeones(RandSample(RNG));
        // Check start position has >= 1 occupied nearest neighbours
        arma::Col<T> neighbours = GetOccupiedNeighbours(pos);
        if(neighbours.n_elem > 0 || count_loop >= count_max) {
          ok_start = true;
        }
        else {
          count_loop++;
        }
      } while (!ok_start);

      // If stuck on a site with no nearest neighbours,
      // set the whole walk to that site
      if (count_loop == count_max) {
        walks = pos * arma::ones<arma::Col<T>>(walk_length);
        boundarydetect.zeros();
      }
      else {
        walks(0) = pos;
        boundarydetect(0) = 0;
        for (int j = 1; j < walk_length; j++) {
          arma::Col<T> neighbours = GetOccupiedNeighbours(pos);
          std::uniform_int_distribution<T> RandChoice(0, static_cast<int>(neighbours.n_elem) - 1);
          pos = neighbours(RandChoice(RNG));
          walks(j) = pos;

          // Check for walks that hit the top boundary
          if (arma::any(firstrow == walks(j - 1))
              && arma::any(lastrow == pos)) {
            boundarydetect(j) = 1;
          }
          // Check for walks that hit the bottom boundary
          else if (arma::any(lastrow == walks(j - 1))
                   && arma::any(firstrow == pos)) {
            boundarydetect(j) = 2;
          }
          // Check for walks that hit the RHS
          else if (walks(j - 1) > (N - L)
                   && pos < L) {
            boundarydetect(j) = 3;
          }
          // Check for walks that hit the LHS
          else if (walks(j - 1) < L
                   && pos > (N - L)) {
            boundarydetect(j) = 4;
          }
          // Else do nothing
          else {
            boundarydetect(j) = 0;
          }
        }
      }

      // Draw CTRW variates from exponential distribution
      ctrwTimes.set_size(walk_length);
      ctrwTimes.imbue( [&]() { return ExponentialDistribution(RNG); } );

      // Transform to Pareto distribution and accumulate
      ctrwTimes = arma::cumsum(arma::exp(ctrwTimes));

      // Only keep times within range [0, true_length]
      arma::uvec temp_time_boundary = arma::find(ctrwTimes >= true_length, 1, "first");
      int time_boundary = temp_time_boundary(0);
      ctrwTimes = ctrwTimes(arma::span(0,time_boundary));
      ctrwTimes(time_boundary) = true_length;

      // Subordinate fractal walk with CTRW
      int counter = 0;
      for (int j = 0; j < true_length; j++) {
        if (j > ctrwTimes(counter)) {
          counter++;
        }
        true_walks(j) = walks(counter);
      }

      // Finally convert the walk to the coordinate system
      int nx_cell = 0;
      int ny_cell = 0;
      for (int nstep = 0; nstep < true_length; nstep++) {
        switch (boundarydetect(nstep)) {
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
        walks_coordinates(0, nstep, i) = lattice_coordinates(0, true_walks(nstep))
                                          + nx_cell * unit_cell(0);
        walks_coordinates(1, nstep, i) = lattice_coordinates(1, true_walks(nstep))
                                          + ny_cell * unit_cell(1);
      }
    }
    return;
  }

  void BuildLattice() {
    // Populate the honeycomb lattice coordinates
    if (latticemode == 1) {
      double xx, yy;
      int count = 0;
      int cur_col = 0;
      for (int i = 0; i < 4*L; i++) {
        for (int j = L - 1; j >= 0; j--) {
          cur_col = i % 4;
          switch (cur_col) {
              case 0:
              default:
                xx = i / 4 * 3;
                yy = j * sqrt3 + sqrt3/2;
                break;
              case 1:
                xx = i / 4 * 3 + 1./2;
                yy = j * sqrt3;
                break;
              case 2:
                xx = i / 4 * 3 + 3./2;
                yy = j * sqrt3;
                break;
              case 3:
                xx = i / 4 * 3 + 2.;
                yy = j * sqrt3 + sqrt3/2;
                break;
          }
          lattice_coordinates(0, count) = xx;
          lattice_coordinates(1, count) = yy;
          lattice_coordinates(2, count) = (lattice(count) == EMPTY) ? 0 : 1;
          count++;
        }
      }
    }
    // Get unit cell size
    unit_cell = arma::max(lattice_coordinates, 1);
    unit_cell(0) += 3/2;
    unit_cell(1) += sqrt3/2;
    return;
  }

  // Check occupied neighbours of a point
  arma::Col<T> GetOccupiedNeighbours(int pos) {
    arma::Col<T> neighbours = nn.col(pos);
    arma::Col<T> neighbour_check(3);
    for (int k = 0; k < nearest; k++) {
      neighbour_check(k) = (lattice(neighbours(k)) == EMPTY) ? 0 : 1;
    }
    neighbours = neighbours.elem( find(neighbour_check == 1) );
    return neighbours;
  }

  // Randomise the order in which sites are occupied
  void Permutation() {
    T j;
    T temp;

    for (int i = 0; i < N; i++) {
      occupation(i) = i;
    }
    for (int i = 0; i < N; i++) {
      j = i + (N-i) * 2.3283064e-10 * UniformDistribution(RNG);
      temp = occupation(i);
      occupation(i) = occupation(j);
      occupation(j) = temp;
    }
    return;
  }

  // Find root of branch
  int FindRoot(int i) {
    if (lattice(i) < 0) {
       return i;
    }
    return lattice(i) = FindRoot(lattice(i));
  }

  // Percolation algorithm
  void Percolate() {
    int s1, s2;
    int r1, r2;
    T big = 0;

    for (int i = 0; i < N; i++) {
      lattice(i) = EMPTY;
    }
    for (int i = 0; i < (threshold * N) - 1; i++) {
      r1 = s1 = occupation[i];
      lattice(s1) = -1;
      for (int j = 0; j < nearest; j++) {
        s2 = nn(j, s1);
        if (lattice(s2) != EMPTY) {
          r2 = FindRoot(s2);
          if (r2 != r1) {
            if (lattice(r1) > lattice(r2)) {
              lattice(r2) += lattice(r1);
              lattice(r1) = r2;
              r1 = r2;
            } else {
              lattice(r1) += lattice(r2);
              lattice(r2) = r1;
            }
            if (-lattice(r1) > big) {
              big = -lattice(r1);
            }
          }
        }
      }
    }
    return;
  }

  // Nearest neighbours of a graphene lattice with
  // periodic boundary conditions
  void BoundariesHoneycomb() {
    int cur_col = 0;
    int count = 0;
    for (int i = 0; i < N; i++) {
      // First site
      if (i == 0) {
        nn(0, i) = i + L;
        nn(1, i) = i + 2*L - 1;
        nn(2, i) = i + N - L;
      }
      // Top right-hand corner
      else if (i == N - L) {
        nn(0, i) = i - 1;
        nn(1, i) = i - L;
        nn(2, i) = i - N + L;
      }
      // Bottom right-hand corner
      else if (i == N - L - 1) {
        nn(0, i) = i - L;
        nn(1, i) = i + L;
        nn(2, i) = i + 1;
      }
      // First column
      else if (i < L) {
        nn(0, i) = i + L - 1;
        nn(1, i) = i + L;
        nn(2, i) = i + N - L;
      }
      // Last column
      else if (i > (N - L)) {
        nn(0, i) = i - L - 1;
        nn(1, i) = i - L;
        nn(2, i) = i - N + L;
      }
      // Run through the rest of the tests
      else {
        switch (cur_col) {
          case 0:
            // First row
            if (arma::any(firstrow == i)) {
              nn(0, i) = i - L;
              nn(1, i) = i + L;
              nn(2, i) = i + 2*L - 1;
            }
            // Otherwise
            else {
              nn(0, i) = i - L;
              nn(1, i) = i + L - 1;
              nn(2, i) = i + L;
            }
            break;
          case 1:
            // Last row
            if (arma::any(lastrow == i)) {
              nn(0, i) = i - L;
              nn(1, i) = i + L;
              nn(2, i) = i - 2*L + 1;
            }
            // Otherwise
            else {
              nn(0, i) = i - L;
              nn(1, i) = i - L + 1;
              nn(2, i) = i + L;
            }
            break;
          case 2:
            // Last row
            if (arma::any(lastrow == i)) {
              nn(0, i) = i - L;
              nn(1, i) = i + L;
              nn(2, i) = i + 1;
            }
            // Otherwise
            else {
              nn(0, i) = i - L;
              nn(1, i) = i + L;
              nn(2, i) = i + L + 1;
            }
            break;
          case 3:
            // First row
            if (arma::any(firstrow == i)) {
              nn(0, i) = i - 1;
              nn(1, i) = i - L;
              nn(2, i) = i + L;
            }
            // Otherwise
            else {
              nn(0, i) = i - L - 1;
              nn(1, i) = i - L;
              nn(2, i) = i + L;
            }
            break;
        }
      }

      // Update current column
      if ((i + 1) % L == 0) {
        count++;
        cur_col = count % 4;
      }
    }
    return;
  }

  // Nearest neighbours of a square lattice
  // with periodic boundary conditions
  void BoundariesSquare() {
    for (int i = 0; i < N; i++) {
      nn(0, i) = (i + 1) % N;
      nn(1, i) = (i + N - 1) % N;
      nn(2, i) = (i + L) % N;
      nn(3, i) = (i + N - L) % N;
      if (i % L == 0) {
        nn(1, i) = i + L - 1;
      }
      if ((i + 1) % L == 0) {
        nn(0, i) = i - L + 1;
      }
    }
    return;
  }
};

#endif
