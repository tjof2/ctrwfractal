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

#include <iostream>
#include <string>

#include "ctrw.hpp"
#include "optionparser/ezOptionParser.hpp"

int main(int argc, const char *argv[])
{
  std::cout << std::string(30, '-') << std::endl;
  std::cout << "CTRWfractal" << std::endl
            << std::endl;
  std::cout << "Author: Tom Furnival" << std::endl;
  std::cout << "Email:  tjof2@cam.ac.uk" << std::endl;
  std::cout << std::string(30, '-') << std::endl
            << std::endl;

  ez::ezOptionParser opt;
  opt.overview = "\nCTRWfractal";
  opt.syntax = "./fractalwalk";
  opt.example = "./fractalwalk\n\n";
  opt.add("",                            // Default.
          0,                             // Required?
          0,                             // Number of args expected.
          0,                             // Delimiter if expecting multiple args.
          "Display usage instructions.", // Help description.
          "-h",                          // Flag token.
          "-help",                       // Flag token.
          "--help",                      // Flag token.
          "--usage"                      // Flag token.
  );
  opt.add("honeycomb", 0, 1, 0, "Filename for output", "-o", "-output");
  opt.add("Honeycomb", 0, 1, 0, "Lattice type", "-l", "-lattice");
  opt.add("-1", 0, 1, 0, "Percolation fraction", "-f", "-fraction");
  opt.add("128", 0, 1, 0, "Dimensions of lattice", "-d", "-dim");
  opt.add("10", 1, 1, 0, "Simulate random walks on this lattice", "-w", "-walks");
  opt.add("1000", 1, 1, 0, "Length of random walks", "-n", "-nsteps");
  opt.add("1", 1, 1, 0, "Power-law beta", "-b", "-beta");
  opt.add("1", 0, 1, 0, "Power-law scale", "-tau");
  opt.add("0", 0, 1, 0, "Gaussian noise on walks", "-g", "-gaussian");
  opt.add("0", 0, 1, 0, "Random seed", "-s", "-seed");
  opt.add("0", 0, 1, 0, "Walk type", "-t", "-type");

  // Check for errors
  opt.parse(argc, argv);
  if (opt.isSet("-h"))
  {
    std::string usage;
    opt.getUsage(usage, 80, ez::ezOptionParser::INTERLEAVE);
    std::cout << usage;
    return 1;
  }
  std::vector<std::string> badOptions;
  if (!opt.gotRequired(badOptions))
  {
    for (int i = 0; i < (int)badOptions.size(); ++i)
    {
      std::cerr << "ERROR: Missing required option " << badOptions[i]
                << std::endl;
    }
    return 1;
  }
  if (!opt.gotExpected(badOptions))
  {
    for (int i = 0; i < (int)badOptions.size(); ++i)
    {
      std::cerr << "ERROR: Got unexpected number of arguments for option "
                << badOptions[i] << std::endl;
    }
    return 1;
  }

  std::string outfile, lattice;
  double fraction, beta, noise, tau0;
  int seed, size, walks, length, type;

  opt.get("-output")->getString(outfile);
  opt.get("-lattice")->getString(lattice);
  opt.get("-fraction")->getDouble(fraction);
  opt.get("-beta")->getDouble(beta);
  opt.get("-tau")->getDouble(tau0);
  opt.get("-gaussian")->getDouble(noise);
  opt.get("-dim")->getInt(size);
  opt.get("-seed")->getInt(seed);
  opt.get("-walks")->getInt(walks);
  opt.get("-nsteps")->getInt(length);
  opt.get("-type")->getInt(type);

  if (fraction == 0. || fraction > 1.) // Check for argument errors
  {
    std::cerr << "ERROR: fraction must be 0 < f <= 1" << std::endl;
    return 1;
  }
  else if (fraction < 0.)
  {
    // See http://dx.doi.org/10.1088/1751-8113/47/13/135001
    // for details on thresholds for percolation:
    //   - Square:     0.592746
    //   - Honeycomb:  0.697040230

    if (lattice.compare("Honeycomb") == 0)
    {
      fraction = 0.697040230;
    }
    else if (lattice.compare("Square") == 0)
    {
      fraction = 0.592746;
    }
  }

  // Generate the lattice and run the walks
  CTRWfractal<int32_t> *sim = new CTRWfractal<int32_t>();
  sim->Initialize(size, fraction, seed, lattice, walks, length, beta, tau0,
                  noise, type);
  sim->Run();
  sim->Save(outfile);

  std::cout << std::endl;
  return 0;
}
