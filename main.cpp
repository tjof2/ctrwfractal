/***************************************************************************

    Copyright (C) 2016 Tom Furnival
    Email: tjof2@cam.ac.uk

    This file is part of CTRWfractal

***************************************************************************/

// C++ headers
#include <iostream>
#include <string>

// Include header
#include "ctrw.hpp"

// Option parser
#include "optionparser/ezOptionParser.hpp"

int main(int argc, const char * argv[]) {
  std::cout<<std::string(30,'-')<<std::endl;
  std::cout<<"CTRWfractal"<<std::endl<<std::endl;
  std::cout<<"Author: Tom Furnival"<<std::endl;
  std::cout<<"Email:  tjof2@cam.ac.uk"<<std::endl;
  std::cout<<std::string(30,'-')<<std::endl<<std::endl;

  ez::ezOptionParser opt;
	opt.overview = "\nCTRWfractal";
	opt.syntax = "./fractalwalk";
	opt.example = "./fractalwalk\n\n";
	opt.add(
		"", 							               // Default.
		0, 								               // Required?
		0, 								               // Number of args expected.
		0, 								               // Delimiter if expecting multiple args.
		"Display usage instructions.", 	 // Help description.
		"-h",     					             // Flag token.
		"-help",  						           // Flag token.
		"--help", 						           // Flag token.
		"--usage" 					             // Flag token.
	);
	opt.add("honeycomb", 0, 1, 0, "Filename for output", "-o", "-output");
  opt.add("Honeycomb", 0, 1, 0, "Lattice type", "-l", "-lattice");
  opt.add("0.5", 0,	1, 0, "Percolation fraction", "-f", "-fraction");
	opt.add("128", 0, 1, 0, "Dimensions of lattice", "-d", "-dim");
  opt.add("10", 1, 1, 0, "Simulate random walks on this lattice", "-w", "-walks");
  opt.add("1000", 1, 1, 0, "Length of random walks", "-n", "-nsteps");
  opt.add("1", 1, 1, 0, "Power-law beta", "-b", "-beta");
	opt.add("0", 0,	1, 0, "Random seed", "-s", "-seed");

	// Check for errors
	opt.parse(argc, argv);
	if (opt.isSet("-h")) {
		std::string usage;
		opt.getUsage(usage,80,ez::ezOptionParser::INTERLEAVE);
		std::cout<<usage;
		return 1;
	}
	std::vector<std::string> badOptions;
	if (!opt.gotRequired(badOptions)) {
		for(int i=0; i < (int)badOptions.size(); ++i) {
			std::cerr << "ERROR: Missing required option "
                << badOptions[i] << std::endl;
		}
		return 1;
	}
	if (!opt.gotExpected(badOptions)) {
		for(int i=0; i < (int)badOptions.size(); ++i) {
			std::cerr << "ERROR: Got unexpected number of arguments for option "
                << badOptions[i] << std::endl;
		}
		return 1;
	}

  std::string outfile, lattice;
  double fraction, beta;
  int seed, size, walks, length;

  opt.get("-output")->getString(outfile);
  opt.get("-lattice")->getString(lattice);
  opt.get("-fraction")->getDouble(fraction);
  opt.get("-beta")->getDouble(beta);
  opt.get("-dim")->getInt(size);
  opt.get("-seed")->getInt(seed);
  opt.get("-walks")->getInt(walks);
  opt.get("-nsteps")->getInt(length);

  // Check for argument errors
  if (fraction == 0. || fraction > 1.) {
    std::cerr << "ERROR: fraction must be 0 < f <= 1" << std::endl;
    return 1;
  }

  // Generate the lattice and run the walks
  CTRWfractal<int32_t> *sim = new CTRWfractal<int32_t>();
  sim->Initialize(size, fraction, seed, lattice, walks, length, beta);
  sim->Run();
  sim->Save(outfile);

  std::cout<<std::endl;
  return 0;
}
