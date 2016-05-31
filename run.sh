 #!/bin/bash          

dim=100
seed=120516
./percolation -o hex-0.1.cluster -l Hexagonal -f 0.1 -s $seed -d $dim
./percolation -o hex-0.2.cluster -l Hexagonal -f 0.2 -s $seed -d $dim
./percolation -o hex-0.3.cluster -l Hexagonal -f 0.3 -s $seed -d $dim
./percolation -o hex-0.4.cluster -l Hexagonal -f 0.4 -s $seed -d $dim
./percolation -o hex-0.5.cluster -l Hexagonal -f 0.5 -s $seed -d $dim
./percolation -o hex-0.6.cluster -l Hexagonal -f 0.6 -s $seed -d $dim
./percolation -o hex-0.7.cluster -l Hexagonal -f 0.7 -s $seed -d $dim
./percolation -o hex-0.8.cluster -l Hexagonal -f 0.8 -s $seed -d $dim
./percolation -o hex-0.9.cluster -l Hexagonal -f 0.9 -s $seed -d $dim

