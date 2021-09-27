# PACE 2018 Steiner Tree Algorithms

This repository contains Rust implementations for the Tracks 1 and 3 of the
[2018 PACE challenge](https://pacechallenge.org/2018/steiner-tree/).

I wrote this for an oral exam for an algorithm engineering class at university.
It is probably not suitable for real-world applications.

I implemented the following algorithms:

* [Dreyfus-Wagner Algorithm](https://doi.org/10.1002/net.3230010302)
* [Algorithm of Kou et al.](https://doi.org/10.1007/BF00288961)
* Algorithm of Takahashi and Matsuyama (H. Takahashi and A. Matsuyama. An approximate solution for the Steiner problem in graphs. 1980)


## Usage (Benchmarks)
The easiest way to run the benchmarks is by running the `benchmark.sh` script in this repository.
The usage is as follows:
``` sh
./benchmark.sh results.csv path/to/pace_instance_directory
```
The benchmark results are then stored in `results.csv`.

## Usage (Tests, Example)
For a "real" example program that runs all the algorithms and does some sanity-checking of
the results see `examples/run_all.rs`. You can e.g. execute it on
[Instance 001 of PACE 2018 Track 1](https://github.com/PACE-challenge/SteinerTree-PACE-2018-instances/blob/master/Track1/instance001.gr).
```sh
cargo run --release --example=run_all path/to/instance001.gr
```
