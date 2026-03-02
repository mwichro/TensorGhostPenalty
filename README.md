# TensorGhost: Matrix-Free Ghost Penalty Evaluation via Tensor Product Factorization

## Overview

**TensorGhost** is a high-performance implementation of ghost penalty stabilization for Cut Finite Element Methods (CutFEM), integrated within the [deal.II](https://www.dealii.org) finite element library. 

This repository provides the source code and examples associated with the paper:
> *Matrix-Free Ghost Penalty Evaluation via Tensor Product Factorization* by Michał Wichrowski.

The is matrix-free approach that exploits the tensor-product structure of high-order basis functions on Cartesian meshes. By factoring the ghost penalty operator into one-dimensional components, we significantly reduce computational complexity and memory overhead, enabling efficient high-order CutFEM simulations in 2D and 3D.

## Key Features

- **Matrix-Free CutFEM**: Efficient evaluation of operators without explicit assembly of global matrices.
- **Tensor Product Factorization**: Implementation of ghost penalty stabilization using precomputed 1D mass and penalty matrices.
- **High-Order Support**: Optimized for high-order Lagrange finite elements (up to degree 8 and beyond).
- **Scalable Design**: Support for parallel distributed computations using MPI and vectorization.
- **Integration with deal.II**: Leverages the state-of-the-art `MatrixFree` framework.

## Project Structure

- `include/`: Header files containing the core operators and generators.
  - `ghost_penalty_operator.h`: Linear operator for tensor-product ghost penalty.
  - `matrix_free_operator.h`: General matrix-free operator structure for CutFEM.
  - `cut_cell_generator.h`: Utilities for handling cut cell info.
- `execs/`: Primary executables and benchmarks.
  - `matrix_free_global.cc`: A global Laplace solver demonstrating the full CutFEM pipeline.
  - `cell_comparison.cc`: Benchmark comparing various evaluation strategies (FEEvaluation vs. GhostPenalty).
- `source/`: Library source files.
- `doc/`: Documentation and related manuscript files.

## Requirements

- **deal.II**: Version 9.7 or later (the methods presented here are partially integrated into the 9.7 release).
- **CMake**: Version 3.10 or later.
- **MPI**: For parallel execution.
- **C++17** compatible compiler.

## Installation and Build

To configure and compile the project, point CMake to your deal.II installation:

```bash
mkdir build
cd build
cmake -Ddeal.II_DIR=/path/to/your/dealii/build/ ..
make -j4
```

## Running Examples

After building, you can run the benchmarking and solver examples:

```bash
# Run the cell-wise benchmarking (2D/3D)
./cell_comparison_2d
./cell_comparison_3d

# Run the global matrix-free solver (2D/3D)
./matrix_free_global_2d
./matrix_free_global_3d
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wichrowski2025tensorghost,
  title={Matrix-Free Ghost Penalty Evaluation via Tensor Product Factorization},
  author={Wichrowski, Michał},
  year={2025},
  journal={TBD}
}
```

## License

This project is licensed under the same terms as deal.II (LGPL v2.1 or later).
