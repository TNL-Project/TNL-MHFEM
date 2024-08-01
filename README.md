# TNL-MHFEM

__TNL-MHFEM__ is an implementation of the __Mixed Hybrid Finite Element Method__ using the
[__Template Numerical Library__](https://gitlab.com/tnl-project/tnl).
This repository contains the complete general numerical scheme as described below and
a few simple examples that show how to adapt the code for a particular problem.

The numerical scheme is implemented for a PDE system written in the general coefficient form
$$
    \sum\limits_{j=1}^n N_{i,j} \frac{\partial Z_j}{\partial t}
    + \sum\limits_{j=1}^n \vec u_{i,j} \cdot \nabla Z_j
    + \nabla \cdot \left[ m_i \left(
        - \sum\limits_{j=1}^n \boldsymbol D_{i,j} \nabla Z_j
        + \vec w_i \right)
    + \sum\limits_{j=1}^n Z_j \vec a_{i,j} \right]
    + \sum\limits_{j=1}^n r_{i,j} Z_j = f_i
$$
for $i \in \{ 1, \ldots, n \}$, where $\vec Z = [Z_1, \ldots, Z_n]^T$ is the vector of
unknown functions depending on spatial coordinates $\vec x \in \Omega \subset \mathbb R^d$
and time $t \in [0, t_{\max}]$, where $d \in \{1, 2, 3\}$ denotes the spatial dimension,
$\Omega$ is a polyhedral domain, and $t_{\max}$ is the final simulation time.
The remaining symbols $`\boldsymbol N = [N_{i,j}]_{i,j=1}^n`$, $`\vec u = [\vec u_{i,j}]_{i,j=1}^n`$,
$`\vec m = [m_i]_{i=1}^n`$, $`\boldsymbol D = [\boldsymbol D_{i,j}]_{i,j=1}^n`$,
$`\vec w = [\vec w_i]_{i=1}^n`$, $`\vec a = [\vec a_{i,j}]_{i,j=1}^n`$,
$`\vec r = [r_{i,j}]_{i,j=1}^n`$, $`\vec f = [f_i]_{i=1}^n`$ are given problem-specific coefficients.

The scheme was originally developed for simulating multicomponent flow and
transport phenomena in porous media, but it can be used for any problem whose
governing equations can be written in a compatible form.
Details related to the numerical scheme can be found in the following paper:

- R. Fučík, J. Klinkovský, J. Solovský, T. Oberhuber, J. Mikyška,
  [Multidimensional mixed-hybrid finite element method for compositional two-phase
  flow in heterogeneous porous media and its parallel implementation on GPU](
  https://doi.org/10.1016/j.cpc.2018.12.004).
  Computer Physics Communications. 2019, 238 165-180.

Since the publication of the paper, several algorithmic as well as computational
improvements were incorporated into the solver.
Most notably, the solver supports MPI computations on distributed unstructured meshes.

## Getting started

1. Install [Git](https://git-scm.com/) and [Git LFS](https://git-lfs.com/).

2. Clone the repository:

       git clone https://gitlab.com/tnl-project/tnl-mhfem.git

3. Install the necessary tools and dependencies:

    - [CMake](https://cmake.org/) build system (version 3.24 or newer)
    - [CUDA](https://docs.nvidia.com/cuda/index.html) toolkit (version 11 or newer)
    - compatible host compiler (e.g. [GCC](https://gcc.gnu.org/) or
      [Clang](https://clang.llvm.org/))
    - [Python 3](https://www.python.org/) (including development header files)
    - [zlib](https://www.zlib.net/) (available in most Linux distributions)
    - [tinyxml2](https://github.com/leethomason/tinyxml2)
    - (optional) MPI library – for distributed computing
      (tested with [OpenMPI](https://www.open-mpi.org/))
    - (optional) [Hypre](https://github.com/hypre-space/hypre/) – library of
      high-performance solvers and preconditioners for sparse linear systems
    - (optional) [Ginkgo](https://github.com/ginkgo-project/ginkgo/) – library of
      high-performance solvers and preconditioners for sparse linear systems

4. Configure the build using `cmake` in the root path of the Git repository:

       cmake -B build -S . <additional_configure_options...>

   This will use `build` in the current path as the build directory.
   The path for the `-S` option corresponds to the root path of the project.

5. Build the targets using `cmake`:

       cmake --build build

6. Install the project using `cmake`:

       cmake --install build --prefix ~/.local

   Using `~/.local` as the installation prefix makes all Python modules automatically available to the interpreter.
   You may need to use a different prefix, for example when using a Python virtual environment.

7. Run the example solver:

       ./examples/HeatEquation/run.py --device cuda

   This will use the default configuration prepared in the [config.ini](examples/HeatEquation/config.ini)
   file. Use the `--help` option to see the options available in `run.py`.

## Getting involved

The TNL project welcomes and encourages participation by everyone. While most of the work for TNL
involves programming in principle, we value and encourage contributions even from people proficient
in other, non-technical areas.

This section provides several areas where both new and experienced TNL users can contribute to the
project. Note that this is not an exhaustive list.

- Join the __code development__. Our [GitLab issues tracker][GitLab issues] collects ideas for
  new features, or you may bring your own.
- Help with __testing and reporting problems__. Testing is an integral part of agile software
  development which refines the code development. Constructive critique is always welcome.
- Contact us and __provide feedback__ on [GitLab][GitLab issues]. We are interested to know how
  and where you use TNL and the TNL-MHFEM module.

[GitLab issues]: https://gitlab.com/tnl-project/tnl-mhfem/-/issues

## License

TNL-MHFEM is provided under the terms of the [MIT License](./LICENSE).
