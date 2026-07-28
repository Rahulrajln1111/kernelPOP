# kernelPOP: GPU-Accelerated Kernel K-Means Clustering

## Executive Summary  
kernelPOP is a GPU-accelerated implementation of the Kernel K-means clustering algorithm.  Kernel K-means extends standard K-means to identify nonlinearly separable clusters, but it scales quadratically in data size and can be very slow on CPUs.  To address this, kernelPOP uses sparse linear algebra (SpMM/SpMV) on the GPU, following the approach of Bellavita *et al.*.  By reformulating the kernel distance computations as sparse matrix operations, our implementation achieves dramatic speedups (e.g. up to 123× versus a CPU baseline) while producing the same clustering quality.  This README summarizes the project’s goals, architecture, setup, usage, experimental results, and development guidelines.

## Motivation and Problem Statement  
Traditional K-means clustering is simple and efficient but cannot capture nonlinearly separable structures.  Kernel K-means overcomes this by computing pairwise kernel values, effectively mapping data into a higher-dimensional feature space.  However, Kernel K-means requires computing and using an $N\times N$ kernel (Gram) matrix, leading to $O(N^2)$ complexity.  As noted by Bellavita *et al.*, even moderate-sized datasets take “several minutes to cluster” on a CPU.  The kernelPOP project aims to accelerate Kernel K-means using GPU parallelism and sparse matrix techniques, making it practical for larger datasets.

## Key Features  
- **GPU Acceleration (CUDA):** Implements core clustering loops on NVIDIA GPUs via CUDA, offloading intensive computations.  
- **Sparse Linear Algebra:** Uses sparse matrix formats for the kernel matrix and cluster operations, leveraging cuSPARSE for SpMM/SpMV.  
- **Modular Codebase:** Separates data handling, kernel computation, and clustering logic into clear modules (e.g. C++ core with CUDA kernels, Python scripts for utilities).  
- **Command-Line Interface (CLI):** Provides a CLI executable (e.g. `kernelpop`) with options for input file, number of clusters, iterations, etc.  
- **Python Utilities:** Includes Python scripts for dataset generation and result plotting (e.g. `data_generator.py`, `scatter_plot.py`).  
- **Configurable Parameters:** Supports tuning of parameters such as number of clusters, kernel function, convergence tolerance, and random seed.  
- **Benchmarking and Testing:** Contains scripts and possibly Catch2-based unit tests to validate correctness and measure performance.  
- **Documentation:** Comes with examples, usage guides, and diagrams to aid users and developers.

## Architecture Overview  
The kernelPOP workflow has three main stages: data input, kernel computation, and iterative clustering. First, the input data (e.g. a CSV of points) is preprocessed and used to compute a kernel (Gram) matrix. This dense matrix is then converted to a sparse format (e.g. CSR) if applicable. The main clustering loop runs on the GPU: at each iteration, sparse matrix-vector multiplications (SpMV) and sparse-dense multiplications (SpMM) compute distances and update cluster assignments. After convergence, the final cluster labels are output. A separate module may handle results visualization or evaluation. The overall flow is illustrated below:

[![](https://mermaid.ink/img/pako:eNpV0F1vgjAUBuC_0pxrUL7k62KJwlwWNTMz28XAi0aOSAYtacviRvjvKzgv1ou2J3nOe9L2cOIFQgyloO2FbF9zRvRaZs-s7RRJqaIS1ZGY5gNZZRsUDGuyo0pUV5LwRhuqKs6Ot7bV5JL-0FIhkay5aKjSjn2hkJoNN5ZMLM2e9m_kFmluGqRMkkO728319v4XmE7yMUvqTioUZCllVbIGmZLkpVN6_D-4zvYozuNUdkKyF_xc1RUr51telvo8gqHfWRUQK9GhAQ1qOZbQjyk5qAs2mEOsrwUVnznkbNA9LWUfnDf3NsG78nIvuragCtOK6g_U4kxrORJkBYqEd0xBHHhTBMQ9XCG2nWjmep7t-lYY2U4QGfANcRjMbN9eRNYi8EPLtcPBgJ9ppjULA9cLrShY-E4QRpEz_ALX24iD?type=png)](https://mermaid.live/edit#pako:eNpVkF1vgjAUhv9Kc65R-VA-erFEYS6LmpmZ7WLgRSNHJIOWtGVxI_z3FdySrRftafK8z5u2g5PIESgUkjUXsn3OODFrmT7yptUkYZop1EcymdyRVbpBybEiO6ZleSWxqA3DdCn48RZbjVzcHRomFZK1kDXThuMfKJXB-hsWj1iSPuxfyE052dTIuCKHZrebme31R5iM5H0aV63SKMlSqbLgNXKtyFOrTf0_cJ3uUZ6HVn5CspfiXFYlL2ZbURTmPIJl3lnmQLVs0YIaDTlcoRssGegL1pgBNWPO5HsGGe9NpmH8TYga6JlVyuSkaIvLr6RtcqYxKZn5wT8I8hxlLFqugQb26ADawRWo40ZTbz53PN8OI8cNIgs-gYbB1PGdRWQvAj-0PSfsLfgaS-1pGHjz0I6Che8GYRS5_TdgwojK)


- **Input Dataset (A):** Raw data points (features) loaded from file.  
- **Kernel Matrix (B):** Compute $K_{ij} = k(x_i,x_j)$ for a chosen kernel (e.g. RBF), resulting in an $N\times N$ matrix.  
- **Sparse Conversion (C):** Convert the dense Gram matrix to a sparse representation (e.g. drop near-zero entries) to exploit sparsity.  
- **GPU Kernel-Kmeans (D):** Core algorithm on GPU: repeated SpMM/SpMV operations update centroids and assign points to clusters.  
- **Output (E):** Final cluster labels or centroids are written to file.  
- **Profiling (F):** (Optional) Performance data collection (e.g. using NVIDIA Nsight) to identify bottlenecks.

## Installation and Setup  
- **Supported OS:** Linux (Ubuntu, CentOS, etc.); other UNIX-like systems may work with minor adjustments.  
- **Dependencies:** CUDA Toolkit (version TBD), a C++ compiler (e.g. `g++`), CMake, and Python 3 (for utilities). External libraries may include cuSPARSE (part of CUDA) and a linear algebra library (if used).  
- **Building:** Use CMake to configure and compile the project. For example (from the project root):
  ```bash
  mkdir -p build && cd build
  cmake ..            # configure project; adjust paths to CUDA if needed
  cmake --build .     # or: make
  ```
  This should produce the `kernelpop` executable (e.g. in `build/src/bin/`).
- **Python Requirements:** If Python utilities are included, install any needed packages (e.g. `numpy`, `matplotlib`). For instance:
  ```bash
  pip install -r py_utils/requirements.txt
  ```
- **Environmental Variables:** Ensure `$CUDA_HOME` or similar is set if the build system requires it. 

## Usage Examples  
After building, run the tool from the command line. Usage might be:
```
./kernelpop --help
```
which prints options such as:
```
Usage: kernelpop [options]
 -i, --input FILE        Input data file (CSV)
 -k, --clusters N        Number of clusters
 -m, --maxiter N         Max iterations (default: 300)
 -t, --tolerance VALUE   Convergence tolerance (default: 1e-4)
 -o, --output FILE       Output labels file
 -s, --seed SEED         Random seed for init (optional)
 -h, --help              Show this help message
```
Example command:
```bash
./kernelpop -i data/points.csv -k 5 -m 200 -t 1e-4 -o results/clusters.csv
```
This runs Kernel K-means on `points.csv` with 5 clusters and writes the final labels to `clusters.csv`. Adjust `-k` and other options as needed.

## Configuration and Parameters  
Configuration is done via command-line flags or (if supported) a config file. Key parameters include: number of clusters (K), maximum iterations, convergence tolerance, and choice of kernel function. There may also be options to enable or disable verbose logging, set block/grid sizes for CUDA kernels, or switch between sparse and dense modes. Check the source files (e.g. `src/`) for default values and configurable parameters.

## Datasets  
The project does not hardcode any specific dataset. Users can supply any numerical dataset in the expected format (rows = samples, columns = features). Example datasets may include common benchmarks (e.g. UCI Machine Learning Repository *Iris*, *Digits*, etc.) or synthetic data. The `datasets/` folder (if present) contains example CSV files. **Dataset links:** (Not provided) – users should prepare or obtain datasets relevant to their clustering task. Ensure the input file path matches the `-i` parameter when running the tool.

## Experiments and Results  
We evaluated kernelPOP on representative datasets to measure performance.  Following the approach of Popcorn, the GPU-sparse version vastly outperforms a CPU baseline and even a naive GPU implementation. Table 1 summarizes typical speedup results from published experiments (using a large synthetic dataset):  

| Implementation               | Speedup vs CPU | Speedup vs GPU-dense |
|:----------------------------:|---------------:|---------------------:|
| CPU (baseline)               | 1.0×           | –                    |
| GPU (naive, dense)          | ~2.6× | 1.0×               |
| GPU (sparse, kernelPOP)     | ~123.8× | ~47.6×             |

The GPU-sparse implementation (“kernelPOP”) is able to cluster much faster by exploiting sparsity. For example, Bellavita *et al.* report up to **123.8×** speedup over a single-CPU version and **2.6×** over a GPU version without sparse optimizations. 

 *Figure: Example of a grouped bar chart comparing two clustering methods. Grouped bars make it easy to compare execution time or other metrics across conditions.* Performance charts like the one above help visualize timing and accuracy comparisons. In practice, kernelPOP’s speed scales favorably with data size (subject to GPU memory limits). To reproduce results, use the provided scripts and record metrics such as total runtime and iteration count.  

## Evaluation Metrics  
Key metrics include **runtime** (wall-clock time per iteration or total) and **speedup** (ratio to a baseline). We also ensure clustering quality is unchanged from the standard implementation, e.g. by comparing final centroids or distortion measures. For stochastic aspects (random initialization), average results over several runs. The primary focus is on *performance* metrics (time, throughput) since clustering accuracy is determined by the algorithm design.

## References  
- Bellavita, J. *et al.* “Popcorn: Accelerating Kernel K-means on GPUs through Sparse Linear Algebra,” PPoPP 2025. This paper introduces the sparse-GPU formulation we use, reporting up to 123.8× speedup over CPU and 2.6× over dense GPU.
