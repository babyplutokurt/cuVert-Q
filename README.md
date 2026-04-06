# CuVert-Q

CuVert-Q is a GPU-accelerated FASTQ compressor built around field-aware encoding and CUDA-based codecs. It splits FASTQ records into identifiers, basecalls, quality scores, and line-length metadata, then compresses those streams with a mix of GPU `nvcomp` Zstd and `libbsc`.

The repository builds three executables:

- `cuvertq`: main compressor/decompressor CLI
- `gpufastq_ablation`: ablation driver for `libbsc`-based comparisons
- `gpufastq_ablation_zstd`: ablation driver for `nvcomp` Zstd comparisons

Compressed output uses the `.cuvf` container format.

## Features

- GPU-accelerated FASTQ compression and decompression
- Chunked processing for large inputs
- Separate handling of identifiers, basecalls, quality scores, and line metadata
- Selectable quality-score codec: `bsc` or `zstd`
- Optional CUDA-backed `libbsc` path
- Round-trip verification mode
- Built-in timing and throughput reporting

## Requirements

- CMake 3.24+
- C++17 and CUDA 17-capable toolchain
- NVIDIA CUDA Toolkit
- `nvcomp`
- OpenMP

Notes:

- The build expects `nvcomp` headers and `libnvcomp.so.5`.
- If `nvcomp` is not passed explicitly, CMake tries to discover it from a Python installation such as `nvidia-libnvcomp-cu12`.
- `CMAKE_CUDA_ARCHITECTURES` is currently set to `120` in [CMakeLists.txt](/home/kurty/Project/cuVert-Q/CMakeLists.txt).

## Repository Layout

- `src/`: CLI, compression workflows, CUDA codecs, and ablation binaries
- `include/gpufastq/`: public headers
- `deps/libbsc/`: vendored `libbsc`
- `build/`: local build output

## Build

Configure with an explicit CUDA toolkit path when needed:

```bash
cmake -S . -B build -DCUDA_TOOLKIT_PATH=/usr/local/cuda-12.8
```

If `nvcomp` is not auto-detected, add `NVCOMP_ROOT`:

```bash
cmake -S . -B build \
  -DCUDA_TOOLKIT_PATH=/usr/local/cuda-12.8 \
  -DNVCOMP_ROOT=/path/to/nvidia/libnvcomp
```

Build all targets:

```bash
cmake --build build -j
```

Primary outputs:

- `build/cuvertq`
- `build/gpufastq_ablation`
- `build/gpufastq_ablation_zstd`

## Main CLI

Usage:

```bash
./build/cuvertq compress   [options] <input.fastq> <output.cuvf>
./build/cuvertq decompress [options] <input.cuvf> <output.fastq>
./build/cuvertq roundtrip  [options] <input.fastq>
```

### Common examples

Compress a FASTQ file:

```bash
./build/cuvertq compress reads.fastq reads.cuvf
```

Decompress a `.cuvf` file:

```bash
./build/cuvertq decompress reads.cuvf reads.fastq
```

Run round-trip verification:

```bash
./build/cuvertq roundtrip reads.fastq
```

Use Zstd for quality scores:

```bash
./build/cuvertq compress --quality-codec zstd reads.fastq reads.cuvf
```

Use CUDA-backed `libbsc` for quality-score compression:

```bash
./build/cuvertq compress --bsc-backend cuda reads.fastq reads.cuvf
```

For the CUDA `libbsc` path, increase `--bsc-gpu-jobs` to raise chunk-level parallelism:

```bash
./build/cuvertq compress --bsc-backend cuda --bsc-gpu-jobs 64 reads.fastq reads.cuvf
```

Enable transposed quality-score layout before Zstd compression:

```bash
./build/cuvertq compress --quality-codec zstd --transpose reads.fastq reads.cuvf
```

Compress packed basecalls with `libbsc` instead of `nvcomp`:

```bash
./build/cuvertq compress --base-bsc reads.fastq reads.cuvf
```

### CLI options

- `--quality-codec <bsc|zstd>`: quality-score codec, default `bsc`
- `--bsc-backend <cpu|cuda>`: backend for `libbsc`, default comes from environment or falls back to `cpu`
- `--bsc-threads <N>`: CPU worker count for CPU `libbsc`
- `--bsc-gpu-jobs <N>`: concurrent chunk jobs for CUDA `libbsc`; increase this to raise GPU-side parallelism
- `--base-bsc`: use `libbsc` for packed basecall compression
- `--base-pack-order <tgca|acgt>`: base packing order, default `tgca`
- `--transpose`: transpose fixed-length quality scores before Zstd compression
- `--stat`: print detailed stage timings
- `--log-stat <FILE>`: write timing and throughput statistics to a log file
- `--chunk-size <N>`: processing chunk size in GB, default `8`

### Environment variables

- `GPUFASTQ_BSC_BACKEND`: default `libbsc` backend
- `GPUFASTQ_BSC_THREADS`: default CPU worker count
- `GPUFASTQ_BSC_GPU_JOBS`: default CUDA chunk job count for the CUDA `libbsc` path

## Compression Model

At a high level, CuVert-Q parses FASTQ input into chunks, separates record fields, compresses each stream with the selected codec path, and writes the result into a `.cuvf` container with a versioned header.

Current file-format constants are defined in [serializer.hpp](/home/kurty/Project/cuVert-Q/include/gpufastq/serializer.hpp):

- Magic: `GFQZ`
- Extension: `.cuvf`
- Format version: `18`

## Validation

There is no standalone test suite in the repository yet. The main correctness check is round-trip reconstruction:

```bash
./build/cuvertq roundtrip sample.fastq
```

For functional changes, also exercise the direct compress/decompress path:

```bash
./build/cuvertq compress sample.fastq sample.cuvf
./build/cuvertq decompress sample.cuvf sample.roundtrip.fastq
cmp sample.fastq sample.roundtrip.fastq
```

## Ablation Tools

`gpufastq_ablation` compares:

- raw FASTQ compressed with `libbsc`
- split FASTQ fields compressed with `libbsc`
- full CuVert-Q compression

Example:

```bash
./build/gpufastq_ablation reads.fastq --bsc-backend cuda --base-bsc
```

`gpufastq_ablation_zstd` compares:

- raw FASTQ compressed with `nvcomp` Zstd
- split FASTQ fields compressed with `nvcomp` Zstd
- full CuVert-Q with Zstd quality compression
- full CuVert-Q with Zstd plus quality transposition

Example:

```bash
./build/gpufastq_ablation_zstd reads.fastq
```

These tools write intermediate output files next to the input and report final size ratios and timings on stderr.

## Troubleshooting

- `nvcomp not found`: configure with `-DNVCOMP_ROOT=/path/to/nvidia/libnvcomp` or install `nvidia-libnvcomp-cu12`
- `nvcomp headers not found`: verify `${NVCOMP_ROOT}/include/nvcomp.h`
- CUDA compiler not found: pass `-DCUDA_TOOLKIT_PATH=/usr/local/cuda-12.8` or your local CUDA install path
- Runtime linker issues: confirm the built binaries can see CUDA, `nvcomp`, and OpenMP runtime libraries

## Development Notes

- The codebase uses C++17/CUDA with 2-space indentation and `snake_case` naming for functions and variables.
- Headers in `include/gpufastq/` generally map directly to implementation files in `src/`.
- `build/` is generated output and should not be treated as source.
