#pragma once

#include "fastq_record.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpufastq {

constexpr size_t NVCOMP_ZSTD_CHUNK_SIZE_DEFAULT = 1 << 20;

ZstdCompressedBlock
nvcomp_zstd_compress(const std::vector<uint8_t> &input,
                     size_t chunk_size = NVCOMP_ZSTD_CHUNK_SIZE_DEFAULT,
                     cudaStream_t stream = 0);

ZstdCompressedBlock
nvcomp_zstd_compress_device(const uint8_t *d_input, size_t input_size,
                            size_t chunk_size = NVCOMP_ZSTD_CHUNK_SIZE_DEFAULT,
                            cudaStream_t stream = 0);

std::vector<uint8_t>
nvcomp_zstd_decompress(const ZstdCompressedBlock &compressed,
                       size_t chunk_size = NVCOMP_ZSTD_CHUNK_SIZE_DEFAULT,
                       cudaStream_t stream = 0);

void nvcomp_zstd_decompress_to_device(
    const ZstdCompressedBlock &compressed, uint8_t **d_output,
    size_t *output_size, size_t chunk_size = NVCOMP_ZSTD_CHUNK_SIZE_DEFAULT,
    cudaStream_t stream = 0);

} // namespace gpufastq
