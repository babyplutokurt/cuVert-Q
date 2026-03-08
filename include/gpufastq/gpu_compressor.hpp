#pragma once

#include "codec_bsc.hpp"
#include "fastq_record.hpp"
#include <cstdint>
#include <vector>

namespace gpufastq {

/// Default chunk size for nvcomp zstd (64KB recommended for best performance)
constexpr size_t DEFAULT_CHUNK_SIZE = 1 << 16; // 65536

/// Compress a single buffer using GPU-accelerated zstd (nvcomp)
std::vector<uint8_t> gpu_compress(const std::vector<uint8_t> &input,
                                  size_t chunk_size = DEFAULT_CHUNK_SIZE);

/// Decompress a single buffer using GPU-accelerated zstd (nvcomp)
std::vector<uint8_t> gpu_decompress(const std::vector<uint8_t> &compressed);

/// Extract identifiers/basecalls/quality plus delta-coded line lengths and compress them.
CompressedFastqData compress_fastq(const FastqData &data,
                                   size_t chunk_size = DEFAULT_CHUNK_SIZE,
                                   const BscConfig &bsc_config = {});

/// Decompress field streams and rebuild the raw FASTQ byte buffer.
FastqData decompress_fastq(const CompressedFastqData &compressed,
                           const BscConfig &bsc_config = {});

} // namespace gpufastq
