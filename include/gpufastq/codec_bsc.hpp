#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gpufastq {

constexpr size_t BSC_QUALITY_CHUNK_SIZE = 8 * 1024 * 1024;

enum class QualityCodec : uint8_t {
  Bsc = 0,
  Zstd = 1,
};

enum class BscBackend {
  Default,
  Cpu,
  Cuda,
};

enum class BasecallPackOrder : uint8_t {
  Tgca = 0,
  Acgt = 1,
};

struct BscConfig {
  QualityCodec quality_codec = QualityCodec::Bsc;
  BscBackend backend = BscBackend::Default;
  size_t threads = 0;
  size_t gpu_jobs = 0;
  BasecallPackOrder basecall_pack_order = BasecallPackOrder::Tgca;
  bool zstd_transpose_quality = false;
  bool stat_mode = false;
  bool base_bsc = false; // New flag for compressing packed bases with libbsc
  std::string log_stat_path;
  size_t chunk_size_gb = 8;
};

struct ResolvedBscConfig {
  BscBackend backend = BscBackend::Cpu;
  size_t parallelism = 1;
};

struct BscChunkedBuffer {
  std::vector<uint8_t> data;
  std::vector<uint64_t> compressed_chunk_sizes;
  std::vector<uint64_t> uncompressed_chunk_sizes;
};

std::string_view bsc_backend_name(BscBackend backend);
std::string_view quality_codec_name(QualityCodec codec);

ResolvedBscConfig resolve_bsc_config(const BscConfig &config,
                                     size_t task_count);

void initialize_bsc_backend(BscBackend backend);

std::vector<uint8_t> bsc_compress_block(const uint8_t *input, size_t input_size,
                                        BscBackend backend);

BscChunkedBuffer
bsc_compress_chunked(const uint8_t *input, size_t input_size,
                     size_t chunk_size = BSC_QUALITY_CHUNK_SIZE,
                     const BscConfig &config = {});

std::vector<uint8_t>
bsc_decompress_chunked(const std::vector<uint8_t> &compressed,
                       const std::vector<uint64_t> &compressed_chunk_sizes,
                       const std::vector<uint64_t> &uncompressed_chunk_sizes,
                       uint64_t expected_size, const BscConfig &config = {});

} // namespace gpufastq
