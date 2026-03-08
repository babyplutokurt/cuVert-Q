#include "gpufastq/codec_bsc.hpp"

#include <libbsc/libbsc.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <thread>
#include <utility>

namespace gpufastq {

namespace {

BscBackend parse_backend_name(const std::string &value) {
  if (value == "cpu") {
    return BscBackend::Cpu;
  }
  if (value == "cuda") {
    return BscBackend::Cuda;
  }
  throw std::runtime_error("BSC backend must be 'cpu' or 'cuda'");
}

int bsc_features_for_backend(BscBackend backend) {
  int features = LIBBSC_FEATURE_FASTMODE;
  if (backend == BscBackend::Cuda) {
#ifdef LIBBSC_CUDA_SUPPORT
    features |= LIBBSC_FEATURE_CUDA;
#else
    throw std::runtime_error(
        "BSC CUDA backend requested, but libbsc was built without CUDA support");
#endif
  }
  return features;
}

void bsc_check_init(BscBackend backend) {
  const int rc = bsc_init(bsc_features_for_backend(backend));
  if (rc != LIBBSC_NO_ERROR) {
    throw std::runtime_error("BSC initialization failed with code: " +
                             std::to_string(rc));
  }
}

std::optional<size_t> parse_positive_size(const char *value) {
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }

  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
    throw std::runtime_error(
        "GPUFASTQ_BSC_THREADS must be a positive integer");
  }
  if (parsed > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("GPUFASTQ_BSC_THREADS exceeds supported range");
  }
  return static_cast<size_t>(parsed);
}

std::optional<BscBackend> parse_backend_env(const char *value) {
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }
  return parse_backend_name(value);
}

size_t resolve_worker_count(size_t requested_workers, const char *env_name,
                           size_t task_count, size_t default_workers) {
  if (task_count == 0) {
    return 1;
  }

  size_t configured_workers = requested_workers;
  if (configured_workers == 0) {
    configured_workers = parse_positive_size(std::getenv(env_name)).value_or(0);
  }

  if (configured_workers == 0) {
    return std::max<size_t>(1, std::min(task_count, default_workers));
  }
  return std::max<size_t>(1, std::min(task_count, configured_workers));
}

BscBackend resolve_backend(BscBackend requested_backend) {
  if (requested_backend != BscBackend::Default) {
    return requested_backend;
  }
  return parse_backend_env(std::getenv("GPUFASTQ_BSC_BACKEND"))
      .value_or(BscBackend::Cpu);
}

size_t resolve_cpu_worker_count(size_t requested_workers, size_t task_count) {
  const unsigned int detected = std::thread::hardware_concurrency();
  const size_t max_workers = detected == 0 ? 1 : static_cast<size_t>(detected);
  return resolve_worker_count(requested_workers, "GPUFASTQ_BSC_THREADS",
                              task_count, max_workers);
}

size_t resolve_gpu_job_count(size_t requested_jobs, size_t task_count) {
  return resolve_worker_count(requested_jobs, "GPUFASTQ_BSC_GPU_JOBS",
                              task_count, 1);
}

size_t resolve_parallelism(const BscConfig &config, size_t task_count,
                           BscBackend backend) {
  if (backend == BscBackend::Cuda) {
    return resolve_gpu_job_count(config.gpu_jobs, task_count);
  }
  return resolve_cpu_worker_count(config.threads, task_count);
}

} // namespace

std::string_view bsc_backend_name(BscBackend backend) {
  switch (backend) {
  case BscBackend::Cpu:
    return "cpu";
  case BscBackend::Cuda:
    return "cuda";
  case BscBackend::Default:
    return "default";
  }
  return "unknown";
}

std::string_view quality_codec_name(QualityCodec codec) {
  switch (codec) {
  case QualityCodec::Bsc:
    return "bsc";
  case QualityCodec::Zstd:
    return "zstd";
  }
  return "unknown";
}

ResolvedBscConfig resolve_bsc_config(const BscConfig &config, size_t task_count) {
  const BscBackend backend = resolve_backend(config.backend);
  return ResolvedBscConfig{
      backend,
      resolve_parallelism(config, task_count, backend),
  };
}

void initialize_bsc_backend(BscBackend backend) { bsc_check_init(backend); }

std::vector<uint8_t> bsc_compress_block(const uint8_t *input, size_t input_size,
                                        BscBackend backend) {
  if (input_size == 0) {
    return {};
  }
  if (input == nullptr) {
    throw std::runtime_error("BSC block compression input pointer is null");
  }
  if (input_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("BSC block exceeds supported size");
  }

  const int features = bsc_features_for_backend(backend);
  std::vector<uint8_t> compressed(input_size + LIBBSC_HEADER_SIZE);
  const int compressed_size = bsc_compress(
      input, compressed.data(), static_cast<int>(input_size), 16, 128,
      LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_ADAPTIVE, features);
  if (compressed_size < LIBBSC_NO_ERROR) {
    throw std::runtime_error("BSC compression failed with code: " +
                             std::to_string(compressed_size));
  }
  compressed.resize(static_cast<size_t>(compressed_size));
  return compressed;
}

BscChunkedBuffer bsc_compress_chunked(const uint8_t *input, size_t input_size,
                                      size_t chunk_size,
                                      const BscConfig &config) {
  BscChunkedBuffer result;
  if (input_size == 0) {
    return result;
  }
  if (input == nullptr) {
    throw std::runtime_error("BSC compression input pointer is null");
  }
  if (chunk_size == 0) {
    throw std::runtime_error("BSC compression chunk size must be non-zero");
  }

  const size_t chunk_count = (input_size + chunk_size - 1) / chunk_size;
  const auto resolved = resolve_bsc_config(config, chunk_count);
  const BscBackend backend = resolved.backend;
  initialize_bsc_backend(backend);

  std::vector<std::vector<uint8_t>> compressed_chunks(chunk_count);
  result.compressed_chunk_sizes.resize(chunk_count);
  result.uncompressed_chunk_sizes.resize(chunk_count);

  std::atomic<size_t> next_chunk{0};
  std::exception_ptr worker_error;
  std::atomic<bool> failed{false};
  std::mutex error_mutex;
  const size_t worker_count = resolved.parallelism;

  const auto worker = [&]() {
    try {
      while (!failed.load(std::memory_order_relaxed)) {
        const size_t chunk_index = next_chunk.fetch_add(1, std::memory_order_relaxed);
        if (chunk_index >= chunk_count) {
          return;
        }

        const size_t offset = chunk_index * chunk_size;
        const size_t current_chunk_size =
            std::min(chunk_size, input_size - offset);
        auto compressed =
            bsc_compress_block(input + offset, current_chunk_size, backend);
        result.compressed_chunk_sizes[chunk_index] =
            static_cast<uint64_t>(compressed.size());
        result.uncompressed_chunk_sizes[chunk_index] = current_chunk_size;
        compressed_chunks[chunk_index] = std::move(compressed);
      }
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
      std::lock_guard<std::mutex> lock(error_mutex);
      if (worker_error == nullptr) {
        worker_error = std::current_exception();
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (size_t i = 0; i < worker_count; ++i) {
    workers.emplace_back(worker);
  }
  for (auto &worker_thread : workers) {
    worker_thread.join();
  }
  if (worker_error != nullptr) {
    std::rethrow_exception(worker_error);
  }

  size_t total_compressed_size = 0;
  for (uint64_t size : result.compressed_chunk_sizes) {
    total_compressed_size += static_cast<size_t>(size);
  }
  result.data.reserve(total_compressed_size);
  for (auto &compressed : compressed_chunks) {
    result.data.insert(result.data.end(), compressed.begin(), compressed.end());
  }

  return result;
}

std::vector<uint8_t>
bsc_decompress_chunked(const std::vector<uint8_t> &compressed,
                       const std::vector<uint64_t> &compressed_chunk_sizes,
                       const std::vector<uint64_t> &uncompressed_chunk_sizes,
                       uint64_t expected_size, const BscConfig &config) {
  if (compressed.empty()) {
    if (!compressed_chunk_sizes.empty() || !uncompressed_chunk_sizes.empty() ||
        expected_size != 0) {
      throw std::runtime_error(
          "BSC chunk metadata is inconsistent for empty payload");
    }
    return {};
  }

  if (compressed_chunk_sizes.size() != uncompressed_chunk_sizes.size()) {
    throw std::runtime_error("BSC chunk metadata sizes do not match");
  }

  const auto resolved =
      resolve_bsc_config(config, compressed_chunk_sizes.size());
  const BscBackend backend = resolved.backend;
  const int features = bsc_features_for_backend(backend);
  bsc_check_init(backend);

  std::vector<uint8_t> output(expected_size);
  std::vector<size_t> compressed_offsets(compressed_chunk_sizes.size());
  std::vector<size_t> uncompressed_offsets(uncompressed_chunk_sizes.size());

  size_t compressed_offset = 0;
  size_t uncompressed_offset = 0;
  for (size_t i = 0; i < compressed_chunk_sizes.size(); ++i) {
    compressed_offsets[i] = compressed_offset;
    uncompressed_offsets[i] = uncompressed_offset;

    const uint64_t compressed_chunk_size = compressed_chunk_sizes[i];
    const uint64_t uncompressed_chunk_size = uncompressed_chunk_sizes[i];
    if (compressed_chunk_size >
            static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        uncompressed_chunk_size >
            static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("BSC chunk exceeds supported size");
    }
    if (compressed_offset + compressed_chunk_size > compressed.size()) {
      throw std::runtime_error("BSC compressed chunk metadata exceeds payload");
    }
    if (uncompressed_offset + uncompressed_chunk_size > output.size()) {
      throw std::runtime_error("BSC uncompressed chunk metadata exceeds output");
    }

    compressed_offset += static_cast<size_t>(compressed_chunk_size);
    uncompressed_offset += static_cast<size_t>(uncompressed_chunk_size);
  }
  if (compressed_offset != compressed.size() ||
      uncompressed_offset != expected_size) {
    throw std::runtime_error("BSC chunked decompression produced an unexpected size");
  }

  std::atomic<size_t> next_chunk{0};
  std::exception_ptr worker_error;
  std::atomic<bool> failed{false};
  std::mutex error_mutex;
  const size_t worker_count = resolved.parallelism;

  const auto worker = [&]() {
    try {
      while (!failed.load(std::memory_order_relaxed)) {
        const size_t chunk_index = next_chunk.fetch_add(1, std::memory_order_relaxed);
        if (chunk_index >= compressed_chunk_sizes.size()) {
          return;
        }

        const uint64_t compressed_chunk_size = compressed_chunk_sizes[chunk_index];
        const uint64_t uncompressed_chunk_size =
            uncompressed_chunk_sizes[chunk_index];
        const size_t comp_offset = compressed_offsets[chunk_index];
        const size_t uncomp_offset = uncompressed_offsets[chunk_index];

        int block_size = 0;
        int data_size = 0;
        const int info_result = bsc_block_info(
            compressed.data() + comp_offset, static_cast<int>(compressed_chunk_size),
            &block_size, &data_size, features);
        if (info_result != LIBBSC_NO_ERROR) {
          throw std::runtime_error("BSC block info failed with code: " +
                                   std::to_string(info_result));
        }
        if (static_cast<uint64_t>(block_size) != compressed_chunk_size ||
            static_cast<uint64_t>(data_size) != uncompressed_chunk_size) {
          throw std::runtime_error(
              "BSC chunk metadata does not match block header");
        }

        const int decomp_result = bsc_decompress(
            compressed.data() + comp_offset, static_cast<int>(compressed_chunk_size),
            output.data() + uncomp_offset,
            static_cast<int>(uncompressed_chunk_size),
            features);
        if (decomp_result != LIBBSC_NO_ERROR) {
          throw std::runtime_error("BSC decompression failed with code: " +
                                   std::to_string(decomp_result));
        }
      }
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
      std::lock_guard<std::mutex> lock(error_mutex);
      if (worker_error == nullptr) {
        worker_error = std::current_exception();
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (size_t i = 0; i < worker_count; ++i) {
    workers.emplace_back(worker);
  }
  for (auto &worker_thread : workers) {
    worker_thread.join();
  }
  if (worker_error != nullptr) {
    std::rethrow_exception(worker_error);
  }

  return output;
}

} // namespace gpufastq
