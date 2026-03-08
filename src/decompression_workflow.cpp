#include "gpufastq/decompression_workflow.hpp"

#include "gpufastq/fastq_parser.hpp"
#include "gpufastq/gpu_compressor.hpp"
#include "gpufastq/serializer.hpp"

#include <chrono>
#include <fstream>
#include <iostream>

namespace gpufastq::workflow {

namespace {

long long
elapsed_ms(const std::chrono::high_resolution_clock::time_point &start,
           const std::chrono::high_resolution_clock::time_point &end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

} // namespace

int decompress(const std::string &input_path, const std::string &output_path,
               const BscConfig &bsc_config) {
  using clock = std::chrono::high_resolution_clock;
  const auto t0 = clock::now();

  std::ifstream infile(input_path, std::ios::binary);
  if (!infile.is_open()) {
    std::cerr << "Cannot open input file: " << input_path << "\n";
    return 1;
  }

  std::ofstream outfile(output_path, std::ios::binary);
  if (!outfile.is_open()) {
    std::cerr << "Cannot open output file: " << output_path << "\n";
    return 1;
  }

  std::cerr << "=== Starting GPU Decompression ===\n";
  if (bsc_config.backend != BscBackend::Default) {
    std::cerr << "BSC backend: " << bsc_backend_name(bsc_config.backend)
              << "\n";
  }

  deserialize_header(infile);

  uint64_t total_records = 0;
  long long t_deserialize = 0;
  long long t_decompress = 0;
  long long t_write = 0;

  int chunk_idx = 1;
  while (true) {
    const auto t_chunk_start = clock::now();
    CompressedFastqData compressed;
    if (!deserialize_chunk(infile, compressed)) {
      break;
    }
    const auto t_chunk_deserialized = clock::now();
    t_deserialize += elapsed_ms(t_chunk_start, t_chunk_deserialized);

    std::cerr << "Chunk " << chunk_idx++
              << " - Records: " << compressed.num_records << "\n";
    total_records += compressed.num_records;

    const auto data = decompress_fastq(compressed, bsc_config);
    const auto t_chunk_decompressed = clock::now();
    t_decompress += elapsed_ms(t_chunk_deserialized, t_chunk_decompressed);

    if (!data.raw_bytes.empty()) {
      outfile.write(reinterpret_cast<const char *>(data.raw_bytes.data()),
                    static_cast<std::streamsize>(data.raw_bytes.size()));
    }
    const auto t_chunk_written = clock::now();
    t_write += elapsed_ms(t_chunk_decompressed, t_chunk_written);
  }

  const auto t3 = clock::now();

  std::cerr << "\n=== Summary ===\n"
            << "  Total records:    " << total_records << "\n"
            << "  Deserialize time: " << t_deserialize << " ms\n"
            << "  Decompress time:  " << t_decompress << " ms\n"
            << "  Write time:       " << t_write << " ms\n"
            << "  Total time:       " << elapsed_ms(t0, t3) << " ms\n";
  return 0;
}

} // namespace gpufastq::workflow
