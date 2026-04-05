#include "gpufastq/compression_workflow.hpp"

#include "gpufastq/fastq_parser.hpp"
#include "gpufastq/gpu_compressor.hpp"
#include "gpufastq/serializer.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace gpufastq::workflow {

namespace {

long long
elapsed_ms(const std::chrono::high_resolution_clock::time_point &start,
           const std::chrono::high_resolution_clock::time_point &end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

size_t compressed_identifier_size(const CompressedIdentifierData &identifiers) {
  size_t total = identifiers.flat_data.payload.size();
  for (const auto &column : identifiers.columns) {
    total += column.values.payload.size();
    total += column.lengths.payload.size();
  }
  return total;
}

} // namespace

int compress(const std::string &input_path, const std::string &output_path,
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
  serialize_header(outfile);

  size_t chunk_size = bsc_config.chunk_size_gb * 1024ULL * 1024ULL * 1024ULL;
  if (chunk_size == 0)
    chunk_size = 8ULL * 1024 * 1024 * 1024ULL;

  uint64_t total_records = 0;
  size_t total_raw_payload = 0;
  size_t total_compressed_payload = 0;
  long long t_parse = 0;
  long long t_compress = 0;
  long long t_serialize = 0;

  if (bsc_config.backend != BscBackend::Default) {
    std::cerr << "BSC backend: " << bsc_backend_name(bsc_config.backend)
              << "\n";
  }

  int chunk_idx = 1;
  while (!infile.eof() && infile.peek() != std::char_traits<char>::eof()) {
    std::cerr << "=== Processing Chunk " << chunk_idx++ << " ===\n";
    const auto t_chunk_start = clock::now();
    auto data = parse_fastq_chunk(infile, chunk_size, bsc_config.stat_mode,
                                  bsc_config.log_stat_path);
    if (data.num_records == 0)
      break;

    const auto stats = compute_field_stats(data);
    const auto t_chunk_parsed = clock::now();
    t_parse += elapsed_ms(t_chunk_start, t_chunk_parsed);

    total_records += data.num_records;
    total_raw_payload += stats.identifiers_size + stats.basecalls_size +
                         stats.quality_scores_size + stats.line_length_size;

    const auto compressed =
        compress_fastq(data, DEFAULT_CHUNK_SIZE, bsc_config);
    const auto t_chunk_compressed = clock::now();
    t_compress += elapsed_ms(t_chunk_parsed, t_chunk_compressed);

    total_compressed_payload +=
        compressed_identifier_size(compressed.identifiers) +
        compressed.basecalls.packed_bases.payload.size() +
        compressed.basecalls.n_counts.payload.size() +
        compressed.basecalls.n_positions.payload.size() +
        compressed.quality_scores.payload.size() +
        compressed.line_lengths.payload.size();

    serialize_chunk(outfile, compressed);
    const auto t_chunk_serialized = clock::now();
    t_serialize += elapsed_ms(t_chunk_compressed, t_chunk_serialized);
  }

  const auto t3 = clock::now();
  const size_t input_size = fs::file_size(input_path);
  const size_t output_size = fs::file_size(output_path);

  std::cerr << "\n=== Summary ===\n"
            << "  Total records:     " << total_records << "\n"
            << "  Input file:        " << input_size << " B\n"
            << "  Output file:       " << output_size << " B\n"
            << "  File ratio:        " << 100.0 * output_size / input_size
            << " %\n"
            << "  Raw payload:       " << total_raw_payload << " B\n"
            << "  Compressed payload:" << total_compressed_payload << " B\n"
            << "  Payload ratio:     "
            << 100.0 * total_compressed_payload / total_raw_payload << " %\n"
            << "  Parse time:        " << t_parse << " ms\n"
            << "  Compress time:     " << t_compress << " ms\n"
            << "  Serialize time:    " << t_serialize << " ms\n"
            << "  Total time:        " << elapsed_ms(t0, t3) << " ms\n";
  return 0;
}

int roundtrip(const std::string &input_path, const BscConfig &bsc_config) {
  std::cerr << "=== Round-trip verification ===\n";

  std::ifstream infile(input_path, std::ios::binary);
  if (!infile.is_open()) {
    std::cerr << "Cannot open input file: " << input_path << "\n";
    return 1;
  }

  const std::string tmp_path = input_path + ".cuvf.tmp";
  std::ofstream outfile(tmp_path, std::ios::binary);
  serialize_header(outfile);

  size_t chunk_size = bsc_config.chunk_size_gb * 1024ULL * 1024ULL * 1024ULL;
  if (chunk_size == 0)
    chunk_size = 8ULL * 1024 * 1024 * 1024ULL;

  bool ok = true;
  int chunk_idx = 1;

  while (!infile.eof() && infile.peek() != std::char_traits<char>::eof()) {
    std::cerr << "Chunk " << chunk_idx++ << "...\n";
    auto original = parse_fastq_chunk(infile, chunk_size, bsc_config.stat_mode,
                                      bsc_config.log_stat_path);
    if (original.num_records == 0)
      break;

    const auto compressed =
        compress_fastq(original, DEFAULT_CHUNK_SIZE, bsc_config);
    serialize_chunk(outfile, compressed);

    // We can't decompress until everything is written, or we could stream from
    // a string. However, roundtrip here is simplified. For chunked roundtrip,
    // we'll write everything, then read back.
  }
  outfile.close();

  // Now verify
  std::ifstream decoded_in(tmp_path, std::ios::binary);
  const uint32_t format_version = deserialize_header(decoded_in);

  std::ifstream orig_in(input_path, std::ios::binary);

  chunk_idx = 1;
  while (!orig_in.eof() && orig_in.peek() != std::char_traits<char>::eof()) {
    auto original = parse_fastq_chunk(orig_in, chunk_size, false);
    if (original.num_records == 0)
      break;

    CompressedFastqData compressed;
    if (!deserialize_chunk(decoded_in, compressed, format_version)) {
      std::cerr << "FAIL: Output stream ended prematurely on chunk "
                << chunk_idx << "\n";
      ok = false;
      break;
    }

    const auto decoded = decompress_fastq(compressed, bsc_config);

    if (original.num_records != decoded.num_records) {
      std::cerr << "FAIL: chunk " << chunk_idx << " record count\n";
      ok = false;
    }
    if (original.raw_bytes != decoded.raw_bytes) {
      std::cerr << "FAIL: chunk " << chunk_idx << " raw FASTQ bytes\n";
      ok = false;
    }
    if (original.line_offsets != decoded.line_offsets) {
      std::cerr << "FAIL: chunk " << chunk_idx << " line offsets\n";
      ok = false;
    }
    if (!ok)
      break;
    chunk_idx++;
  }

  // Ensure decoded_in doesn't have more chunks
  CompressedFastqData dummy;
  if (ok && deserialize_chunk(decoded_in, dummy, format_version)) {
    std::cerr << "FAIL: Output stream has unexpected extra chunks\n";
    ok = false;
  }

  fs::remove(tmp_path);

  std::cerr << (ok ? "\nROUND-TRIP: PASSED ✓\n" : "\nROUND-TRIP: FAILED ✗\n");
  return ok ? 0 : 1;
}

} // namespace gpufastq::workflow
