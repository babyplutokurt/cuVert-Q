#include "gpufastq/codec_bsc.hpp"
#include "gpufastq/compression_workflow.hpp"
#include "gpufastq/fastq_parser.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using sys_clock_t = std::chrono::high_resolution_clock;

void write_to_file(std::ofstream &ofs, const gpufastq::BscChunkedBuffer &buf) {
  if (!buf.data.empty()) {
    ofs.write(reinterpret_cast<const char *>(buf.data.data()), buf.data.size());
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <input.fastq> [options]\n";
    std::cerr << "Options:\n"
              << "  --bsc-threads N\n"
              << "  --bsc-gpu-jobs N\n"
              << "  --base-bsc\n"
              << "  --bsc-backend <cpu|cuda>\n"
              << "  --chunk-size N\n"
              << "  --stat\n"
              << "  --log-stat F\n";
    return 1;
  }

  std::vector<std::string> positional_args;
  gpufastq::BscConfig bsc_config;
  bsc_config.quality_codec =
      gpufastq::QualityCodec::Bsc; // Ensure BSC is used for the Cu-Ratio pass

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--bsc-threads" && i + 1 < argc) {
      bsc_config.threads = std::stoull(argv[++i]);
    } else if (arg == "--bsc-gpu-jobs" && i + 1 < argc) {
      bsc_config.gpu_jobs = std::stoull(argv[++i]);
    } else if (arg == "--base-bsc") {
      bsc_config.base_bsc = true;
    } else if (arg == "--bsc-backend" && i + 1 < argc) {
      std::string val = argv[++i];
      if (val == "cpu")
        bsc_config.backend = gpufastq::BscBackend::Cpu;
      else if (val == "cuda")
        bsc_config.backend = gpufastq::BscBackend::Cuda;
    } else if (arg == "--chunk-size" && i + 1 < argc) {
      bsc_config.chunk_size_gb = std::stoull(argv[++i]);
    } else if (arg == "--stat") {
      bsc_config.stat_mode = true;
    } else if (arg == "--log-stat" && i + 1 < argc) {
      bsc_config.log_stat_path = argv[++i];
    } else {
      positional_args.push_back(arg);
    }
  }

  if (positional_args.empty()) {
    std::cerr << "Error: Missing input path.\n";
    return 1;
  }
  std::string input_path = positional_args[0];

  std::ifstream infile(input_path, std::ios::binary);
  if (!infile.is_open()) {
    std::cerr << "Cannot open " << input_path << "\n";
    return 1;
  }
  infile.close(); // just checking

  size_t chunk_size = bsc_config.chunk_size_gb * 1024ULL * 1024ULL * 1024ULL;
  if (chunk_size == 0)
    chunk_size = 8ULL * 1024 * 1024 * 1024ULL;

  std::string raw_out = input_path + ".raw.bsc";
  std::string split_id_out = input_path + ".split.id.bsc";
  std::string split_base_out = input_path + ".split.base.bsc";
  std::string split_plus_out = input_path + ".split.plus.bsc";
  std::string split_qual_out = input_path + ".split.qual.bsc";
  std::string full_out = input_path + ".full.cuvf";

  size_t orig_size = fs::file_size(input_path);
  std::cerr << "--- Ablation Study ---" << std::endl;
  std::cerr << "Input file: " << input_path << std::endl;
  std::cerr << "Original size: " << orig_size << " B" << std::endl;

  // 1. Raw + libbsc
  std::cerr << "\n[1/3] Running Raw + libbsc..." << std::endl;
  double raw_time = 0.0;
  {
    auto t_start = sys_clock_t::now();
    std::ifstream in(input_path, std::ios::binary);
    std::ofstream out_raw(raw_out, std::ios::binary);

    int c = 1;
    while (!in.eof() && in.peek() != std::char_traits<char>::eof()) {
      std::cerr << "  Chunk " << c++ << "..." << std::endl;
      auto data = gpufastq::parse_fastq_chunk(in, chunk_size, false, "");
      if (data.num_records == 0)
        break;

      auto bsc = gpufastq::bsc_compress_chunked(
          data.raw_bytes.data(), data.raw_bytes.size(),
          gpufastq::BSC_QUALITY_CHUNK_SIZE, bsc_config);
      write_to_file(out_raw, bsc);
    }
    raw_time =
        std::chrono::duration<double>(sys_clock_t::now() - t_start).count();
  }
  size_t raw_size = fs::file_size(raw_out);
  std::cerr << "  -> Raw Compressed Size: " << raw_size << " B" << std::endl;

  size_t split_id_raw = 0;
  size_t split_base_raw = 0;
  size_t split_plus_raw = 0;
  size_t split_qual_raw = 0;

  double split_total_time = 0.0;
  double split_id_time = 0.0;
  double split_base_time = 0.0;
  double split_plus_time = 0.0;
  double split_qual_time = 0.0;

  // 2. Split + libbsc
  std::cerr << "\n[2/3] Running Split + libbsc..." << std::endl;
  {
    auto t_start = sys_clock_t::now();
    std::ifstream in(input_path, std::ios::binary);
    std::ofstream out_id(split_id_out, std::ios::binary);
    std::ofstream out_base(split_base_out, std::ios::binary);
    std::ofstream out_plus(split_plus_out, std::ios::binary);
    std::ofstream out_qual(split_qual_out, std::ios::binary);

    int c = 1;
    while (!in.eof() && in.peek() != std::char_traits<char>::eof()) {
      std::cerr << "  Chunk " << c++ << "..." << std::endl;
      auto data = gpufastq::parse_fastq_chunk(in, chunk_size, false, "");
      if (data.num_records == 0)
        break;

      std::vector<uint8_t> ids, bases, pluses, quals;
      size_t est = data.raw_bytes.size() / 4;
      ids.reserve(est);
      bases.reserve(est);
      pluses.reserve(est);
      quals.reserve(est);

      for (uint64_t r = 0; r < data.num_records; ++r) {
        for (int line = 0; line < 4; ++line) {
          uint64_t start_off = data.line_offsets[r * 4 + line];
          uint64_t end_off = data.line_offsets[r * 4 + line + 1];
          if (line == 0)
            ids.insert(ids.end(), data.raw_bytes.begin() + start_off,
                       data.raw_bytes.begin() + end_off);
          else if (line == 1)
            bases.insert(bases.end(), data.raw_bytes.begin() + start_off,
                         data.raw_bytes.begin() + end_off);
          else if (line == 2)
            pluses.insert(pluses.end(), data.raw_bytes.begin() + start_off,
                          data.raw_bytes.begin() + end_off);
          else if (line == 3)
            quals.insert(quals.end(), data.raw_bytes.begin() + start_off,
                         data.raw_bytes.begin() + end_off);
        }
      }

      split_id_raw += ids.size();
      split_base_raw += bases.size();
      split_plus_raw += pluses.size();
      split_qual_raw += quals.size();

      auto t0 = sys_clock_t::now();
      write_to_file(out_id, gpufastq::bsc_compress_chunked(
                                ids.data(), ids.size(),
                                gpufastq::BSC_QUALITY_CHUNK_SIZE, bsc_config));
      auto t1 = sys_clock_t::now();
      split_id_time += std::chrono::duration<double>(t1 - t0).count();

      t0 = sys_clock_t::now();
      write_to_file(out_base,
                    gpufastq::bsc_compress_chunked(
                        bases.data(), bases.size(),
                        gpufastq::BSC_QUALITY_CHUNK_SIZE, bsc_config));
      t1 = sys_clock_t::now();
      split_base_time += std::chrono::duration<double>(t1 - t0).count();

      t0 = sys_clock_t::now();
      write_to_file(out_plus,
                    gpufastq::bsc_compress_chunked(
                        pluses.data(), pluses.size(),
                        gpufastq::BSC_QUALITY_CHUNK_SIZE, bsc_config));
      t1 = sys_clock_t::now();
      split_plus_time += std::chrono::duration<double>(t1 - t0).count();

      t0 = sys_clock_t::now();
      write_to_file(out_qual,
                    gpufastq::bsc_compress_chunked(
                        quals.data(), quals.size(),
                        gpufastq::BSC_QUALITY_CHUNK_SIZE, bsc_config));
      t1 = sys_clock_t::now();
      split_qual_time += std::chrono::duration<double>(t1 - t0).count();
    }
    split_total_time =
        std::chrono::duration<double>(sys_clock_t::now() - t_start).count();
  }
  size_t split_id_comp = fs::file_size(split_id_out);
  size_t split_base_comp = fs::file_size(split_base_out);
  size_t split_plus_comp = fs::file_size(split_plus_out);
  size_t split_qual_comp = fs::file_size(split_qual_out);
  size_t split_size =
      split_id_comp + split_base_comp + split_plus_comp + split_qual_comp;
  std::cerr << "  -> Split Compressed Size: " << split_size << " B"
            << std::endl;

  // 3. Full Cu-Ratio (GPUFastQ)
  std::cerr << "\n[3/3] Running Full Cu-Ratio (GPUFastQ) + libbsc..."
            << std::endl;
  gpufastq::workflow::compress(input_path, full_out, bsc_config);
  size_t full_size = fs::file_size(full_out);
  std::cerr << "  -> Full Compressed Size: " << full_size << " B" << std::endl;

  std::cerr << "\n============================================\n";
  std::cerr << "        Final Ablation Results              \n";
  std::cerr << "============================================\n";
  std::cerr << "Original Data:  " << orig_size << " B\n";
  std::cerr << "Raw + BSC:      " << raw_size << " B ("
            << 100.0 * raw_size / orig_size << "%) in " << raw_time << " s\n";
  std::cerr << "Split + BSC:    " << split_size << " B ("
            << 100.0 * split_size / orig_size << "%) in " << split_total_time
            << " s\n";
  if (split_id_raw > 0)
    std::cerr << "  - Identifiers: " << split_id_comp << " / " << split_id_raw
              << " B (" << 100.0 * split_id_comp / split_id_raw << "%) in "
              << split_id_time << " s\n";
  if (split_base_raw > 0)
    std::cerr << "  - Basecalls:   " << split_base_comp << " / "
              << split_base_raw << " B ("
              << 100.0 * split_base_comp / split_base_raw << "%) in "
              << split_base_time << " s\n";
  if (split_plus_raw > 0)
    std::cerr << "  - Plus lines:  " << split_plus_comp << " / "
              << split_plus_raw << " B ("
              << 100.0 * split_plus_comp / split_plus_raw << "%) in "
              << split_plus_time << " s\n";
  if (split_qual_raw > 0)
    std::cerr << "  - Qualities:   " << split_qual_comp << " / "
              << split_qual_raw << " B ("
              << 100.0 * split_qual_comp / split_qual_raw << "%) in "
              << split_qual_time << " s\n";
  std::cerr << "Full Pipeline:  " << full_size << " B ("
            << 100.0 * full_size / orig_size << "%)\n\n";

  // Clean the compressed cache files
  std::cerr << "Cleaning up cache files..." << std::endl;
  fs::remove(raw_out);
  fs::remove(split_id_out);
  fs::remove(split_base_out);
  fs::remove(split_plus_out);
  fs::remove(split_qual_out);
  fs::remove(full_out);
  std::cerr << "Cleanup done.\n";

  return 0;
}
