#include "gpufastq/compression_workflow.hpp"
#include "gpufastq/decompression_workflow.hpp"

#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

size_t parse_positive_size_arg(const std::string &flag,
                               const std::string &value) {
  size_t pos = 0;
  const unsigned long long parsed = std::stoull(value, &pos, 10);
  if (pos != value.size() || parsed == 0 ||
      parsed >
          static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error(flag + " must be a positive integer");
  }
  return static_cast<size_t>(parsed);
}

gpufastq::BscBackend parse_bsc_backend_arg(const std::string &value) {
  if (value == "cpu") {
    return gpufastq::BscBackend::Cpu;
  }
  if (value == "cuda") {
    return gpufastq::BscBackend::Cuda;
  }
  throw std::runtime_error("--bsc-backend must be 'cpu' or 'cuda'");
}

gpufastq::QualityCodec parse_quality_codec_arg(const std::string &value) {
  if (value == "bsc") {
    return gpufastq::QualityCodec::Bsc;
  }
  if (value == "zstd") {
    return gpufastq::QualityCodec::Zstd;
  }
  throw std::runtime_error("--quality-codec must be 'bsc' or 'zstd'");
}

gpufastq::BasecallPackOrder parse_basecall_pack_order_arg(
    const std::string &value) {
  if (value == "tgca") {
    return gpufastq::BasecallPackOrder::Tgca;
  }
  if (value == "acgt") {
    return gpufastq::BasecallPackOrder::Acgt;
  }
  throw std::runtime_error("--base-pack-order must be 'tgca' or 'acgt'");
}

} // namespace

void print_usage(const char *prog) {
  std::cerr
      << "GPUFastQ — GPU-accelerated FASTQ compression (nvcomp zstd)\n\n"
      << "Usage:\n"
      << "  " << prog << " compress   [options] <input.fastq> <output.cuvf>\n"
      << "  " << prog << " decompress [options] <input.cuvf> <output.fastq>\n"
      << "  " << prog << " roundtrip  [options] <input.fastq>\n\n"
      << "Options:\n"
      << "  --quality-codec C  Quality codec: bsc or zstd (default: bsc)\n"
      << "  --bsc-backend B    BSC backend for quality scores (and basecalls "
         "if enabled): cpu or cuda (default: env GPUFASTQ_BSC_BACKEND, "
         "else cpu)\n"
      << "  --bsc-threads N    Override CPU worker count for CPU BSC mode "
         "(default: env GPUFASTQ_BSC_THREADS, else auto)\n"
      << "  --bsc-gpu-jobs N   Override concurrent chunk jobs for CUDA BSC "
         "mode (default: env GPUFASTQ_BSC_GPU_JOBS, else 32)\n"
      << "  --base-bsc         Use libbsc instead of nvcomp for packed "
         "basecall compression (default: off)\n"
      << "  --base-pack-order O Pack A/C/G/T as tgca or acgt "
         "(default: tgca)\n"
      << "  --transpose        Transpose fixed-length quality scores before "
         "Zstd compression (default: off)\n"
      << "  --stat             Print detailed timing statistics for each "
         "stage (default: off)\n"
      << "  --log-stat F       Log detailed timing and throughput statistics "
         "to a file (default: disabled)\n"
      << "  --chunk-size N     Chunk size in GB for processing (default: 8)\n\n"
      << "Environment:\n"
      << "  GPUFASTQ_BSC_BACKEND  Default BSC backend when --bsc-backend is "
         "not set\n"
      << "  GPUFASTQ_BSC_THREADS  Default CPU worker count when --bsc-threads "
         "is not set\n"
      << "  GPUFASTQ_BSC_GPU_JOBS Default CUDA chunk job count when "
         "--bsc-gpu-jobs is not set\n"
      << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string cmd = argv[1];
  try {
    gpufastq::BscConfig bsc_config;
    std::vector<const char *> positional_args;
    positional_args.reserve(static_cast<size_t>(argc));

    for (int argi = 2; argi < argc; ++argi) {
      const std::string arg = argv[argi];
      if (arg == "--transpose") {
        bsc_config.zstd_transpose_quality = true;
        continue;
      }
      if (arg == "--stat") {
        bsc_config.stat_mode = true;
        continue;
      }
      if (arg == "--base-bsc") {
        bsc_config.base_bsc = true;
        continue;
      }
      if (arg == "--base-pack-order") {
        if (argi + 1 >= argc) {
          throw std::runtime_error("Missing value for " + arg);
        }
        bsc_config.basecall_pack_order =
            parse_basecall_pack_order_arg(argv[++argi]);
        continue;
      }
      if (arg == "--log-stat") {
        if (argi + 1 >= argc) {
          throw std::runtime_error("Missing value for " + arg);
        }
        bsc_config.log_stat_path = argv[++argi];
        continue;
      }
      if (arg == "--quality-codec" || arg == "--bsc-backend" ||
          arg == "--bsc-threads" || arg == "--bsc-gpu-jobs" ||
          arg == "--chunk-size") {
        if (argi + 1 >= argc) {
          throw std::runtime_error("Missing value for " + arg);
        }
        const std::string value = argv[++argi];
        if (arg == "--quality-codec") {
          bsc_config.quality_codec = parse_quality_codec_arg(value);
        } else if (arg == "--bsc-backend") {
          bsc_config.backend = parse_bsc_backend_arg(value);
        } else if (arg == "--bsc-threads") {
          bsc_config.threads = parse_positive_size_arg(arg, value);
        } else if (arg == "--bsc-gpu-jobs") {
          bsc_config.gpu_jobs = parse_positive_size_arg(arg, value);
        } else {
          bsc_config.chunk_size_gb = parse_positive_size_arg(arg, value);
        }
        continue;
      }
      positional_args.push_back(argv[argi]);
    }

    if (cmd == "compress" && positional_args.size() >= 2)
      return gpufastq::workflow::compress(positional_args[0],
                                          positional_args[1], bsc_config);
    if (cmd == "decompress" && positional_args.size() >= 2)
      return gpufastq::workflow::decompress(positional_args[0],
                                            positional_args[1], bsc_config);
    if (cmd == "roundtrip" && positional_args.size() >= 1)
      return gpufastq::workflow::roundtrip(positional_args[0], bsc_config);

    print_usage(argv[0]);
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
