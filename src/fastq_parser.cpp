#include "gpufastq/fastq_parser.hpp"
#include "gpufastq/codec_gpu.cuh"

#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace gpufastq {

namespace {

constexpr uint64_t IDENTIFIER_DISCOVERY_RECORDS = 100;

bool is_identifier_separator(uint8_t ch) {
  return ch == ':' || ch == '/' || ch == '-' || ch == '.' ||
         std::isspace(static_cast<unsigned char>(ch)) != 0;
}

struct TokenizedIdentifier {
  std::vector<std::string> tokens;
  std::vector<std::string> separators;
};

uint64_t line_content_length(const std::vector<uint64_t> &line_offsets,
                             uint64_t line_idx);

TokenizedIdentifier tokenize_identifier(const uint8_t *data, size_t size) {
  TokenizedIdentifier out;
  std::string token;
  std::string separator;
  bool in_separator = false;

  for (size_t i = 0; i < size; ++i) {
    const char ch = static_cast<char>(data[i]);
    if (is_identifier_separator(data[i])) {
      if (!token.empty()) {
        out.tokens.push_back(std::move(token));
        token.clear();
      }
      separator.push_back(ch);
      in_separator = true;
      continue;
    }

    if (in_separator) {
      out.separators.push_back(std::move(separator));
      separator.clear();
      in_separator = false;
    }
    token.push_back(ch);
  }

  if (!token.empty()) {
    out.tokens.push_back(std::move(token));
  }
  if (!separator.empty()) {
    out.separators.push_back(std::move(separator));
  }

  return out;
}

bool token_is_int32(const std::string &token) {
  if (token.empty()) {
    return false;
  }

  size_t index = 0;
  if (token[0] == '+' || token[0] == '-') {
    if (token.size() == 1) {
      return false;
    }
    index = 1;
  }

  for (; index < token.size(); ++index) {
    if (!std::isdigit(static_cast<unsigned char>(token[index]))) {
      return false;
    }
  }

  try {
    const long long value = std::stoll(token);
    if (std::to_string(value) != token) {
      return false;
    }
    return value >= std::numeric_limits<int32_t>::min() &&
           value <= std::numeric_limits<int32_t>::max();
  } catch (...) {
    return false;
  }
}

IdentifierLayout discover_identifier_layout(const FastqData &data) {
  IdentifierLayout layout;
  const uint64_t sample_records =
      std::min<uint64_t>(IDENTIFIER_DISCOVERY_RECORDS, data.num_records);
  if (sample_records == 0) {
    return layout;
  }

  const uint64_t first_len = line_content_length(data.line_offsets, 0);
  if (first_len <= 1) {
    return layout;
  }

  const uint64_t first_start = data.line_offsets[0] + 1;
  auto reference =
      tokenize_identifier(data.raw_bytes.data() + first_start, first_len - 1);
  if (reference.tokens.empty() ||
      reference.separators.size() + 1 != reference.tokens.size()) {
    return layout;
  }

  std::vector<bool> numeric(reference.tokens.size(), true);
  for (size_t i = 0; i < reference.tokens.size(); ++i) {
    numeric[i] = token_is_int32(reference.tokens[i]);
  }

  for (uint64_t record = 1; record < sample_records; ++record) {
    const uint64_t id_line = 4 * record;
    const uint64_t id_len = line_content_length(data.line_offsets, id_line);
    if (id_len <= 1) {
      return {};
    }

    const uint64_t id_start = data.line_offsets[id_line] + 1;
    auto tokenized =
        tokenize_identifier(data.raw_bytes.data() + id_start, id_len - 1);
    if (tokenized.tokens.size() != reference.tokens.size() ||
        tokenized.separators != reference.separators) {
      return {};
    }
    for (size_t i = 0; i < tokenized.tokens.size(); ++i) {
      numeric[i] = numeric[i] && token_is_int32(tokenized.tokens[i]);
    }
  }

  layout.columnar = true;
  layout.separators = std::move(reference.separators);
  layout.column_kinds.reserve(reference.tokens.size());
  for (bool is_numeric : numeric) {
    layout.column_kinds.push_back(is_numeric ? IdentifierColumnKind::Int32
                                             : IdentifierColumnKind::String);
  }
  return layout;
}

uint64_t line_content_length(const std::vector<uint64_t> &line_offsets,
                             uint64_t line_idx) {
  const uint64_t line_start = line_offsets[line_idx];
  const uint64_t next_line_start = line_offsets[line_idx + 1];
  if (next_line_start <= line_start) {
    throw std::runtime_error("FASTQ line offsets are not strictly increasing");
  }
  return next_line_start - line_start - 1;
}

void validate_fastq_layout(const FastqData &data) {
  if (data.raw_bytes.empty()) {
    if (data.line_offsets.size() != 1 || data.line_offsets[0] != 0 ||
        data.num_records != 0) {
      throw std::runtime_error("Empty FASTQ metadata is inconsistent");
    }
    return;
  }

  if (data.raw_bytes.back() != '\n') {
    throw std::runtime_error("FASTQ file must end with a newline");
  }
  if (data.line_offsets.empty() || data.line_offsets.front() != 0) {
    throw std::runtime_error("FASTQ line index must start at byte offset 0");
  }
  if (data.line_offsets.back() != data.raw_bytes.size()) {
    throw std::runtime_error(
        "FASTQ line index sentinel is inconsistent with the file size");
  }

  const uint64_t num_lines = data.line_offsets.size() - 1;
  if (num_lines % 4 != 0) {
    throw std::runtime_error(
        "FASTQ file does not contain a multiple of 4 lines");
  }
  if (data.num_records != num_lines / 4) {
    throw std::runtime_error(
        "FASTQ record count does not match the line index");
  }

  for (uint64_t record = 0; record < data.num_records; ++record) {
    const uint64_t id_line = 4 * record;
    const uint64_t seq_line = id_line + 1;
    const uint64_t plus_line = id_line + 2;
    const uint64_t qual_line = id_line + 3;

    const uint64_t id_start = data.line_offsets[id_line];
    const uint64_t seq_start = data.line_offsets[seq_line];
    const uint64_t plus_start = data.line_offsets[plus_line];
    const uint64_t qual_start = data.line_offsets[qual_line];

    if (data.raw_bytes[id_start] != '@') {
      throw std::runtime_error("Expected '@' at record " +
                               std::to_string(record + 1));
    }
    if (data.raw_bytes[plus_start] != '+') {
      throw std::runtime_error("Expected '+' at record " +
                               std::to_string(record + 1));
    }

    const uint64_t id_len = line_content_length(data.line_offsets, id_line);
    const uint64_t seq_len = line_content_length(data.line_offsets, seq_line);
    const uint64_t plus_len = line_content_length(data.line_offsets, plus_line);
    const uint64_t qual_len = line_content_length(data.line_offsets, qual_line);

    if (id_len <= 1) {
      throw std::runtime_error("Identifier line is empty at record " +
                               std::to_string(record + 1));
    }
    if (plus_len != 1) {
      throw std::runtime_error(
          "Only '+' separator lines without comments are supported");
    }
    if (seq_len != qual_len) {
      throw std::runtime_error("Sequence/quality length mismatch at record " +
                               std::to_string(record + 1));
    }
    if (seq_start != id_start + id_len + 1) {
      throw std::runtime_error(
          "Identifier line length does not match line index");
    }
    if (plus_start != seq_start + seq_len + 1) {
      throw std::runtime_error(
          "Sequence line length does not match line index");
    }
    if (qual_start != plus_start + plus_len + 1) {
      throw std::runtime_error("Plus line length does not match line index");
    }
  }
}

} // namespace

FastqData parse_fastq(const std::string &filepath, bool stat_mode,
                      const std::string &log_stat_path) {
  using clock = std::chrono::high_resolution_clock;
  const auto t0 = clock::now();

  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open FASTQ file: " + filepath);
  }

  const auto end_pos = file.tellg();
  if (end_pos < 0) {
    throw std::runtime_error("Cannot determine FASTQ file size: " + filepath);
  }

  FastqData data;
  data.raw_bytes.resize(static_cast<size_t>(end_pos));

  file.seekg(0, std::ios::beg);
  if (!data.raw_bytes.empty()) {
    file.read(reinterpret_cast<char *>(data.raw_bytes.data()),
              static_cast<std::streamsize>(data.raw_bytes.size()));
    if (!file) {
      throw std::runtime_error("Failed to read FASTQ file: " + filepath);
    }
  }

  if (data.raw_bytes.empty()) {
    data.line_offsets = {0};
    return data;
  }

  const auto t1 = clock::now();

  data.line_offsets = build_line_offsets_gpu(data.raw_bytes);
  data.num_records = (data.line_offsets.size() - 1) / 4;

  const auto t2 = clock::now();

  validate_fastq_layout(data);

  const auto t3 = clock::now();

  const auto quality_analysis =
      analyze_quality_lengths(data.line_offsets, data.num_records);
  data.quality_lengths = quality_analysis.lengths;
  data.quality_layout = quality_analysis.layout;
  data.fixed_quality_length = quality_analysis.fixed_length;

  const auto t4 = clock::now();
  data.identifier_layout = discover_identifier_layout(data);

  const auto t5 = clock::now();

  if (stat_mode || !log_stat_path.empty()) {
    auto ms = [](const auto &start, const auto &end) {
      return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
    };
    long t_read = ms(t0, t1);
    long t_idx = ms(t1, t2);
    long t_val = ms(t2, t3);
    long t_qual = ms(t3, t4);
    long t_schema = ms(t4, t5);

    if (stat_mode) {
      std::cerr << "Parser stage timings:" << std::endl;
      std::cerr << "  Read file:       " << t_read << " ms" << std::endl;
      std::cerr << "  Indexing:        " << t_idx << " ms" << std::endl;
      std::cerr << "  Validate layout: " << t_val << " ms" << std::endl;
      std::cerr << "  Analyze quality: " << t_qual << " ms" << std::endl;
      std::cerr << "  Discover schema: " << t_schema << " ms" << std::endl;
    }

    if (!log_stat_path.empty()) {
      std::ofstream log(log_stat_path, std::ios::app);
      if (log.is_open()) {
        log << "Parser:\n";
        log << "  Read file:       " << t_read << " ms, "
            << data.raw_bytes.size() << " B\n";
        log << "  Indexing:        " << t_idx << " ms, "
            << data.raw_bytes.size() << " B\n";
        log << "  Validate layout: " << t_val << " ms, "
            << data.raw_bytes.size() << " B\n";
        log << "  Analyze quality: " << t_qual << " ms, "
            << data.raw_bytes.size() << " B\n";
        log << "  Discover schema: " << t_schema << " ms, "
            << data.raw_bytes.size() << " B\n";
      }
    }
  }

  return data;
}

FastqData parse_fastq_chunk(std::ifstream &file, size_t chunk_size,
                            bool stat_mode, const std::string &log_stat_path) {
  using clock = std::chrono::high_resolution_clock;
  const auto t0 = clock::now();

  const std::streampos start_pos = file.tellg();
  FastqData data;
  data.raw_bytes.resize(chunk_size);
  file.read(reinterpret_cast<char *>(data.raw_bytes.data()),
            static_cast<std::streamsize>(chunk_size));
  const size_t bytes_read = file.gcount();
  data.raw_bytes.resize(bytes_read);

  if (data.raw_bytes.empty()) {
    data.line_offsets = {0};
    return data;
  }

  const auto t1 = clock::now();

  data.line_offsets = build_line_offsets_gpu(data.raw_bytes);
  bool is_eof = file.eof() && file.peek() == std::char_traits<char>::eof();

  if (!is_eof) {
    if (data.line_offsets.size() < 2) {
      throw std::runtime_error(
          "Chunk size too small to contain even a single newline");
    }
    data.num_records = (data.line_offsets.size() - 2) / 4;
    if (data.num_records == 0) {
      throw std::runtime_error(
          "Chunk size too small to contain a single FASTQ record");
    }
    const uint64_t consumed_bytes = data.line_offsets[data.num_records * 4];

    data.raw_bytes.resize(consumed_bytes);
    data.line_offsets.resize(data.num_records * 4 + 1);
    data.line_offsets.back() = consumed_bytes;

    file.clear(); // Clear EOF flag if set
    file.seekg(start_pos + static_cast<std::streampos>(consumed_bytes));
  } else {
    data.num_records = (data.line_offsets.size() - 1) / 4;
  }

  const auto t2 = clock::now();

  validate_fastq_layout(data);

  const auto t3 = clock::now();

  const auto quality_analysis =
      analyze_quality_lengths(data.line_offsets, data.num_records);
  data.quality_lengths = quality_analysis.lengths;
  data.quality_layout = quality_analysis.layout;
  data.fixed_quality_length = quality_analysis.fixed_length;

  const auto t4 = clock::now();
  data.identifier_layout = discover_identifier_layout(data);

  const auto t5 = clock::now();

  if (stat_mode || !log_stat_path.empty()) {
    auto ms = [](const auto &start, const auto &end) {
      return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
    };
    long t_read = ms(t0, t1);
    long t_idx = ms(t1, t2);
    long t_val = ms(t2, t3);
    long t_qual = ms(t3, t4);
    long t_schema = ms(t4, t5);

    if (stat_mode) {
      std::cerr << "Parser stage timings (chunk):" << std::endl;
      std::cerr << "  Read chunk:      " << t_read << " ms" << std::endl;
      std::cerr << "  Indexing:        " << t_idx << " ms" << std::endl;
      std::cerr << "  Validate layout: " << t_val << " ms" << std::endl;
      std::cerr << "  Analyze quality: " << t_qual << " ms" << std::endl;
      std::cerr << "  Discover schema: " << t_schema << " ms" << std::endl;
    }

    if (!log_stat_path.empty()) {
      std::ofstream log(log_stat_path, std::ios::app);
      if (log.is_open()) {
        log << "Parser (chunk):\n";
        log << "  Read chunk:      " << t_read << " ms, "
            << data.raw_bytes.size() << " B\n";
        log << "  Indexing:        " << t_idx << " ms, "
            << data.raw_bytes.size() << " B\n";
        log << "  Validate layout: " << t_val << " ms, "
            << data.raw_bytes.size() << " B\n";
        log << "  Analyze quality: " << t_qual << " ms, "
            << data.raw_bytes.size() << " B\n";
        log << "  Discover schema: " << t_schema << " ms, "
            << data.raw_bytes.size() << " B\n";
      }
    }
  }

  return data;
}

FastqFieldStats compute_field_stats(const FastqData &data) {
  validate_fastq_layout(data);

  FastqFieldStats stats;
  if (!data.line_offsets.empty()) {
    for (size_t i = 0; i + 1 < data.line_offsets.size(); ++i) {
      const uint64_t line_length =
          data.line_offsets[i + 1] - data.line_offsets[i];
      if (line_length > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("FASTQ line length exceeds uint32_t range");
      }
    }
    stats.line_length_size = data.line_offsets.size() * sizeof(uint32_t);
  }
  for (uint64_t record = 0; record < data.num_records; ++record) {
    const uint64_t id_line = 4 * record;
    const uint64_t seq_line = id_line + 1;

    stats.identifiers_size +=
        line_content_length(data.line_offsets, id_line) - 1;
    stats.basecalls_size += line_content_length(data.line_offsets, seq_line);
  }

  if (data.quality_lengths.size() == static_cast<size_t>(data.num_records)) {
    for (uint32_t quality_length : data.quality_lengths) {
      stats.quality_scores_size += quality_length;
    }
  } else {
    for (uint64_t record = 0; record < data.num_records; ++record) {
      const uint64_t qual_line = 4 * record + 3;
      stats.quality_scores_size +=
          line_content_length(data.line_offsets, qual_line);
    }
  }

  return stats;
}

void write_fastq(const std::string &filepath, const FastqData &data) {
  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open output file: " + filepath);
  }

  if (!data.raw_bytes.empty()) {
    file.write(reinterpret_cast<const char *>(data.raw_bytes.data()),
               static_cast<std::streamsize>(data.raw_bytes.size()));
    if (!file) {
      throw std::runtime_error("Failed to write FASTQ file: " + filepath);
    }
  }
}

} // namespace gpufastq
