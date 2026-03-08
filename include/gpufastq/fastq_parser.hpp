#pragma once

#include "fastq_record.hpp"
#include <string>

namespace gpufastq {

/// Parse a FASTQ file into a raw byte buffer and a line-start index.
FastqData parse_fastq(const std::string &filepath, bool stat_mode = false,
                      const std::string &log_stat_path = "");

/// Parse a chunk of a FASTQ file into a raw byte buffer and a line-start index.
/// Ensures complete records are extracted by rewinding the file pointer if a
/// record is cut.
FastqData parse_fastq_chunk(std::ifstream &file, size_t chunk_size,
                            bool stat_mode = false,
                            const std::string &log_stat_path = "");

/// Summarize the byte size of the extracted FASTQ field streams and line
/// lengths.
FastqFieldStats compute_field_stats(const FastqData &data);

/// Write a FASTQ file from its raw byte representation.
void write_fastq(const std::string &filepath, const FastqData &data);

} // namespace gpufastq
