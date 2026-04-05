#pragma once

#include "fastq_record.hpp"
#include <cstdint>
#include <string>

namespace gpufastq {

/// File extension for compressed FASTQ
constexpr const char *COMPRESSED_EXTENSION = ".cuvf";

/// Magic bytes: "GFQZ" in little-endian
constexpr uint32_t MAGIC = 0x5A514647;

/// File format version
constexpr uint32_t FORMAT_VERSION = 17;

/// Serialize compressed FASTQ data to a binary .cuvf file (single chunk)
void serialize(const std::string &filepath, const CompressedFastqData &data);

/// Deserialize compressed FASTQ data from a .cuvf file (single chunk)
CompressedFastqData deserialize(const std::string &filepath);

/// Serializes multiple chunks to an open file stream.
/// First, writes the MAGIC and FORMAT_VERSION headers once.
void serialize_header(std::ofstream &file);

/// Appends a single CompressedFastqData chunk to the stream.
void serialize_chunk(std::ofstream &file, const CompressedFastqData &data);

/// Deserializes the MAGIC and FORMAT_VERSION headers from the stream.
uint32_t deserialize_header(std::ifstream &file);

/// Reads a single CompressedFastqData chunk from the stream. Returns false if
/// EOF.
bool deserialize_chunk(std::ifstream &file, CompressedFastqData &data,
                       uint32_t format_version);

} // namespace gpufastq
