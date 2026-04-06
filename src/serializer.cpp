#include "gpufastq/serializer.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace gpufastq {

namespace {

template <typename T> void write_val(std::ofstream &f, const T &val) {
  f.write(reinterpret_cast<const char *>(&val), sizeof(T));
}

template <typename T> T read_val(std::ifstream &f) {
  T val;
  f.read(reinterpret_cast<char *>(&val), sizeof(T));
  if (!f && f.gcount() == 0) {
    throw std::runtime_error("Unexpected end of file");
  } else if (!f) {
    throw std::runtime_error("Unexpected partial end of file");
  }
  return val;
}

template <typename T> bool read_val_opt(std::ifstream &f, T &val) {
  f.read(reinterpret_cast<char *>(&val), sizeof(T));
  if (f.gcount() == 0)
    return false;
  if (!f)
    throw std::runtime_error("Unexpected partial end of file");
  return true;
}

void write_blob(std::ofstream &f, const std::vector<uint8_t> &data) {
  if (!data.empty()) {
    f.write(reinterpret_cast<const char *>(data.data()), data.size());
  }
}

void write_string(std::ofstream &f, const std::string &value) {
  write_val(f, static_cast<uint64_t>(value.size()));
  if (!value.empty()) {
    f.write(value.data(), static_cast<std::streamsize>(value.size()));
  }
}

void write_index(std::ofstream &f, const std::vector<uint64_t> &index) {
  for (uint64_t value : index) {
    write_val(f, value);
  }
}

std::vector<uint8_t> read_blob(std::ifstream &f, size_t size) {
  std::vector<uint8_t> data(size);
  if (size > 0) {
    f.read(reinterpret_cast<char *>(data.data()), size);
    if (!f) {
      throw std::runtime_error("Unexpected end of file reading blob");
    }
  }
  return data;
}

std::string read_string(std::ifstream &f) {
  const uint64_t size = read_val<uint64_t>(f);
  std::string value(static_cast<size_t>(size), '\0');
  if (size > 0) {
    f.read(value.data(), static_cast<std::streamsize>(size));
    if (!f) {
      throw std::runtime_error("Unexpected end of file reading string");
    }
  }
  return value;
}

std::vector<uint64_t> read_index(std::ifstream &f, size_t size) {
  std::vector<uint64_t> index(size);
  for (size_t i = 0; i < size; ++i) {
    index[i] = read_val<uint64_t>(f);
  }
  return index;
}

void validate_chunked_stream(const std::vector<uint8_t> &payload,
                             size_t original_size,
                             const std::vector<uint64_t> &chunk_sizes,
                             const char *name) {
  const bool has_payload = !payload.empty();
  const bool has_chunks = !chunk_sizes.empty();
  const bool has_original = original_size != 0;
  if (has_payload != has_chunks || has_payload != has_original) {
    throw std::runtime_error(std::string("Cannot serialize inconsistent ") +
                             name + " chunk metadata");
  }
}

void validate_identifier_column(const CompressedIdentifierColumn &column,
                                uint64_t num_records) {
  validate_chunked_stream(column.values.payload, column.values.original_size,
                          column.compressed_value_chunk_sizes,
                          "identifier-column-values");
  if (column.raw_text_size == 0 && num_records != 0) {
    throw std::runtime_error(
        "Cannot serialize identifier column without raw text size");
  }
  if (column.kind == IdentifierColumnKind::String) {
    if (column.encoding != IdentifierColumnEncoding::Plain) {
      throw std::runtime_error(
          "Cannot serialize string identifier column with non-plain encoding");
    }
    if (column.lengths.original_size != num_records * sizeof(uint32_t)) {
      throw std::runtime_error(
          "Cannot serialize string identifier column with invalid length size");
    }
    validate_chunked_stream(
        column.lengths.payload, column.lengths.original_size,
        column.compressed_length_chunk_sizes, "identifier-column-lengths");
    return;
  }

  if (column.kind != IdentifierColumnKind::Int32) {
    throw std::runtime_error("Cannot serialize unknown identifier column kind");
  }
  if (column.encoding != IdentifierColumnEncoding::Plain &&
      column.encoding != IdentifierColumnEncoding::Delta &&
      column.encoding != IdentifierColumnEncoding::DeltaVarint) {
    throw std::runtime_error(
        "Cannot serialize numeric identifier column with unknown encoding");
  }
  if ((column.encoding == IdentifierColumnEncoding::Plain ||
       column.encoding == IdentifierColumnEncoding::Delta) &&
      column.values.original_size != num_records * sizeof(int32_t)) {
    throw std::runtime_error("Cannot serialize fixed-width numeric identifier "
                             "column with invalid value size");
  }
  if (column.encoding == IdentifierColumnEncoding::DeltaVarint &&
      column.values.original_size == 0 && num_records != 0) {
    throw std::runtime_error("Cannot serialize delta-varint numeric identifier "
                             "column with empty values");
  }
  if (column.lengths.original_size != 0 || !column.lengths.payload.empty() ||
      !column.compressed_length_chunk_sizes.empty()) {
    throw std::runtime_error(
        "Cannot serialize numeric identifier column with length metadata");
  }
}

void validate_identifier_stream(const CompressedIdentifierData &data,
                                uint64_t num_records) {
  if (data.mode == IdentifierCompressionMode::Flat) {
    validate_chunked_stream(data.flat_data.payload,
                            data.flat_data.original_size,
                            data.compressed_flat_chunk_sizes, "identifier");
    if (!data.columns.empty() || data.layout.columnar ||
        !data.layout.separators.empty() || !data.layout.column_kinds.empty()) {
      throw std::runtime_error(
          "Cannot serialize flat identifiers with columnar metadata");
    }
    return;
  }

  if (data.mode != IdentifierCompressionMode::Columnar) {
    throw std::runtime_error("Cannot serialize unknown identifier mode");
  }
  if (data.layout.column_kinds.empty() || data.columns.empty() ||
      data.layout.column_kinds.size() != data.columns.size()) {
    throw std::runtime_error(
        "Cannot serialize columnar identifiers without matching columns");
  }
  if (data.layout.separators.size() + 1 != data.layout.column_kinds.size()) {
    throw std::runtime_error(
        "Cannot serialize columnar identifiers with inconsistent separators");
  }
  if (!data.flat_data.payload.empty() || data.flat_data.original_size != 0 ||
      !data.compressed_flat_chunk_sizes.empty()) {
    throw std::runtime_error(
        "Cannot serialize columnar identifiers with flat fallback payload");
  }

  for (size_t i = 0; i < data.columns.size(); ++i) {
    if (data.columns[i].kind != data.layout.column_kinds[i]) {
      throw std::runtime_error(
          "Cannot serialize identifier columns with mismatched kinds");
    }
    validate_identifier_column(data.columns[i], num_records);
  }
}

void validate_basecall_stream(const CompressedBasecallData &data) {
  if (data.original_size == 0) {
    const bool has_data =
        data.n_block_size != 0 || data.packed_bases.original_size != 0 ||
        !data.packed_bases.payload.empty() ||
        !data.compressed_packed_chunk_sizes.empty() ||
        !data.uncompressed_packed_chunk_sizes.empty() ||
        data.n_counts.original_size != 0 || !data.n_counts.payload.empty() ||
        data.n_positions.original_size != 0 ||
        !data.n_positions.payload.empty() ||
        !data.compressed_n_position_chunk_sizes.empty();
    if (has_data) {
      throw std::runtime_error(
          "Cannot serialize inconsistent empty basecall metadata");
    }
    return;
  }

  if (data.n_block_size == 0) {
    throw std::runtime_error(
        "Cannot serialize basecall metadata without an N-index block size");
  }
  if (data.packed_codec != BasecallPackedCodec::Zstd &&
      data.packed_codec != BasecallPackedCodec::Bsc) {
    throw std::runtime_error(
        "Cannot serialize basecall metadata with unknown packed codec");
  }
  if (data.pack_order != BasecallPackOrder::Tgca &&
      data.pack_order != BasecallPackOrder::Acgt) {
    throw std::runtime_error(
        "Cannot serialize basecall metadata with unknown pack order");
  }

  const uint64_t expected_block_count =
      (data.original_size + data.n_block_size - 1) / data.n_block_size;
  if (data.n_counts.original_size != expected_block_count * sizeof(uint16_t)) {
    throw std::runtime_error(
        "Cannot serialize basecall metadata with an unexpected N-count size");
  }

  const size_t expected_packed_size = (data.original_size + 3) / 4;
  if (data.packed_bases.original_size != expected_packed_size) {
    throw std::runtime_error(
        "Cannot serialize basecall metadata with an unexpected packed size");
  }

  validate_chunked_stream(
      data.packed_bases.payload, data.packed_bases.original_size,
      data.compressed_packed_chunk_sizes, "packed-basecall");
  if (data.packed_codec == BasecallPackedCodec::Bsc) {
    if (data.compressed_packed_chunk_sizes.size() !=
        data.uncompressed_packed_chunk_sizes.size()) {
      throw std::runtime_error(
          "Cannot serialize BSC packed-basecall metadata with mismatched chunk"
          " sizes");
    }
  } else if (!data.uncompressed_packed_chunk_sizes.empty()) {
    throw std::runtime_error(
        "Cannot serialize Zstd packed-basecall metadata with BSC chunk sizes");
  }
  if ((data.n_counts.original_size == 0) != data.n_counts.payload.empty()) {
    throw std::runtime_error(
        "Cannot serialize inconsistent compressed N-count metadata");
  }
  validate_chunked_stream(data.n_positions.payload,
                          data.n_positions.original_size,
                          data.compressed_n_position_chunk_sizes, "N-position");
}

void validate_plus_line_kinds(const CompressedFastqData &data) {
  if (data.plus_line_kinds.empty()) {
    return;
  }
  if (data.plus_line_kinds.size() != static_cast<size_t>(data.num_records)) {
    throw std::runtime_error(
        "Cannot serialize plus-line metadata with inconsistent record count");
  }
  for (uint8_t kind : data.plus_line_kinds) {
    if (kind != static_cast<uint8_t>(PlusLineKind::BarePlus) &&
        kind != static_cast<uint8_t>(PlusLineKind::CopyIdentifier)) {
      throw std::runtime_error("Cannot serialize unknown plus-line kind");
    }
  }
}

} // namespace

void serialize_header(std::ofstream &file) {
  write_val(file, MAGIC);
  write_val(file, FORMAT_VERSION);
}

void serialize_chunk(std::ofstream &file, const CompressedFastqData &data) {
  validate_identifier_stream(data.identifiers, data.num_records);
  validate_basecall_stream(data.basecalls);
  if (data.quality_codec != QualityCodec::Bsc &&
      data.quality_codec != QualityCodec::Zstd) {
    throw std::runtime_error("Decoded unknown quality codec");
  }
  validate_chunked_stream(data.quality_scores.payload,
                          data.quality_scores.original_size,
                          data.compressed_quality_chunk_sizes, "quality");
  if (data.quality_codec != QualityCodec::Bsc &&
      data.quality_codec != QualityCodec::Zstd) {
    throw std::runtime_error("Cannot serialize unknown quality codec");
  }
  if (data.quality_layout != QualityLayoutKind::FixedLength &&
      data.quality_layout != QualityLayoutKind::VariableLength) {
    throw std::runtime_error("Cannot serialize unknown quality layout");
  }
  if (data.quality_layout == QualityLayoutKind::FixedLength) {
    if (data.num_records == 0) {
      if (data.fixed_quality_length != 0) {
        throw std::runtime_error(
            "Cannot serialize empty FASTQ with fixed quality length");
      }
    } else if (data.quality_scores.original_size !=
               static_cast<uint64_t>(data.fixed_quality_length) *
                   data.num_records) {
      throw std::runtime_error("Cannot serialize fixed-length quality metadata "
                               "with inconsistent size");
    }
  } else if (data.fixed_quality_length != 0) {
    throw std::runtime_error(
        "Cannot serialize variable-length quality metadata with fixed length");
  }
  if ((!data.compressed_quality_chunk_sizes.empty()) !=
      (!data.uncompressed_quality_chunk_sizes.empty())) {
    throw std::runtime_error(
        "Cannot serialize inconsistent quality chunk size metadata");
  }
  if (data.compressed_quality_chunk_sizes.size() !=
      data.uncompressed_quality_chunk_sizes.size()) {
    throw std::runtime_error(
        "Cannot serialize mismatched quality chunk size vectors");
  }
  validate_chunked_stream(
      data.line_lengths.payload, data.line_lengths.original_size,
      data.compressed_line_length_chunk_sizes, "line-length");
  validate_plus_line_kinds(data);

  write_val(file, data.num_records);
  write_val(file, data.line_offset_count);

  write_val(file, static_cast<uint8_t>(data.identifiers.mode));
  write_val(file, static_cast<uint64_t>(data.identifiers.original_size));
  write_val(file,
            static_cast<uint64_t>(data.identifiers.layout.separators.size()));
  write_val(file,
            static_cast<uint64_t>(data.identifiers.layout.column_kinds.size()));
  write_val(file, static_cast<uint64_t>(data.identifiers.columns.size()));
  write_val(file,
            static_cast<uint64_t>(data.identifiers.flat_data.original_size));
  write_val(file,
            static_cast<uint64_t>(data.identifiers.flat_data.payload.size()));
  write_val(file, static_cast<uint64_t>(
                      data.identifiers.compressed_flat_chunk_sizes.size()));

  write_val(file, data.basecalls.original_size);
  write_val(file, data.basecalls.n_block_size);
  write_val(file, static_cast<uint8_t>(data.basecalls.packed_codec));
  write_val(file, static_cast<uint8_t>(data.basecalls.pack_order));
  write_val(file,
            static_cast<uint64_t>(data.basecalls.packed_bases.original_size));
  write_val(file,
            static_cast<uint64_t>(data.basecalls.packed_bases.payload.size()));
  write_val(file, static_cast<uint64_t>(
                      data.basecalls.compressed_packed_chunk_sizes.size()));
  write_val(file, static_cast<uint64_t>(
                      data.basecalls.uncompressed_packed_chunk_sizes.size()));
  write_val(file, static_cast<uint64_t>(data.basecalls.n_counts.original_size));
  write_val(file,
            static_cast<uint64_t>(data.basecalls.n_counts.payload.size()));
  write_val(file,
            static_cast<uint64_t>(data.basecalls.n_positions.original_size));
  write_val(file,
            static_cast<uint64_t>(data.basecalls.n_positions.payload.size()));
  write_val(file, static_cast<uint64_t>(
                      data.basecalls.compressed_n_position_chunk_sizes.size()));

  write_val(file, static_cast<uint8_t>(data.quality_codec));
  write_val(file, static_cast<uint8_t>(data.quality_layout));
  write_val(file, data.fixed_quality_length);
  write_val(file, static_cast<uint8_t>(data.quality_transposed ? 1 : 0));
  write_val(file, static_cast<uint64_t>(data.quality_scores.original_size));
  write_val(file, static_cast<uint64_t>(data.quality_scores.payload.size()));
  write_val(file,
            static_cast<uint64_t>(data.compressed_quality_chunk_sizes.size()));

  write_val(file, static_cast<uint64_t>(data.line_lengths.original_size));
  write_val(file, static_cast<uint64_t>(data.line_lengths.payload.size()));
  write_val(file, static_cast<uint64_t>(
                      data.compressed_line_length_chunk_sizes.size()));
  write_val(file, static_cast<uint64_t>(data.plus_line_kinds.size()));

  for (const auto &separator : data.identifiers.layout.separators) {
    write_string(file, separator);
  }
  for (IdentifierColumnKind kind : data.identifiers.layout.column_kinds) {
    write_val(file, static_cast<uint8_t>(kind));
  }
  write_index(file, data.identifiers.compressed_flat_chunk_sizes);
  for (const auto &column : data.identifiers.columns) {
    write_val(file, static_cast<uint8_t>(column.kind));
    write_val(file, static_cast<uint8_t>(column.encoding));
    write_val(file, static_cast<uint64_t>(column.raw_text_size));
    write_val(file, static_cast<uint64_t>(column.values.original_size));
    write_val(file, static_cast<uint64_t>(column.values.payload.size()));
    write_val(file, static_cast<uint64_t>(
                        column.compressed_value_chunk_sizes.size()));
    write_val(file, static_cast<uint64_t>(column.lengths.original_size));
    write_val(file, static_cast<uint64_t>(column.lengths.payload.size()));
    write_val(file, static_cast<uint64_t>(
                        column.compressed_length_chunk_sizes.size()));
    write_index(file, column.compressed_value_chunk_sizes);
    write_index(file, column.compressed_length_chunk_sizes);
  }
  write_index(file, data.basecalls.compressed_packed_chunk_sizes);
  write_index(file, data.basecalls.uncompressed_packed_chunk_sizes);
  write_index(file, data.basecalls.compressed_n_position_chunk_sizes);
  write_index(file, data.compressed_quality_chunk_sizes);
  write_index(file, data.uncompressed_quality_chunk_sizes);
  write_index(file, data.compressed_line_length_chunk_sizes);
  for (uint8_t kind : data.plus_line_kinds) {
    write_val(file, kind);
  }

  write_blob(file, data.identifiers.flat_data.payload);
  for (const auto &column : data.identifiers.columns) {
    write_blob(file, column.values.payload);
    write_blob(file, column.lengths.payload);
  }
  write_blob(file, data.basecalls.packed_bases.payload);
  write_blob(file, data.basecalls.n_counts.payload);
  write_blob(file, data.basecalls.n_positions.payload);
  write_blob(file, data.quality_scores.payload);
  write_blob(file, data.line_lengths.payload);
}

void serialize(const std::string &filepath, const CompressedFastqData &data) {
  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open output file: " + filepath);
  }
  serialize_header(file);
  serialize_chunk(file, data);
}

CompressedFastqData deserialize(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open input file: " + filepath);
  }
  const uint32_t format_version = deserialize_header(file);
  CompressedFastqData data;
  if (!deserialize_chunk(file, data, format_version)) {
    throw std::runtime_error("Expected at least one chunk in file");
  }
  return data;
}

uint32_t deserialize_header(std::ifstream &file) {
  const uint32_t magic = read_val<uint32_t>(file);
  if (magic != MAGIC) {
    throw std::runtime_error("Invalid file format: bad magic number");
  }

  const uint32_t version = read_val<uint32_t>(file);
  if (version != 16 && version != 17 && version != FORMAT_VERSION) {
    throw std::runtime_error("Unsupported format version: " +
                             std::to_string(version));
  }
  return version;
}

bool deserialize_chunk(std::ifstream &file, CompressedFastqData &data,
                       uint32_t format_version) {
  if (!read_val_opt(file, data.num_records)) {
    return false;
  }
  data.line_offset_count = read_val<uint64_t>(file);

  data.identifiers.mode =
      static_cast<IdentifierCompressionMode>(read_val<uint8_t>(file));
  data.identifiers.original_size = read_val<uint64_t>(file);
  const uint64_t identifier_separator_count = read_val<uint64_t>(file);
  const uint64_t identifier_kind_count = read_val<uint64_t>(file);
  const uint64_t identifier_column_count = read_val<uint64_t>(file);
  data.identifiers.flat_data.original_size = read_val<uint64_t>(file);
  const uint64_t comp_id_size = read_val<uint64_t>(file);
  const uint64_t comp_id_chunks = read_val<uint64_t>(file);

  data.basecalls.original_size = read_val<uint64_t>(file);
  data.basecalls.n_block_size = read_val<uint32_t>(file);
  data.basecalls.packed_codec =
      static_cast<BasecallPackedCodec>(read_val<uint8_t>(file));
  if (format_version >= 17) {
    data.basecalls.pack_order =
        static_cast<BasecallPackOrder>(read_val<uint8_t>(file));
  } else {
    data.basecalls.pack_order = BasecallPackOrder::Tgca;
  }
  data.basecalls.packed_bases.original_size = read_val<uint64_t>(file);
  const uint64_t comp_packed_size = read_val<uint64_t>(file);
  const uint64_t comp_packed_chunks = read_val<uint64_t>(file);
  const uint64_t uncomp_packed_chunks = read_val<uint64_t>(file);
  data.basecalls.n_counts.original_size = read_val<uint64_t>(file);
  const uint64_t comp_n_count_size = read_val<uint64_t>(file);
  data.basecalls.n_positions.original_size = read_val<uint64_t>(file);
  const uint64_t comp_n_pos_size = read_val<uint64_t>(file);
  const uint64_t comp_n_pos_chunks = read_val<uint64_t>(file);

  data.quality_codec = static_cast<QualityCodec>(read_val<uint8_t>(file));
  data.quality_layout = static_cast<QualityLayoutKind>(read_val<uint8_t>(file));
  data.fixed_quality_length = read_val<uint32_t>(file);
  data.quality_transposed = read_val<uint8_t>(file) != 0;
  data.quality_scores.original_size = read_val<uint64_t>(file);
  const uint64_t comp_qual_size = read_val<uint64_t>(file);
  const uint64_t comp_qual_chunks = read_val<uint64_t>(file);

  data.line_lengths.original_size = read_val<uint64_t>(file);
  const uint64_t comp_index_size = read_val<uint64_t>(file);
  const uint64_t comp_index_chunks = read_val<uint64_t>(file);
  uint64_t plus_line_kind_count = 0;
  if (format_version >= 18) {
    plus_line_kind_count = read_val<uint64_t>(file);
  }

  data.identifiers.layout.separators.reserve(
      static_cast<size_t>(identifier_separator_count));
  for (uint64_t i = 0; i < identifier_separator_count; ++i) {
    data.identifiers.layout.separators.push_back(read_string(file));
  }
  data.identifiers.layout.column_kinds.reserve(
      static_cast<size_t>(identifier_kind_count));
  for (uint64_t i = 0; i < identifier_kind_count; ++i) {
    data.identifiers.layout.column_kinds.push_back(
        static_cast<IdentifierColumnKind>(read_val<uint8_t>(file)));
  }
  data.identifiers.layout.columnar =
      data.identifiers.mode == IdentifierCompressionMode::Columnar;
  data.identifiers.compressed_flat_chunk_sizes =
      read_index(file, comp_id_chunks);
  data.identifiers.columns.resize(static_cast<size_t>(identifier_column_count));
  std::vector<uint64_t> identifier_value_payload_sizes(
      static_cast<size_t>(identifier_column_count));
  std::vector<uint64_t> identifier_length_payload_sizes(
      static_cast<size_t>(identifier_column_count));
  for (auto &column : data.identifiers.columns) {
    column.kind = static_cast<IdentifierColumnKind>(read_val<uint8_t>(file));
    column.encoding =
        static_cast<IdentifierColumnEncoding>(read_val<uint8_t>(file));
    column.raw_text_size = read_val<uint64_t>(file);
    column.values.original_size = read_val<uint64_t>(file);
    const uint64_t value_payload_size = read_val<uint64_t>(file);
    const uint64_t value_chunk_count = read_val<uint64_t>(file);
    column.lengths.original_size = read_val<uint64_t>(file);
    const uint64_t length_payload_size = read_val<uint64_t>(file);
    const uint64_t length_chunk_count = read_val<uint64_t>(file);
    column.compressed_value_chunk_sizes = read_index(file, value_chunk_count);
    column.compressed_length_chunk_sizes = read_index(file, length_chunk_count);
    identifier_value_payload_sizes[&column - data.identifiers.columns.data()] =
        value_payload_size;
    identifier_length_payload_sizes[&column - data.identifiers.columns.data()] =
        length_payload_size;
  }
  data.basecalls.compressed_packed_chunk_sizes =
      read_index(file, comp_packed_chunks);
  data.basecalls.uncompressed_packed_chunk_sizes =
      read_index(file, uncomp_packed_chunks);
  data.basecalls.compressed_n_position_chunk_sizes =
      read_index(file, comp_n_pos_chunks);
  data.compressed_quality_chunk_sizes = read_index(file, comp_qual_chunks);
  data.uncompressed_quality_chunk_sizes = read_index(file, comp_qual_chunks);
  data.compressed_line_length_chunk_sizes = read_index(file, comp_index_chunks);
  data.plus_line_kinds.clear();
  if (format_version >= 18) {
    data.plus_line_kinds.reserve(static_cast<size_t>(plus_line_kind_count));
    for (uint64_t i = 0; i < plus_line_kind_count; ++i) {
      data.plus_line_kinds.push_back(read_val<uint8_t>(file));
    }
  }
  if (data.plus_line_kinds.empty() && data.num_records != 0) {
    data.plus_line_kinds.resize(
        static_cast<size_t>(data.num_records),
        static_cast<uint8_t>(PlusLineKind::BarePlus));
  }

  data.identifiers.flat_data.payload = read_blob(file, comp_id_size);
  for (size_t i = 0; i < data.identifiers.columns.size(); ++i) {
    data.identifiers.columns[i].values.payload =
        read_blob(file, identifier_value_payload_sizes[i]);
    data.identifiers.columns[i].lengths.payload =
        read_blob(file, identifier_length_payload_sizes[i]);
  }
  data.basecalls.packed_bases.payload = read_blob(file, comp_packed_size);
  data.basecalls.n_counts.payload = read_blob(file, comp_n_count_size);
  data.basecalls.n_positions.payload = read_blob(file, comp_n_pos_size);
  data.quality_scores.payload = read_blob(file, comp_qual_size);
  data.line_lengths.payload = read_blob(file, comp_index_size);

  validate_identifier_stream(data.identifiers, data.num_records);
  validate_basecall_stream(data.basecalls);
  validate_chunked_stream(data.quality_scores.payload,
                          data.quality_scores.original_size,
                          data.compressed_quality_chunk_sizes, "quality");
  validate_plus_line_kinds(data);
  return true;
}

} // namespace gpufastq
