#include "gpufastq/codec_bsc.hpp"
#include "gpufastq/codec_gpu.cuh"
#include "gpufastq/codec_gpu_nvcomp.cuh"
#include "gpufastq/fastq_parser.hpp"
#include "gpufastq/gpu_compressor.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>

namespace gpufastq {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +      \
                               ":" + std::to_string(__LINE__) + ": " +         \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

namespace {

constexpr size_t MAX_FIELD_SLICE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BASECALL_N_BLOCK_SIZE = 8192;

struct ChunkedCompressedBuffer {
  std::vector<uint8_t> data;
  std::vector<uint64_t> chunk_sizes;
};

struct TokenizedIdentifier {
  std::vector<std::string> tokens;
  std::vector<std::string> separators;
};

struct IdentifierColumnBuffers {
  IdentifierColumnKind kind = IdentifierColumnKind::String;
  uint64_t raw_text_size = 0;
  std::vector<uint8_t> string_values;
  std::vector<uint32_t> string_lengths;
  std::vector<int32_t> int_values;
};

struct DeviceIdentifierSchema {
  uint8_t *separator_bytes = nullptr;
  uint64_t *separator_offsets = nullptr;
  size_t separator_count = 0;
  size_t column_count = 0;
};

std::vector<uint8_t>
gpu_decompress_chunked(const std::vector<uint8_t> &compressed,
                       const std::vector<uint64_t> &chunk_sizes,
                       uint64_t expected_size);

ChunkedCompressedBuffer gpu_compress_device_chunked(const uint8_t *d_input,
                                                    size_t input_size,
                                                    size_t field_slice_size,
                                                    size_t nvcomp_chunk_size,
                                                    cudaStream_t stream);

BscChunkedBuffer bsc_compress_device_chunked(const uint8_t *d_input,
                                             size_t input_size,
                                             size_t chunk_size,
                                             const BscConfig &config,
                                             cudaStream_t stream);

struct DeviceFieldBuffers {
  uint8_t *identifiers = nullptr;
  uint8_t *basecalls = nullptr;
  uint8_t *quality_scores = nullptr;
};

struct EncodedBasecallBuffers {
  uint8_t *packed_bases = nullptr;
  uint32_t *n_counts = nullptr;
  uint64_t *n_offsets = nullptr;
  uint16_t *n_positions = nullptr;
};

struct UInt32ToUInt64 {
  __host__ __device__ uint64_t operator()(uint32_t value) const {
    return static_cast<uint64_t>(value);
  }
};

std::vector<uint8_t> transpose_fixed_length_quality_scores(
    const std::vector<uint8_t> &quality_scores, uint64_t num_records,
    uint32_t quality_length) {
  if (quality_scores.empty()) {
    return {};
  }
  if (num_records == 0 ||
      quality_scores.size() !=
          num_records * static_cast<uint64_t>(quality_length)) {
    throw std::runtime_error(
        "Fixed-length quality transpose received inconsistent input size");
  }

  std::vector<uint8_t> transposed(quality_scores.size());
  for (uint32_t column = 0; column < quality_length; ++column) {
    const uint64_t column_offset = static_cast<uint64_t>(column) * num_records;
    for (uint64_t record = 0; record < num_records; ++record) {
      transposed[column_offset + record] =
          quality_scores[record * static_cast<uint64_t>(quality_length) +
                         column];
    }
  }
  return transposed;
}

std::vector<uint8_t> inverse_transpose_fixed_length_quality_scores(
    const std::vector<uint8_t> &transposed_quality_scores, uint64_t num_records,
    uint32_t quality_length) {
  if (transposed_quality_scores.empty()) {
    return {};
  }
  if (num_records == 0 ||
      transposed_quality_scores.size() !=
          num_records * static_cast<uint64_t>(quality_length)) {
    throw std::runtime_error("Fixed-length quality inverse transpose received "
                             "inconsistent input size");
  }

  std::vector<uint8_t> quality_scores(transposed_quality_scores.size());
  for (uint32_t column = 0; column < quality_length; ++column) {
    const uint64_t column_offset = static_cast<uint64_t>(column) * num_records;
    for (uint64_t record = 0; record < num_records; ++record) {
      quality_scores[record * static_cast<uint64_t>(quality_length) + column] =
          transposed_quality_scores[column_offset + record];
    }
  }
  return quality_scores;
}

uint64_t sum_sizes(const std::vector<uint64_t> &sizes) {
  return std::accumulate(sizes.begin(), sizes.end(), uint64_t{0});
}

const char *identifier_column_kind_name(IdentifierColumnKind kind) {
  switch (kind) {
  case IdentifierColumnKind::String:
    return "string";
  case IdentifierColumnKind::Int32:
    return "int32";
  }
  return "unknown";
}

const char *identifier_column_encoding_name(IdentifierColumnEncoding encoding) {
  switch (encoding) {
  case IdentifierColumnEncoding::Plain:
    return "plain";
  case IdentifierColumnEncoding::Delta:
    return "delta";
  case IdentifierColumnEncoding::DeltaVarint:
    return "delta-varint";
  }
  return "unknown";
}

double compression_ratio_percent(uint64_t compressed_size, uint64_t raw_size) {
  if (raw_size == 0) {
    return 0.0;
  }
  return 100.0 * static_cast<double>(compressed_size) /
         static_cast<double>(raw_size);
}

uint64_t line_content_length(const std::vector<uint64_t> &line_offsets,
                             uint64_t line_idx) {
  return line_offsets[line_idx + 1] - line_offsets[line_idx] - 1;
}

bool is_identifier_separator(uint8_t ch) {
  return ch == ':' || ch == '/' || ch == '-' || ch == '.' ||
         std::isspace(static_cast<unsigned char>(ch)) != 0;
}

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
    return value >= std::numeric_limits<int32_t>::min() &&
           value <= std::numeric_limits<int32_t>::max() &&
           std::to_string(value) == token;
  } catch (...) {
    return false;
  }
}

int32_t parse_int32_token(const std::string &token) {
  const long long value = std::stoll(token);
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max() ||
      std::to_string(value) != token) {
    throw std::runtime_error("Identifier token is not a canonical int32: " +
                             token);
  }
  return static_cast<int32_t>(value);
}

std::vector<uint8_t> int32_vector_to_bytes(const std::vector<int32_t> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(int32_t));
  if (!values.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

std::vector<int32_t> delta_encode_int32(const std::vector<int32_t> &values) {
  std::vector<int32_t> deltas(values.size());
  if (values.empty()) {
    return deltas;
  }

  deltas[0] = values[0];
  for (size_t i = 1; i < values.size(); ++i) {
    const int64_t delta =
        static_cast<int64_t>(values[i]) - static_cast<int64_t>(values[i - 1]);
    if (delta < std::numeric_limits<int32_t>::min() ||
        delta > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error("Identifier delta exceeds int32 range");
    }
    deltas[i] = static_cast<int32_t>(delta);
  }
  return deltas;
}

std::vector<int32_t> delta_decode_int32(const int32_t *values, size_t count) {
  std::vector<int32_t> decoded(count);
  if (count == 0) {
    return decoded;
  }

  int64_t running = values[0];
  if (running < std::numeric_limits<int32_t>::min() ||
      running > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("Identifier delta decode overflow");
  }
  decoded[0] = static_cast<int32_t>(running);
  for (size_t i = 1; i < count; ++i) {
    running += values[i];
    if (running < std::numeric_limits<int32_t>::min() ||
        running > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error("Identifier delta decode overflow");
    }
    decoded[i] = static_cast<int32_t>(running);
  }
  return decoded;
}

__host__ __device__ uint32_t zigzag_encode_int32(int32_t value) {
  return (static_cast<uint32_t>(value) << 1) ^
         static_cast<uint32_t>(value >> 31);
}

__host__ __device__ int32_t zigzag_decode_int32(uint32_t value) {
  return static_cast<int32_t>(
      (value >> 1) ^ static_cast<uint32_t>(-static_cast<int32_t>(value & 1)));
}

std::vector<uint8_t> encode_varint_u32(const std::vector<int32_t> &values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * 2);
  for (int32_t value : values) {
    uint32_t encoded = zigzag_encode_int32(value);
    while (encoded >= 0x80u) {
      bytes.push_back(static_cast<uint8_t>(encoded) | 0x80u);
      encoded >>= 7;
    }
    bytes.push_back(static_cast<uint8_t>(encoded));
  }
  return bytes;
}

std::vector<int32_t> decode_varint_u32(const std::vector<uint8_t> &bytes,
                                       size_t expected_count) {
  std::vector<int32_t> values;
  values.reserve(expected_count);

  uint32_t current = 0;
  uint32_t shift = 0;
  for (uint8_t byte : bytes) {
    current |= static_cast<uint32_t>(byte & 0x7fu) << shift;
    if ((byte & 0x80u) == 0) {
      values.push_back(zigzag_decode_int32(current));
      current = 0;
      shift = 0;
      continue;
    }
    shift += 7;
    if (shift >= 35) {
      throw std::runtime_error("Identifier varint payload is malformed");
    }
  }

  if (shift != 0) {
    throw std::runtime_error("Identifier varint payload ended mid-value");
  }
  if (values.size() != expected_count) {
    throw std::runtime_error(
        "Identifier varint payload decoded an unexpected count");
  }
  return values;
}

template <typename T> void cuda_free_if_set(T *ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

void free_device_identifier_schema(DeviceIdentifierSchema *schema) {
  if (schema == nullptr) {
    return;
  }
  cuda_free_if_set(schema->separator_bytes);
  cuda_free_if_set(schema->separator_offsets);
  schema->separator_count = 0;
  schema->column_count = 0;
}

DeviceIdentifierSchema
copy_identifier_schema_to_device(const IdentifierLayout &layout,
                                 cudaStream_t stream) {
  DeviceIdentifierSchema schema;
  schema.separator_count = layout.separators.size();
  schema.column_count = layout.column_kinds.size();

  std::vector<uint64_t> separator_offsets(schema.separator_count + 1, 0);
  std::vector<uint8_t> separator_bytes;
  for (size_t i = 0; i < layout.separators.size(); ++i) {
    separator_offsets[i] = separator_bytes.size();
    separator_bytes.insert(separator_bytes.end(), layout.separators[i].begin(),
                           layout.separators[i].end());
  }
  separator_offsets[schema.separator_count] = separator_bytes.size();

  if (!separator_bytes.empty()) {
    CUDA_CHECK(cudaMalloc(&schema.separator_bytes, separator_bytes.size()));
    CUDA_CHECK(cudaMemcpyAsync(schema.separator_bytes, separator_bytes.data(),
                               separator_bytes.size(), cudaMemcpyHostToDevice,
                               stream));
  }

  CUDA_CHECK(cudaMalloc(&schema.separator_offsets,
                        separator_offsets.size() * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpyAsync(schema.separator_offsets, separator_offsets.data(),
                             separator_offsets.size() * sizeof(uint64_t),
                             cudaMemcpyHostToDevice, stream));

  return schema;
}

__device__ inline bool device_is_identifier_separator(uint8_t ch) {
  return ch == ':' || ch == '/' || ch == '-' || ch == '.' || ch == ' ' ||
         ch == '\t' || ch == '\r' || ch == '\f' || ch == '\v';
}

__device__ bool advance_identifier_separator(const uint8_t *raw_bytes,
                                             uint64_t pos, uint64_t line_end,
                                             const uint8_t *separator_bytes,
                                             const uint64_t *separator_offsets,
                                             uint32_t separator_index,
                                             uint64_t *next_pos) {
  const uint64_t start = separator_offsets[separator_index];
  const uint64_t end = separator_offsets[separator_index + 1];
  const uint64_t length = end - start;
  if (pos + length > line_end) {
    return false;
  }
  for (uint64_t i = 0; i < length; ++i) {
    if (raw_bytes[pos + i] != separator_bytes[start + i]) {
      return false;
    }
  }
  *next_pos = pos + length;
  return true;
}

__device__ bool
locate_identifier_token(const uint8_t *raw_bytes, const uint64_t *line_offsets,
                        uint64_t record, uint32_t target_column,
                        uint32_t column_count, const uint8_t *separator_bytes,
                        const uint64_t *separator_offsets,
                        uint64_t *token_start, uint32_t *token_length) {
  const uint64_t id_line = 4 * record;
  uint64_t pos = line_offsets[id_line] + 1;
  const uint64_t line_end = line_offsets[id_line + 1] - 1;
  if (pos >= line_end) {
    return false;
  }

  for (uint32_t column = 0; column <= target_column; ++column) {
    const uint64_t current_start = pos;
    while (pos < line_end && !device_is_identifier_separator(raw_bytes[pos])) {
      ++pos;
    }
    if (pos == current_start) {
      return false;
    }

    if (column == target_column) {
      *token_start = current_start;
      *token_length = static_cast<uint32_t>(pos - current_start);
      if (column + 1 == column_count) {
        return pos == line_end;
      }
      uint64_t next_pos = 0;
      return advance_identifier_separator(raw_bytes, pos, line_end,
                                          separator_bytes, separator_offsets,
                                          column, &next_pos);
    }

    if (column >= column_count - 1) {
      return false;
    }
    uint64_t next_pos = 0;
    if (!advance_identifier_separator(raw_bytes, pos, line_end, separator_bytes,
                                      separator_offsets, column, &next_pos)) {
      return false;
    }
    pos = next_pos;
  }

  return false;
}

__device__ bool parse_identifier_int32_device(const uint8_t *raw_bytes,
                                              uint64_t token_start,
                                              uint32_t token_length,
                                              int32_t *value_out) {
  constexpr int64_t kInt32Min = -2147483648ll;
  constexpr int64_t kInt32Max = 2147483647ll;
  if (token_length == 0) {
    return false;
  }

  uint32_t index = 0;
  bool negative = false;
  if (raw_bytes[token_start] == '+') {
    return false;
  }
  if (raw_bytes[token_start] == '-') {
    negative = true;
    index = 1;
    if (token_length == 1) {
      return false;
    }
  }

  const uint8_t first_digit = raw_bytes[token_start + index];
  if (first_digit < '0' || first_digit > '9') {
    return false;
  }
  const uint32_t digit_count = token_length - index;
  if (digit_count > 1 && first_digit == '0') {
    return false;
  }

  const int64_t limit = negative ? (kInt32Max + 1) : kInt32Max;
  int64_t value = 0;
  for (; index < token_length; ++index) {
    const uint8_t ch = raw_bytes[token_start + index];
    if (ch < '0' || ch > '9') {
      return false;
    }
    value = value * 10 + static_cast<int64_t>(ch - '0');
    if (value > limit) {
      return false;
    }
  }

  if (negative) {
    if (value == 0) {
      return false;
    }
    value = -value;
  }

  if (value < kInt32Min || value > kInt32Max) {
    return false;
  }

  *value_out = static_cast<int32_t>(value);
  return true;
}

__global__ void extract_identifier_string_lengths_kernel(
    const uint8_t *raw_bytes, const uint64_t *line_offsets,
    uint64_t num_records, uint32_t target_column, uint32_t column_count,
    const uint8_t *separator_bytes, const uint64_t *separator_offsets,
    uint32_t *lengths, unsigned long long *invalid_record) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_records) {
    return;
  }

  uint64_t token_start = 0;
  uint32_t token_length = 0;
  if (!locate_identifier_token(raw_bytes, line_offsets, idx, target_column,
                               column_count, separator_bytes, separator_offsets,
                               &token_start, &token_length)) {
    atomicMin(invalid_record, static_cast<unsigned long long>(idx));
    return;
  }
  lengths[idx] = token_length;
}

__global__ void scatter_identifier_string_values_kernel(
    const uint8_t *raw_bytes, const uint64_t *line_offsets,
    uint64_t num_records, uint32_t target_column, uint32_t column_count,
    const uint8_t *separator_bytes, const uint64_t *separator_offsets,
    const uint64_t *value_offsets, const uint32_t *lengths, uint8_t *values,
    unsigned long long *invalid_record) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_records) {
    return;
  }

  uint64_t token_start = 0;
  uint32_t token_length = 0;
  if (!locate_identifier_token(raw_bytes, line_offsets, idx, target_column,
                               column_count, separator_bytes, separator_offsets,
                               &token_start, &token_length) ||
      token_length != lengths[idx]) {
    atomicMin(invalid_record, static_cast<unsigned long long>(idx));
    return;
  }

  const uint64_t dst = value_offsets[idx];
  for (uint32_t i = 0; i < token_length; ++i) {
    values[dst + i] = raw_bytes[token_start + i];
  }
}

__global__ void extract_identifier_int32_values_kernel(
    const uint8_t *raw_bytes, const uint64_t *line_offsets,
    uint64_t num_records, uint32_t target_column, uint32_t column_count,
    const uint8_t *separator_bytes, const uint64_t *separator_offsets,
    uint32_t *token_lengths, int32_t *values,
    unsigned long long *invalid_record) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_records) {
    return;
  }

  uint64_t token_start = 0;
  uint32_t token_length = 0;
  if (!locate_identifier_token(raw_bytes, line_offsets, idx, target_column,
                               column_count, separator_bytes, separator_offsets,
                               &token_start, &token_length)) {
    atomicMin(invalid_record, static_cast<unsigned long long>(idx));
    return;
  }

  int32_t parsed_value = 0;
  if (!parse_identifier_int32_device(raw_bytes, token_start, token_length,
                                     &parsed_value)) {
    atomicMin(invalid_record, static_cast<unsigned long long>(idx));
    return;
  }

  token_lengths[idx] = token_length;
  values[idx] = parsed_value;
}

__global__ void delta_encode_int32_kernel(const int32_t *values,
                                          int32_t *deltas, uint64_t count) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  if (idx == 0) {
    deltas[idx] = values[idx];
    return;
  }
  deltas[idx] = values[idx] - values[idx - 1];
}

__device__ inline uint32_t identifier_varint_size(int32_t value) {
  uint32_t encoded = zigzag_encode_int32(value);
  uint32_t size = 1;
  while (encoded >= 0x80u) {
    encoded >>= 7;
    ++size;
  }
  return size;
}

__global__ void compute_identifier_varint_sizes_kernel(const int32_t *values,
                                                       uint64_t count,
                                                       uint32_t *sizes) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  sizes[idx] = identifier_varint_size(values[idx]);
}

__global__ void scatter_identifier_varints_kernel(const int32_t *values,
                                                  const uint64_t *offsets,
                                                  uint64_t count,
                                                  uint8_t *output) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  uint64_t dst = offsets[idx];
  uint32_t encoded = zigzag_encode_int32(values[idx]);
  while (encoded >= 0x80u) {
    output[dst++] = static_cast<uint8_t>(encoded) | 0x80u;
    encoded >>= 7;
  }
  output[dst] = static_cast<uint8_t>(encoded);
}

ChunkedCompressedBuffer host_compress_chunked(const std::vector<uint8_t> &input,
                                              size_t field_slice_size,
                                              size_t nvcomp_chunk_size) {
  ChunkedCompressedBuffer result;
  if (input.empty()) {
    return result;
  }
  if (field_slice_size == 0 || field_slice_size > MAX_FIELD_SLICE_SIZE) {
    throw std::runtime_error(
        "Requested field slice size is out of supported range");
  }

  for (size_t offset = 0; offset < input.size(); offset += field_slice_size) {
    const size_t slice_size = std::min(field_slice_size, input.size() - offset);
    std::vector<uint8_t> slice(
        input.begin() + static_cast<std::ptrdiff_t>(offset),
        input.begin() + static_cast<std::ptrdiff_t>(offset + slice_size));
    auto compressed = nvcomp_zstd_compress(slice, nvcomp_chunk_size);
    result.chunk_sizes.push_back(compressed.payload.size());
    result.data.insert(result.data.end(), compressed.payload.begin(),
                       compressed.payload.end());
  }

  return result;
}

std::vector<uint8_t> extract_flat_identifiers_host(const FastqData &data) {
  const FastqFieldStats stats = compute_field_stats(data);
  std::vector<uint8_t> identifiers(static_cast<size_t>(stats.identifiers_size));

  uint64_t offset = 0;
  for (uint64_t record = 0; record < data.num_records; ++record) {
    const uint64_t id_line = 4 * record;
    const uint64_t id_len = line_content_length(data.line_offsets, id_line) - 1;
    const uint64_t id_start = data.line_offsets[id_line] + 1;
    std::memcpy(identifiers.data() + offset, data.raw_bytes.data() + id_start,
                static_cast<size_t>(id_len));
    offset += id_len;
  }

  return identifiers;
}

CompressedIdentifierData compress_identifiers_flat(const FastqData &data,
                                                   size_t field_slice_size,
                                                   size_t nvcomp_chunk_size) {
  CompressedIdentifierData result;
  result.mode = IdentifierCompressionMode::Flat;
  result.original_size = compute_field_stats(data).identifiers_size;
  result.flat_data.original_size = result.original_size;
  auto identifiers = extract_flat_identifiers_host(data);
  auto chunks =
      host_compress_chunked(identifiers, field_slice_size, nvcomp_chunk_size);
  result.flat_data.payload = std::move(chunks.data);
  result.compressed_flat_chunk_sizes = std::move(chunks.chunk_sizes);
  return result;
}

CompressedIdentifierData
compress_identifiers_columnar(const FastqData &data, const uint8_t *d_raw_bytes,
                              const uint64_t *d_line_offsets,
                              size_t field_slice_size, size_t nvcomp_chunk_size,
                              cudaStream_t stream) {
  if (!data.identifier_layout.columnar ||
      data.identifier_layout.column_kinds.empty()) {
    return compress_identifiers_flat(data, field_slice_size, nvcomp_chunk_size);
  }

  CompressedIdentifierData result;
  result.mode = IdentifierCompressionMode::Columnar;
  result.original_size = compute_field_stats(data).identifiers_size;
  result.layout = data.identifier_layout;
  result.layout.columnar = true;
  result.columns.resize(data.identifier_layout.column_kinds.size());
  if (data.num_records == 0) {
    for (size_t i = 0; i < result.columns.size(); ++i) {
      result.columns[i].kind = data.identifier_layout.column_kinds[i];
    }
    return result;
  }

  DeviceIdentifierSchema schema;
  uint32_t *d_lengths = nullptr;
  uint64_t *d_offsets = nullptr;
  int32_t *d_int_values = nullptr;
  int32_t *d_delta_values = nullptr;
  uint32_t *d_varint_sizes = nullptr;
  uint64_t *d_varint_offsets = nullptr;
  uint8_t *d_string_values = nullptr;
  uint8_t *d_varint_bytes = nullptr;
  unsigned long long *d_invalid_record = nullptr;
  bool fallback_to_flat = false;
  const uint32_t block_size = 256;
  const uint32_t grid_size =
      static_cast<uint32_t>((data.num_records + block_size - 1) / block_size);

  try {
    schema = copy_identifier_schema_to_device(data.identifier_layout, stream);
    if (data.num_records > 0) {
      CUDA_CHECK(cudaMalloc(&d_lengths, data.num_records * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_offsets, data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&d_int_values, data.num_records * sizeof(int32_t)));
      CUDA_CHECK(
          cudaMalloc(&d_delta_values, data.num_records * sizeof(int32_t)));
      CUDA_CHECK(
          cudaMalloc(&d_varint_sizes, data.num_records * sizeof(uint32_t)));
      CUDA_CHECK(
          cudaMalloc(&d_varint_offsets, data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&d_invalid_record, sizeof(unsigned long long)));
    }

    for (size_t i = 0; i < result.columns.size(); ++i) {
      auto &out = result.columns[i];
      out.kind = data.identifier_layout.column_kinds[i];
      out.encoding = IdentifierColumnEncoding::Plain;
      d_string_values = nullptr;
      d_varint_bytes = nullptr;

      unsigned long long invalid_record =
          std::numeric_limits<unsigned long long>::max();
      if (data.num_records > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_invalid_record, &invalid_record,
                                   sizeof(unsigned long long),
                                   cudaMemcpyHostToDevice, stream));
      }

      if (out.kind == IdentifierColumnKind::String) {
        extract_identifier_string_lengths_kernel<<<grid_size, block_size, 0,
                                                   stream>>>(
            d_raw_bytes, d_line_offsets, data.num_records,
            static_cast<uint32_t>(i),
            static_cast<uint32_t>(result.columns.size()),
            schema.separator_bytes, schema.separator_offsets, d_lengths,
            d_invalid_record);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(&invalid_record, d_invalid_record,
                              sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        if (invalid_record != std::numeric_limits<unsigned long long>::max()) {
          fallback_to_flat = true;
          break;
        }

        auto length_begin = thrust::make_transform_iterator(
            thrust::device_pointer_cast(d_lengths), UInt32ToUInt64{});
        out.raw_text_size =
            thrust::reduce(thrust::cuda::par.on(stream), length_begin,
                           length_begin + data.num_records, uint64_t{0},
                           thrust::plus<uint64_t>());

        if (out.raw_text_size > 0) {
          thrust::exclusive_scan(thrust::cuda::par.on(stream), length_begin,
                                 length_begin + data.num_records,
                                 thrust::device_pointer_cast(d_offsets));

          CUDA_CHECK(cudaMalloc(&d_string_values, out.raw_text_size));
          scatter_identifier_string_values_kernel<<<grid_size, block_size, 0,
                                                    stream>>>(
              d_raw_bytes, d_line_offsets, data.num_records,
              static_cast<uint32_t>(i),
              static_cast<uint32_t>(result.columns.size()),
              schema.separator_bytes, schema.separator_offsets, d_offsets,
              d_lengths, d_string_values, d_invalid_record);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaStreamSynchronize(stream));
          CUDA_CHECK(cudaMemcpy(&invalid_record, d_invalid_record,
                                sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost));
          if (invalid_record !=
              std::numeric_limits<unsigned long long>::max()) {
            fallback_to_flat = true;
            break;
          }
        }

        out.values.original_size = out.raw_text_size;
        auto value_chunks = gpu_compress_device_chunked(
            d_string_values, out.raw_text_size, field_slice_size,
            nvcomp_chunk_size, stream);
        out.values.payload = std::move(value_chunks.data);
        out.compressed_value_chunk_sizes = std::move(value_chunks.chunk_sizes);
        out.lengths.original_size = data.num_records * sizeof(uint32_t);
        auto length_chunks = gpu_compress_device_chunked(
            reinterpret_cast<const uint8_t *>(d_lengths),
            out.lengths.original_size, field_slice_size, nvcomp_chunk_size,
            stream);
        out.lengths.payload = std::move(length_chunks.data);
        out.compressed_length_chunk_sizes =
            std::move(length_chunks.chunk_sizes);
      } else {
        extract_identifier_int32_values_kernel<<<grid_size, block_size, 0,
                                                 stream>>>(
            d_raw_bytes, d_line_offsets, data.num_records,
            static_cast<uint32_t>(i),
            static_cast<uint32_t>(result.columns.size()),
            schema.separator_bytes, schema.separator_offsets, d_lengths,
            d_int_values, d_invalid_record);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(&invalid_record, d_invalid_record,
                              sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        if (invalid_record != std::numeric_limits<unsigned long long>::max()) {
          fallback_to_flat = true;
          break;
        }

        auto length_begin = thrust::make_transform_iterator(
            thrust::device_pointer_cast(d_lengths), UInt32ToUInt64{});
        out.raw_text_size =
            thrust::reduce(thrust::cuda::par.on(stream), length_begin,
                           length_begin + data.num_records, uint64_t{0},
                           thrust::plus<uint64_t>());

        const uint64_t plain_size = data.num_records * sizeof(int32_t);
        auto plain_chunks = gpu_compress_device_chunked(
            reinterpret_cast<const uint8_t *>(d_int_values), plain_size,
            field_slice_size, nvcomp_chunk_size, stream);

        delta_encode_int32_kernel<<<grid_size, block_size, 0, stream>>>(
            d_int_values, d_delta_values, data.num_records);
        CUDA_CHECK(cudaGetLastError());
        auto delta_chunks = gpu_compress_device_chunked(
            reinterpret_cast<const uint8_t *>(d_delta_values), plain_size,
            field_slice_size, nvcomp_chunk_size, stream);

        compute_identifier_varint_sizes_kernel<<<grid_size, block_size, 0,
                                                 stream>>>(
            d_delta_values, data.num_records, d_varint_sizes);
        CUDA_CHECK(cudaGetLastError());
        auto varint_begin = thrust::make_transform_iterator(
            thrust::device_pointer_cast(d_varint_sizes), UInt32ToUInt64{});
        thrust::exclusive_scan(thrust::cuda::par.on(stream), varint_begin,
                               varint_begin + data.num_records,
                               thrust::device_pointer_cast(d_varint_offsets));
        const uint64_t delta_varint_size = thrust::reduce(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(d_varint_sizes),
            thrust::device_pointer_cast(d_varint_sizes) + data.num_records,
            uint64_t{0}, thrust::plus<uint64_t>());
        if (delta_varint_size > 0) {
          CUDA_CHECK(cudaMalloc(&d_varint_bytes, delta_varint_size));
          scatter_identifier_varints_kernel<<<grid_size, block_size, 0,
                                              stream>>>(
              d_delta_values, d_varint_offsets, data.num_records,
              d_varint_bytes);
          CUDA_CHECK(cudaGetLastError());
        }
        auto delta_varint_chunks = gpu_compress_device_chunked(
            d_varint_bytes, delta_varint_size, field_slice_size,
            nvcomp_chunk_size, stream);

        const uint64_t plain_comp_size = sum_sizes(plain_chunks.chunk_sizes);
        const uint64_t delta_comp_size = sum_sizes(delta_chunks.chunk_sizes);
        const uint64_t delta_varint_comp_size =
            sum_sizes(delta_varint_chunks.chunk_sizes);

        if (delta_varint_comp_size < delta_comp_size &&
            delta_varint_comp_size < plain_comp_size) {
          out.encoding = IdentifierColumnEncoding::DeltaVarint;
          out.values.original_size = delta_varint_size;
          out.values.payload = std::move(delta_varint_chunks.data);
          out.compressed_value_chunk_sizes =
              std::move(delta_varint_chunks.chunk_sizes);
        } else if (delta_comp_size < plain_comp_size) {
          out.encoding = IdentifierColumnEncoding::Delta;
          out.values.original_size = plain_size;
          out.values.payload = std::move(delta_chunks.data);
          out.compressed_value_chunk_sizes =
              std::move(delta_chunks.chunk_sizes);
        } else {
          out.encoding = IdentifierColumnEncoding::Plain;
          out.values.original_size = plain_size;
          out.values.payload = std::move(plain_chunks.data);
          out.compressed_value_chunk_sizes =
              std::move(plain_chunks.chunk_sizes);
        }
      }

      cuda_free_if_set(d_string_values);
      cuda_free_if_set(d_varint_bytes);
      d_string_values = nullptr;
      d_varint_bytes = nullptr;
      if (fallback_to_flat) {
        break;
      }
    }
  } catch (...) {
    cuda_free_if_set(d_string_values);
    cuda_free_if_set(d_varint_bytes);
    cuda_free_if_set(d_lengths);
    cuda_free_if_set(d_offsets);
    cuda_free_if_set(d_int_values);
    cuda_free_if_set(d_delta_values);
    cuda_free_if_set(d_varint_sizes);
    cuda_free_if_set(d_varint_offsets);
    cuda_free_if_set(d_invalid_record);
    free_device_identifier_schema(&schema);
    throw;
  }

  cuda_free_if_set(d_lengths);
  cuda_free_if_set(d_offsets);
  cuda_free_if_set(d_int_values);
  cuda_free_if_set(d_delta_values);
  cuda_free_if_set(d_varint_sizes);
  cuda_free_if_set(d_varint_offsets);
  cuda_free_if_set(d_invalid_record);
  free_device_identifier_schema(&schema);
  if (fallback_to_flat) {
    return compress_identifiers_flat(data, field_slice_size, nvcomp_chunk_size);
  }
  return result;
}

std::vector<uint8_t>
decompress_identifiers(const CompressedIdentifierData &data,
                       uint64_t num_records) {
  if (data.mode == IdentifierCompressionMode::Flat) {
    return gpu_decompress_chunked(data.flat_data.payload,
                                  data.compressed_flat_chunk_sizes,
                                  data.flat_data.original_size);
  }
  if (data.mode != IdentifierCompressionMode::Columnar) {
    throw std::runtime_error("Unknown identifier compression mode");
  }
  if (data.layout.column_kinds.empty() || data.columns.empty() ||
      data.layout.column_kinds.size() != data.columns.size() ||
      data.layout.separators.size() + 1 != data.columns.size()) {
    throw std::runtime_error("Decoded identifier column metadata is invalid");
  }

  struct DecodedColumn {
    IdentifierColumnKind kind = IdentifierColumnKind::String;
    IdentifierColumnEncoding encoding = IdentifierColumnEncoding::Plain;
    std::vector<uint8_t> value_bytes;
    std::vector<uint32_t> lengths;
    std::vector<int32_t> decoded_int_values;
    const int32_t *int_values = nullptr;
    size_t value_offset = 0;
    size_t record_count = 0;
  };

  std::vector<DecodedColumn> columns(data.columns.size());
  for (size_t i = 0; i < data.columns.size(); ++i) {
    columns[i].kind = data.columns[i].kind;
    columns[i].encoding = data.columns[i].encoding;
    columns[i].value_bytes =
        gpu_decompress_chunked(data.columns[i].values.payload,
                               data.columns[i].compressed_value_chunk_sizes,
                               data.columns[i].values.original_size);
    if (columns[i].kind == IdentifierColumnKind::String) {
      const auto length_bytes =
          gpu_decompress_chunked(data.columns[i].lengths.payload,
                                 data.columns[i].compressed_length_chunk_sizes,
                                 data.columns[i].lengths.original_size);
      if (length_bytes.size() != num_records * sizeof(uint32_t)) {
        throw std::runtime_error(
            "Decoded identifier string-length payload has an unexpected size");
      }
      columns[i].lengths.resize(static_cast<size_t>(num_records));
      if (!columns[i].lengths.empty()) {
        std::memcpy(columns[i].lengths.data(), length_bytes.data(),
                    length_bytes.size());
      }
    } else {
      if (columns[i].encoding == IdentifierColumnEncoding::DeltaVarint) {
        auto delta_values = decode_varint_u32(columns[i].value_bytes,
                                              static_cast<size_t>(num_records));
        columns[i].decoded_int_values = delta_decode_int32(
            delta_values.data(), static_cast<size_t>(num_records));
        columns[i].int_values = columns[i].decoded_int_values.data();
      } else {
        if (columns[i].value_bytes.size() != num_records * sizeof(int32_t)) {
          throw std::runtime_error(
              "Decoded identifier numeric payload has an unexpected size");
        }
        const auto *encoded_values =
            reinterpret_cast<const int32_t *>(columns[i].value_bytes.data());
        if (columns[i].encoding == IdentifierColumnEncoding::Delta) {
          columns[i].decoded_int_values = delta_decode_int32(
              encoded_values, static_cast<size_t>(num_records));
          columns[i].int_values = columns[i].decoded_int_values.data();
        } else {
          columns[i].int_values = encoded_values;
        }
      }
    }
  }

  std::vector<uint8_t> identifiers;
  identifiers.reserve(static_cast<size_t>(data.original_size));
  for (uint64_t record = 0; record < num_records; ++record) {
    for (size_t column = 0; column < columns.size(); ++column) {
      if (columns[column].kind == IdentifierColumnKind::String) {
        const uint32_t len =
            columns[column].lengths[static_cast<size_t>(record)];
        if (columns[column].value_offset + len >
            columns[column].value_bytes.size()) {
          throw std::runtime_error(
              "Decoded identifier string column exceeds its payload");
        }
        identifiers.insert(
            identifiers.end(),
            columns[column].value_bytes.begin() +
                static_cast<std::ptrdiff_t>(columns[column].value_offset),
            columns[column].value_bytes.begin() +
                static_cast<std::ptrdiff_t>(columns[column].value_offset +
                                            len));
        columns[column].value_offset += len;
      } else {
        const auto token = std::to_string(columns[column].int_values[record]);
        identifiers.insert(identifiers.end(), token.begin(), token.end());
      }

      if (column + 1 < data.layout.separators.size() + 1) {
        const auto &sep = data.layout.separators[column];
        identifiers.insert(identifiers.end(), sep.begin(), sep.end());
      }
    }
  }

  if (identifiers.size() != data.original_size) {
    throw std::runtime_error(
        "Decoded identifier columns reconstructed an unexpected size");
  }
  return identifiers;
}

__global__ void compute_field_lengths_kernel(const uint64_t *line_offsets,
                                             uint64_t *identifier_lengths,
                                             uint64_t *basecall_lengths,
                                             uint64_t *quality_lengths,
                                             uint64_t num_records) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_records) {
    return;
  }

  const uint64_t id_line = 4 * idx;
  const uint64_t seq_line = id_line + 1;
  const uint64_t plus_line = id_line + 2;
  const uint64_t qual_line = id_line + 3;

  identifier_lengths[idx] = line_offsets[seq_line] - line_offsets[id_line] - 2;
  basecall_lengths[idx] = line_offsets[plus_line] - line_offsets[seq_line] - 1;
  quality_lengths[idx] =
      line_offsets[qual_line + 1] - line_offsets[qual_line] - 1;
}

__global__ void gather_fields_kernel(
    const uint8_t *raw_bytes, const uint64_t *line_offsets,
    const uint64_t *identifier_offsets, const uint64_t *basecall_offsets,
    const uint64_t *quality_offsets, const uint64_t *identifier_lengths,
    const uint64_t *basecall_lengths, const uint64_t *quality_lengths,
    uint8_t *identifiers, uint8_t *basecalls, uint8_t *quality_scores,
    uint64_t num_records) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_records) {
    return;
  }

  const uint64_t id_line = 4 * idx;
  const uint64_t seq_line = id_line + 1;
  const uint64_t qual_line = id_line + 3;

  const uint64_t id_src = line_offsets[id_line] + 1;
  const uint64_t seq_src = line_offsets[seq_line];
  const uint64_t qual_src = line_offsets[qual_line];

  const uint64_t id_dst = identifier_offsets[idx];
  const uint64_t seq_dst = basecall_offsets[idx];
  const uint64_t qual_dst = quality_offsets[idx];

  if (identifiers != nullptr) {
    for (uint64_t i = 0; i < identifier_lengths[idx]; ++i) {
      identifiers[id_dst + i] = raw_bytes[id_src + i];
    }
  }
  for (uint64_t i = 0; i < basecall_lengths[idx]; ++i) {
    basecalls[seq_dst + i] = raw_bytes[seq_src + i];
  }
  for (uint64_t i = 0; i < quality_lengths[idx]; ++i) {
    quality_scores[qual_dst + i] = raw_bytes[qual_src + i];
  }
}

__device__ inline uint8_t encode_basecall_2bit(uint8_t base, bool *is_n,
                                               bool *is_valid,
                                               BasecallPackOrder pack_order) {
  switch (base) {
  case 'A':
  case 'a':
    *is_n = false;
    *is_valid = true;
    return pack_order == BasecallPackOrder::Acgt ? 0 : 3;
  case 'C':
  case 'c':
    *is_n = false;
    *is_valid = true;
    return pack_order == BasecallPackOrder::Acgt ? 1 : 2;
  case 'G':
  case 'g':
    *is_n = false;
    *is_valid = true;
    return pack_order == BasecallPackOrder::Acgt ? 2 : 1;
  case 'T':
  case 't':
    *is_n = false;
    *is_valid = true;
    return pack_order == BasecallPackOrder::Acgt ? 3 : 0;
  case 'N':
  case 'n':
    *is_n = true;
    *is_valid = true;
    return pack_order == BasecallPackOrder::Acgt ? 0 : 3;
  default:
    *is_n = false;
    *is_valid = false;
    return 0;
  }
}

__global__ void count_n_basecalls_kernel(const uint8_t *basecalls,
                                         uint64_t basecall_count,
                                         uint32_t *n_counts,
                                         uint64_t *invalid_position,
                                         BasecallPackOrder pack_order) {
  const uint64_t block_index = blockIdx.x;
  const uint64_t block_start =
      block_index * static_cast<uint64_t>(BASECALL_N_BLOCK_SIZE);
  if (block_start >= basecall_count) {
    return;
  }

  __shared__ uint32_t shared_count;
  if (threadIdx.x == 0) {
    shared_count = 0;
  }
  __syncthreads();

  const uint64_t block_end =
      min(block_start + static_cast<uint64_t>(BASECALL_N_BLOCK_SIZE),
          basecall_count);
  uint32_t local_count = 0;
  for (uint64_t index = block_start + threadIdx.x; index < block_end;
       index += blockDim.x) {
    bool is_n = false;
    bool is_valid = false;
    encode_basecall_2bit(basecalls[index], &is_n, &is_valid, pack_order);
    if (!is_valid) {
      atomicMin(reinterpret_cast<unsigned long long *>(invalid_position),
                static_cast<unsigned long long>(index));
      continue;
    }
    if (is_n) {
      ++local_count;
    }
  }

  if (local_count != 0) {
    atomicAdd(&shared_count, local_count);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    n_counts[block_index] = shared_count;
  }
}

__global__ void pack_basecalls_2bit_kernel(const uint8_t *basecalls,
                                           uint64_t basecall_count,
                                           uint8_t *packed_bases,
                                           uint64_t *invalid_position,
                                           BasecallPackOrder pack_order) {
  const uint64_t packed_index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t base_index = packed_index * 4;
  if (base_index >= basecall_count) {
    return;
  }

  uint8_t packed_value = 0;
  for (uint32_t lane = 0; lane < 4; ++lane) {
    const uint64_t index = base_index + lane;
    if (index >= basecall_count) {
      break;
    }

    bool is_n = false;
    bool is_valid = false;
    const uint8_t code =
        encode_basecall_2bit(basecalls[index], &is_n, &is_valid, pack_order);
    if (!is_valid) {
      atomicMin(reinterpret_cast<unsigned long long *>(invalid_position),
                static_cast<unsigned long long>(index));
      return;
    }
    packed_value |= static_cast<uint8_t>(code << (2 * lane));
  }
  packed_bases[packed_index] = packed_value;
}

__global__ void scatter_n_positions_kernel(const uint8_t *basecalls,
                                           uint64_t basecall_count,
                                           const uint64_t *n_offsets,
                                           uint16_t *n_positions,
                                           uint64_t *invalid_position,
                                           BasecallPackOrder pack_order) {
  const uint64_t block_index = blockIdx.x;
  const uint64_t block_start =
      block_index * static_cast<uint64_t>(BASECALL_N_BLOCK_SIZE);
  if (block_start >= basecall_count) {
    return;
  }

  const uint64_t block_end =
      min(block_start + static_cast<uint64_t>(BASECALL_N_BLOCK_SIZE),
          basecall_count);

  typedef cub::BlockScan<uint32_t, 256> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ uint32_t block_cursor;

  if (threadIdx.x == 0) {
    block_cursor = 0;
  }
  __syncthreads();

  for (uint64_t index = block_start + threadIdx.x; index < block_end;
       index += blockDim.x) {
    bool is_n = false;
    bool is_valid = false;
    encode_basecall_2bit(basecalls[index], &is_n, &is_valid, pack_order);
    if (!is_valid) {
      atomicMin(reinterpret_cast<unsigned long long *>(invalid_position),
                static_cast<unsigned long long>(index));
      is_n = false;
    }

    uint32_t thread_data = is_n ? 1 : 0;
    uint32_t slot_offset;
    uint32_t aggregate;
    BlockScan(temp_storage).ExclusiveSum(thread_data, slot_offset, aggregate);
    __syncthreads();

    uint32_t local_cursor = 0;
    if (threadIdx.x == 0) {
      local_cursor = atomicAdd(&block_cursor, aggregate);
    }
    __shared__ uint32_t shared_cursor;
    if (threadIdx.x == 0) {
      shared_cursor = local_cursor;
    }
    __syncthreads();

    if (is_n) {
      n_positions[n_offsets[block_index] + shared_cursor + slot_offset] =
          static_cast<uint16_t>(index - block_start);
    }
  }
}

__global__ void delta_encode_n_positions_kernel(uint16_t *n_positions,
                                                const uint64_t *n_offsets,
                                                const uint32_t *n_counts,
                                                uint64_t block_count) {
  const uint64_t block_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_index >= block_count) {
    return;
  }

  uint32_t count = n_counts[block_index];
  if (count <= 1) {
    return;
  }

  uint64_t offset = n_offsets[block_index];
  for (int32_t i = static_cast<int32_t>(count) - 1; i > 0; --i) {
    n_positions[offset + i] =
        n_positions[offset + i] - n_positions[offset + i - 1];
  }
}

ChunkedCompressedBuffer gpu_compress_device_chunked(const uint8_t *d_input,
                                                    size_t input_size,
                                                    size_t field_slice_size,
                                                    size_t nvcomp_chunk_size,
                                                    cudaStream_t stream) {
  ChunkedCompressedBuffer result;
  if (input_size == 0) {
    return result;
  }
  if (field_slice_size == 0 || field_slice_size > MAX_FIELD_SLICE_SIZE) {
    throw std::runtime_error(
        "Requested field slice size is out of supported range");
  }

  for (size_t offset = 0; offset < input_size; offset += field_slice_size) {
    const size_t slice_size = std::min(field_slice_size, input_size - offset);
    auto compressed = nvcomp_zstd_compress_device(d_input + offset, slice_size,
                                                  nvcomp_chunk_size, stream);
    result.chunk_sizes.push_back(compressed.payload.size());
    result.data.insert(result.data.end(), compressed.payload.begin(),
                       compressed.payload.end());
  }

  return result;
}

std::vector<uint8_t>
gpu_decompress_chunked(const std::vector<uint8_t> &compressed,
                       const std::vector<uint64_t> &chunk_sizes,
                       uint64_t expected_size) {
  if (compressed.empty()) {
    if (!chunk_sizes.empty() || expected_size != 0) {
      throw std::runtime_error(
          "Compressed chunk metadata is inconsistent for empty payload");
    }
    return {};
  }

  if (sum_sizes(chunk_sizes) != compressed.size()) {
    throw std::runtime_error(
        "Compressed chunk sizes do not match payload size");
  }

  std::vector<uint8_t> output;
  output.reserve(expected_size);

  size_t offset = 0;
  for (uint64_t chunk_size : chunk_sizes) {
    std::vector<uint8_t> slice(
        compressed.begin() + static_cast<std::ptrdiff_t>(offset),
        compressed.begin() + static_cast<std::ptrdiff_t>(offset + chunk_size));
    const size_t expected_chunk_size = static_cast<size_t>(std::min<uint64_t>(
        MAX_FIELD_SLICE_SIZE, expected_size - output.size()));
    ZstdCompressedBlock block{std::move(slice), expected_chunk_size};
    auto decompressed = nvcomp_zstd_decompress(block);
    output.insert(output.end(), decompressed.begin(), decompressed.end());
    offset += chunk_size;
  }

  if (offset != compressed.size() || output.size() != expected_size) {
    throw std::runtime_error(
        "Chunked decompression produced an unexpected size");
  }

  return output;
}

BscChunkedBuffer bsc_compress_device_chunked(const uint8_t *d_input,
                                             size_t input_size,
                                             size_t chunk_size,
                                             const BscConfig &config,
                                             cudaStream_t stream) {
  BscChunkedBuffer result;
  if (input_size == 0) {
    return result;
  }
  if (d_input == nullptr) {
    throw std::runtime_error("BSC device compression input pointer is null");
  }
  if (chunk_size == 0) {
    throw std::runtime_error("BSC compression chunk size must be non-zero");
  }

  const size_t chunk_count = (input_size + chunk_size - 1) / chunk_size;
  const auto resolved = resolve_bsc_config(config, chunk_count);
  initialize_bsc_backend(resolved.backend);

  struct LoadedChunk {
    size_t index = 0;
    std::vector<uint8_t> data;
  };

  std::deque<LoadedChunk> queue;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  const size_t max_queue_depth = std::max<size_t>(1, resolved.parallelism * 2);
  bool producer_done = false;
  std::exception_ptr worker_error;
  std::mutex error_mutex;
  std::atomic<bool> failed{false};

  std::vector<std::vector<uint8_t>> compressed_chunks(chunk_count);
  result.compressed_chunk_sizes.resize(chunk_count);
  result.uncompressed_chunk_sizes.resize(chunk_count);

  const auto worker = [&]() {
    try {
      while (true) {
        LoadedChunk chunk;
        {
          std::unique_lock<std::mutex> lock(queue_mutex);
          queue_cv.wait(lock, [&]() {
            return failed.load(std::memory_order_relaxed) || !queue.empty() ||
                   producer_done;
          });
          if (failed.load(std::memory_order_relaxed)) {
            return;
          }
          if (queue.empty()) {
            if (producer_done) {
              return;
            }
            continue;
          }
          chunk = std::move(queue.front());
          queue.pop_front();
        }
        queue_cv.notify_all();

        auto compressed = bsc_compress_block(
            chunk.data.data(), chunk.data.size(), resolved.backend);
        result.compressed_chunk_sizes[chunk.index] =
            static_cast<uint64_t>(compressed.size());
        result.uncompressed_chunk_sizes[chunk.index] = chunk.data.size();
        compressed_chunks[chunk.index] = std::move(compressed);
      }
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
      std::lock_guard<std::mutex> lock(error_mutex);
      if (worker_error == nullptr) {
        worker_error = std::current_exception();
      }
      queue_cv.notify_all();
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(resolved.parallelism);
  for (size_t i = 0; i < resolved.parallelism; ++i) {
    workers.emplace_back(worker);
  }

  try {
    for (size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
      if (failed.load(std::memory_order_relaxed)) {
        break;
      }

      const size_t offset = chunk_index * chunk_size;
      const size_t current_chunk_size =
          std::min(chunk_size, input_size - offset);
      std::vector<uint8_t> host_chunk(current_chunk_size);
      CUDA_CHECK(cudaMemcpyAsync(host_chunk.data(), d_input + offset,
                                 current_chunk_size, cudaMemcpyDeviceToHost,
                                 stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      std::unique_lock<std::mutex> lock(queue_mutex);
      queue_cv.wait(lock, [&]() {
        return failed.load(std::memory_order_relaxed) ||
               queue.size() < max_queue_depth;
      });
      if (failed.load(std::memory_order_relaxed)) {
        break;
      }
      queue.push_back(LoadedChunk{chunk_index, std::move(host_chunk)});
      lock.unlock();
      queue_cv.notify_all();
    }
  } catch (...) {
    failed.store(true, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (worker_error == nullptr) {
        worker_error = std::current_exception();
      }
    }
    queue_cv.notify_all();
  }

  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    producer_done = true;
  }
  queue_cv.notify_all();
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

CompressedBasecallData
compress_basecalls_device(const uint8_t *d_basecalls, uint64_t basecall_count,
                          size_t field_slice_size, size_t nvcomp_chunk_size,
                          cudaStream_t stream, const BscConfig &bsc_config) {
  CompressedBasecallData result;
  result.original_size = basecall_count;
  result.n_block_size = BASECALL_N_BLOCK_SIZE;
  result.packed_codec = BasecallPackedCodec::Zstd;
  result.pack_order = bsc_config.basecall_pack_order;
  if (basecall_count == 0) {
    return result;
  }

  const uint64_t packed_size = (basecall_count + 3) / 4;
  const uint64_t block_count =
      (basecall_count + BASECALL_N_BLOCK_SIZE - 1) / BASECALL_N_BLOCK_SIZE;
  result.packed_bases.original_size = packed_size;

  EncodedBasecallBuffers encoded;
  uint64_t *d_invalid_position = nullptr;
  uint64_t invalid_position = std::numeric_limits<uint64_t>::max();

  try {
    CUDA_CHECK(cudaMalloc(&encoded.packed_bases, packed_size));
    CUDA_CHECK(cudaMalloc(&encoded.n_counts, block_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&encoded.n_offsets, block_count * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_invalid_position, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_invalid_position, &invalid_position,
                               sizeof(uint64_t), cudaMemcpyHostToDevice,
                               stream));

    const uint32_t kernel_block_size = 256;
    count_n_basecalls_kernel<<<static_cast<uint32_t>(block_count),
                               kernel_block_size, 0, stream>>>(
        d_basecalls, basecall_count, encoded.n_counts, d_invalid_position,
        bsc_config.basecall_pack_order);
    CUDA_CHECK(cudaGetLastError());

    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           thrust::device_pointer_cast(encoded.n_counts),
                           thrust::device_pointer_cast(encoded.n_counts) +
                               block_count,
                           thrust::device_pointer_cast(encoded.n_offsets));

    std::vector<uint32_t> h_n_counts(block_count);
    CUDA_CHECK(cudaMemcpyAsync(h_n_counts.data(), encoded.n_counts,
                               block_count * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));

    const uint32_t pack_grid_size = static_cast<uint32_t>(
        (packed_size + kernel_block_size - 1) / kernel_block_size);
    pack_basecalls_2bit_kernel<<<pack_grid_size, kernel_block_size, 0,
                                 stream>>>(
        d_basecalls, basecall_count, encoded.packed_bases, d_invalid_position,
        bsc_config.basecall_pack_order);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&invalid_position, d_invalid_position,
                          sizeof(uint64_t), cudaMemcpyDeviceToHost));
    if (invalid_position != std::numeric_limits<uint64_t>::max()) {
      throw std::runtime_error("Encountered a non-ACGTN basecall at offset " +
                               std::to_string(invalid_position));
    }

    uint64_t total_n_count = 0;
    std::vector<uint16_t> h_n_counts_16(block_count);
    for (uint64_t i = 0; i < block_count; ++i) {
      if (h_n_counts[i] > BASECALL_N_BLOCK_SIZE) {
        throw std::runtime_error("Basecall N-count overflowed its block size");
      }
      h_n_counts_16[i] = static_cast<uint16_t>(h_n_counts[i]);
      total_n_count += h_n_counts[i];
    }

    std::vector<uint8_t> h_n_count_bytes(block_count * sizeof(uint16_t));
    if (!h_n_counts_16.empty()) {
      std::memcpy(h_n_count_bytes.data(), h_n_counts_16.data(),
                  h_n_count_bytes.size());
    }
    result.n_counts =
        nvcomp_zstd_compress(h_n_count_bytes, nvcomp_chunk_size, stream);

    if (total_n_count > 0) {
      result.n_positions.original_size = total_n_count * sizeof(uint16_t);
      CUDA_CHECK(
          cudaMalloc(&encoded.n_positions, result.n_positions.original_size));
      scatter_n_positions_kernel<<<static_cast<uint32_t>(block_count),
                                   kernel_block_size, 0, stream>>>(
          d_basecalls, basecall_count, encoded.n_offsets, encoded.n_positions,
          d_invalid_position, bsc_config.basecall_pack_order);
      CUDA_CHECK(cudaGetLastError());
      const uint32_t delta_block_size = 256;
      const uint32_t delta_grid_size = static_cast<uint32_t>(
          (block_count + delta_block_size - 1) / delta_block_size);
      delta_encode_n_positions_kernel<<<delta_grid_size, delta_block_size, 0,
                                        stream>>>(
          encoded.n_positions, encoded.n_offsets, encoded.n_counts,
          block_count);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpy(&invalid_position, d_invalid_position,
                            sizeof(uint64_t), cudaMemcpyDeviceToHost));
      if (invalid_position != std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error("Encountered a non-ACGTN basecall at offset " +
                                 std::to_string(invalid_position));
      }
    }

    if (bsc_config.base_bsc) {
      auto packed_chunks = bsc_compress_device_chunked(
          encoded.packed_bases, packed_size, BSC_QUALITY_CHUNK_SIZE, bsc_config,
          stream);
      result.packed_codec = BasecallPackedCodec::Bsc;
      result.packed_bases.payload = std::move(packed_chunks.data);
      result.compressed_packed_chunk_sizes =
          std::move(packed_chunks.compressed_chunk_sizes);
      result.uncompressed_packed_chunk_sizes =
          std::move(packed_chunks.uncompressed_chunk_sizes);
    } else {
      auto packed_chunks = gpu_compress_device_chunked(
          encoded.packed_bases, packed_size, field_slice_size,
          nvcomp_chunk_size, stream);
      result.packed_codec = BasecallPackedCodec::Zstd;
      result.packed_bases.payload = std::move(packed_chunks.data);
      result.compressed_packed_chunk_sizes =
          std::move(packed_chunks.chunk_sizes);
      result.uncompressed_packed_chunk_sizes.clear();
    }

    if (result.n_positions.original_size > 0) {
      auto n_position_chunks = gpu_compress_device_chunked(
          reinterpret_cast<const uint8_t *>(encoded.n_positions),
          result.n_positions.original_size, field_slice_size, nvcomp_chunk_size,
          stream);
      result.n_positions.payload = std::move(n_position_chunks.data);
      result.compressed_n_position_chunk_sizes =
          std::move(n_position_chunks.chunk_sizes);
    }
  } catch (...) {
    cuda_free_if_set(d_invalid_position);
    cuda_free_if_set(encoded.n_positions);
    cuda_free_if_set(encoded.n_offsets);
    cuda_free_if_set(encoded.n_counts);
    cuda_free_if_set(encoded.packed_bases);
    throw;
  }

  cuda_free_if_set(d_invalid_position);
  cuda_free_if_set(encoded.n_positions);
  cuda_free_if_set(encoded.n_offsets);
  cuda_free_if_set(encoded.n_counts);
  cuda_free_if_set(encoded.packed_bases);
  return result;
}

std::vector<uint8_t>
decode_basecalls(const CompressedBasecallData &compressed,
                 const std::vector<uint8_t> &n_count_bytes,
                 const std::vector<uint8_t> &packed_bases,
                 const std::vector<uint8_t> &n_position_bytes) {
  if (compressed.original_size == 0) {
    return {};
  }

  if (compressed.n_block_size == 0) {
    throw std::runtime_error("Decoded basecall metadata is missing block size");
  }

  const uint64_t expected_block_count =
      (compressed.original_size + compressed.n_block_size - 1) /
      compressed.n_block_size;
  if (n_count_bytes.size() != expected_block_count * sizeof(uint16_t)) {
    throw std::runtime_error("Decoded basecall N-count metadata is invalid");
  }
  const auto *n_counts =
      reinterpret_cast<const uint16_t *>(n_count_bytes.data());

  const size_t expected_packed_size =
      static_cast<size_t>((compressed.original_size + 3) / 4);
  if (packed_bases.size() != expected_packed_size) {
    throw std::runtime_error(
        "Decoded packed basecalls have an unexpected size");
  }

  uint64_t total_n_count = 0;
  for (uint64_t i = 0; i < expected_block_count; ++i) {
    total_n_count += n_counts[i];
  }
  if (n_position_bytes.size() != total_n_count * sizeof(uint16_t)) {
    throw std::runtime_error(
        "Decoded N-position payload has an unexpected size");
  }

  std::vector<uint8_t> basecalls(compressed.original_size);
  static constexpr uint8_t BASECALL_DECODE_TABLE_ACGT[4] = {'A', 'C', 'G',
                                                            'T'};
  static constexpr uint8_t BASECALL_DECODE_TABLE_TGCA[4] = {'T', 'G', 'C',
                                                            'A'};
  const uint8_t *decode_table =
      compressed.pack_order == BasecallPackOrder::Acgt
          ? BASECALL_DECODE_TABLE_ACGT
          : BASECALL_DECODE_TABLE_TGCA;
  for (uint64_t index = 0; index < compressed.original_size; ++index) {
    const uint8_t packed_value = packed_bases[index / 4];
    const uint8_t code =
        static_cast<uint8_t>((packed_value >> (2 * (index % 4))) & 0x3);
    basecalls[index] = decode_table[code];
  }

  const auto *n_positions =
      reinterpret_cast<const uint16_t *>(n_position_bytes.data());
  uint64_t n_offset = 0;
  for (uint64_t block_index = 0; block_index < expected_block_count;
       ++block_index) {
    const uint64_t block_start = block_index * compressed.n_block_size;
    const uint64_t block_end = std::min<uint64_t>(
        block_start + compressed.n_block_size, compressed.original_size);
    uint16_t current_pos = 0;
    for (uint16_t count = 0; count < n_counts[block_index]; ++count) {
      current_pos += n_positions[n_offset++];
      if (block_start + current_pos >= block_end) {
        throw std::runtime_error(
            "Decoded N-position metadata points outside its basecall block");
      }
      basecalls[block_start + current_pos] = 'N';
    }
  }

  return basecalls;
}

FastqData rebuild_fastq(const std::vector<uint64_t> &line_offsets,
                        const std::vector<uint8_t> &identifiers,
                        const std::vector<uint8_t> &basecalls,
                        const std::vector<uint8_t> &quality_scores,
                        uint64_t num_records) {
  FastqData data;
  data.line_offsets = line_offsets;
  data.num_records = num_records;

  if (line_offsets.empty()) {
    throw std::runtime_error("Decoded line-offset metadata is empty");
  }

  const uint64_t file_size = line_offsets.back();
  data.raw_bytes.resize(file_size);
  data.quality_lengths.resize(static_cast<size_t>(num_records), 0);

  uint64_t id_offset = 0;
  uint64_t seq_offset = 0;
  uint64_t qual_offset = 0;
  bool fixed_quality_length_set = false;
  uint32_t fixed_quality_length = 0;

  for (uint64_t record = 0; record < num_records; ++record) {
    const uint64_t id_line = 4 * record;
    const uint64_t seq_line = id_line + 1;
    const uint64_t plus_line = id_line + 2;
    const uint64_t qual_line = id_line + 3;

    const uint64_t id_start = line_offsets[id_line];
    const uint64_t seq_start = line_offsets[seq_line];
    const uint64_t plus_start = line_offsets[plus_line];
    const uint64_t qual_start = line_offsets[qual_line];
    const uint64_t next_start = line_offsets[qual_line + 1];

    const uint64_t id_len = seq_start - id_start - 2;
    const uint64_t seq_len = plus_start - seq_start - 1;
    const uint64_t plus_len = qual_start - plus_start - 1;
    const uint64_t qual_len = next_start - qual_start - 1;
    if (qual_len > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("Decoded quality length exceeds uint32_t range");
    }
    data.quality_lengths[static_cast<size_t>(record)] =
        static_cast<uint32_t>(qual_len);
    if (!fixed_quality_length_set) {
      fixed_quality_length = static_cast<uint32_t>(qual_len);
      fixed_quality_length_set = true;
    } else if (fixed_quality_length != static_cast<uint32_t>(qual_len)) {
      data.quality_layout = QualityLayoutKind::VariableLength;
    }

    if (plus_len != 1) {
      throw std::runtime_error("Decoded line-offset metadata is incompatible "
                               "with ignored plus lines");
    }
    if (id_offset + id_len > identifiers.size() ||
        seq_offset + seq_len > basecalls.size() ||
        qual_offset + qual_len > quality_scores.size()) {
      throw std::runtime_error(
          "Decoded FASTQ field stream exceeds its uncompressed size");
    }

    data.raw_bytes[id_start] = '@';
    std::memcpy(data.raw_bytes.data() + id_start + 1,
                identifiers.data() + id_offset, id_len);
    data.raw_bytes[seq_start - 1] = '\n';

    std::memcpy(data.raw_bytes.data() + seq_start,
                basecalls.data() + seq_offset, seq_len);
    data.raw_bytes[plus_start - 1] = '\n';

    data.raw_bytes[plus_start] = '+';
    data.raw_bytes[qual_start - 1] = '\n';

    std::memcpy(data.raw_bytes.data() + qual_start,
                quality_scores.data() + qual_offset, qual_len);
    data.raw_bytes[next_start - 1] = '\n';

    id_offset += id_len;
    seq_offset += seq_len;
    qual_offset += qual_len;
  }

  if (id_offset != identifiers.size() || seq_offset != basecalls.size() ||
      qual_offset != quality_scores.size()) {
    throw std::runtime_error(
        "Decoded FASTQ field streams contain trailing bytes outside the index");
  }

  if (num_records == 0) {
    data.quality_layout = QualityLayoutKind::FixedLength;
    data.fixed_quality_length = 0;
  } else if (data.quality_layout == QualityLayoutKind::VariableLength) {
    data.fixed_quality_length = 0;
  } else {
    data.quality_layout = QualityLayoutKind::FixedLength;
    data.fixed_quality_length = fixed_quality_length;
  }

  return data;
}

} // namespace

std::vector<uint8_t> gpu_compress(const std::vector<uint8_t> &input,
                                  size_t chunk_size) {
  if (input.empty()) {
    return {};
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint8_t *d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size()));
  CUDA_CHECK(cudaMemcpyAsync(d_input, input.data(), input.size(),
                             cudaMemcpyHostToDevice, stream));

  std::vector<uint8_t> output =
      nvcomp_zstd_compress_device(d_input, input.size(), chunk_size, stream)
          .payload;

  cudaFree(d_input);
  cudaStreamDestroy(stream);
  return output;
}

std::vector<uint8_t> gpu_decompress(const std::vector<uint8_t> &compressed) {
  if (compressed.empty()) {
    return {};
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<uint8_t> output =
      nvcomp_zstd_decompress(ZstdCompressedBlock{compressed, compressed.size()},
                             NVCOMP_ZSTD_CHUNK_SIZE_DEFAULT, stream);
  cudaStreamDestroy(stream);
  return output;
}

CompressedFastqData compress_fastq(const FastqData &data, size_t chunk_size,
                                   const BscConfig &bsc_config) {
  using Clock = std::chrono::steady_clock;
  const auto compression_start = Clock::now();
  const FastqFieldStats stats = compute_field_stats(data);
  const size_t field_slice_size = MAX_FIELD_SLICE_SIZE;

  CompressedFastqData result;
  result.num_records = data.num_records;
  result.identifiers.original_size = stats.identifiers_size;
  result.quality_scores.original_size = stats.quality_scores_size;
  result.quality_codec = bsc_config.quality_codec;
  // Deprecated for now: keep fixed/variable detection in FastqData, but store
  // quality scores in the original row-major layout for all files.
  result.quality_layout = QualityLayoutKind::VariableLength;
  result.fixed_quality_length = 0;
  result.line_lengths.original_size = stats.line_length_size;
  result.line_offset_count = data.line_offsets.size();
  const bool use_columnar_identifiers = data.identifier_layout.columnar;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint8_t *d_raw_bytes = nullptr;
  uint64_t *d_line_offsets = nullptr;
  uint64_t *d_identifier_lengths = nullptr;
  uint64_t *d_basecall_lengths = nullptr;
  uint64_t *d_quality_lengths = nullptr;
  uint64_t *d_identifier_offsets = nullptr;
  uint64_t *d_basecall_offsets = nullptr;
  uint64_t *d_quality_offsets = nullptr;
  uint32_t *d_line_lengths = nullptr;
  DeviceFieldBuffers fields;
  long long field_prep_ms = 0;
  long long identifier_ms = 0;
  long long basecall_ms = 0;
  long long quality_ms = 0;
  long long line_length_ms = 0;

  try {
    const auto field_prep_start = Clock::now();
    if (!data.raw_bytes.empty()) {
      CUDA_CHECK(cudaMalloc(&d_raw_bytes, data.raw_bytes.size()));
      CUDA_CHECK(cudaMemcpyAsync(d_raw_bytes, data.raw_bytes.data(),
                                 data.raw_bytes.size(), cudaMemcpyHostToDevice,
                                 stream));
    }

    if (!data.line_offsets.empty()) {
      CUDA_CHECK(cudaMalloc(&d_line_offsets,
                            data.line_offsets.size() * sizeof(uint64_t)));
      CUDA_CHECK(cudaMemcpyAsync(d_line_offsets, data.line_offsets.data(),
                                 data.line_offsets.size() * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, stream));
    }

    if (data.num_records > 0) {
      CUDA_CHECK(cudaMalloc(&d_identifier_lengths,
                            data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_basecall_lengths, data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_quality_lengths, data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&d_identifier_offsets,
                            data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_basecall_offsets, data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_quality_offsets, data.num_records * sizeof(uint64_t)));

      const uint32_t block_size = 256;
      const uint32_t grid_size = static_cast<uint32_t>(
          (data.num_records + block_size - 1) / block_size);

      compute_field_lengths_kernel<<<grid_size, block_size, 0, stream>>>(
          d_line_offsets, d_identifier_lengths, d_basecall_lengths,
          d_quality_lengths, data.num_records);
      CUDA_CHECK(cudaGetLastError());

      thrust::exclusive_scan(thrust::cuda::par.on(stream),
                             thrust::device_pointer_cast(d_identifier_lengths),
                             thrust::device_pointer_cast(d_identifier_lengths) +
                                 data.num_records,
                             thrust::device_pointer_cast(d_identifier_offsets));
      thrust::exclusive_scan(thrust::cuda::par.on(stream),
                             thrust::device_pointer_cast(d_basecall_lengths),
                             thrust::device_pointer_cast(d_basecall_lengths) +
                                 data.num_records,
                             thrust::device_pointer_cast(d_basecall_offsets));
      thrust::exclusive_scan(thrust::cuda::par.on(stream),
                             thrust::device_pointer_cast(d_quality_lengths),
                             thrust::device_pointer_cast(d_quality_lengths) +
                                 data.num_records,
                             thrust::device_pointer_cast(d_quality_offsets));

      if (!use_columnar_identifiers && stats.identifiers_size > 0) {
        CUDA_CHECK(cudaMalloc(&fields.identifiers, stats.identifiers_size));
      }
      if (stats.basecalls_size > 0) {
        CUDA_CHECK(cudaMalloc(&fields.basecalls, stats.basecalls_size));
      }
      if (stats.quality_scores_size > 0) {
        CUDA_CHECK(
            cudaMalloc(&fields.quality_scores, stats.quality_scores_size));
      }

      gather_fields_kernel<<<grid_size, block_size, 0, stream>>>(
          d_raw_bytes, d_line_offsets, d_identifier_offsets, d_basecall_offsets,
          d_quality_offsets, d_identifier_lengths, d_basecall_lengths,
          d_quality_lengths, fields.identifiers, fields.basecalls,
          fields.quality_scores, data.num_records);
      CUDA_CHECK(cudaGetLastError());
    }

    if (!data.line_offsets.empty()) {
      const uint64_t line_length_count = data.line_offsets.size();
      CUDA_CHECK(
          cudaMalloc(&d_line_lengths, line_length_count * sizeof(uint32_t)));
      delta_encode_offsets_to_lengths(d_line_offsets, d_line_lengths,
                                      line_length_count, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    field_prep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        Clock::now() - field_prep_start)
                        .count();

    std::cerr << "Compressing identifiers (" << stats.identifiers_size
              << " bytes)..." << std::endl;
    const auto identifier_start = Clock::now();
    if (use_columnar_identifiers) {
      result.identifiers =
          compress_identifiers_columnar(data, d_raw_bytes, d_line_offsets,
                                        field_slice_size, chunk_size, stream);
      uint64_t compressed_identifier_size =
          result.identifiers.flat_data.payload.size();
      for (const auto &column : result.identifiers.columns) {
        compressed_identifier_size += column.values.payload.size();
        compressed_identifier_size += column.lengths.payload.size();
      }
      std::cerr << "  Mode: "
                << (result.identifiers.mode ==
                            IdentifierCompressionMode::Columnar
                        ? "columnar"
                        : "flat")
                << std::endl;
      if (result.identifiers.mode == IdentifierCompressionMode::Columnar) {
        std::cerr << "  Columns: " << result.identifiers.columns.size()
                  << std::endl;
        for (size_t i = 0; i < result.identifiers.columns.size(); ++i) {
          const auto &column = result.identifiers.columns[i];
          const uint64_t raw_value_size = column.raw_text_size;
          const uint64_t comp_value_size = column.values.payload.size();
          const uint64_t raw_length_size = column.lengths.original_size;
          const uint64_t comp_length_size = column.lengths.payload.size();
          const uint64_t raw_total = raw_value_size + raw_length_size;
          const uint64_t comp_total = comp_value_size + comp_length_size;
          std::cerr << "    [" << i << "] "
                    << identifier_column_kind_name(column.kind) << "/"
                    << identifier_column_encoding_name(column.encoding)
                    << " values " << raw_value_size << " -> " << comp_value_size
                    << " B";
          if (column.kind == IdentifierColumnKind::String) {
            std::cerr << ", lengths " << raw_length_size << " -> "
                      << comp_length_size << " B";
          }
          std::cerr << ", total " << raw_total << " -> " << comp_total << " B ("
                    << compression_ratio_percent(comp_total, raw_total) << " %)"
                    << std::endl;
        }
      }
      std::cerr << "  -> " << compressed_identifier_size << " bytes"
                << std::endl;
    } else {
      auto id_chunks = gpu_compress_device_chunked(
          fields.identifiers, stats.identifiers_size, field_slice_size,
          chunk_size, stream);
      result.identifiers.mode = IdentifierCompressionMode::Flat;
      result.identifiers.flat_data.original_size = stats.identifiers_size;
      result.identifiers.flat_data.payload = std::move(id_chunks.data);
      result.identifiers.compressed_flat_chunk_sizes =
          std::move(id_chunks.chunk_sizes);
      std::cerr << "  Mode: flat" << std::endl;
      std::cerr << "  -> " << result.identifiers.flat_data.payload.size()
                << " bytes" << std::endl;
    }
    identifier_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        Clock::now() - identifier_start)
                        .count();
    std::cerr << "  Time: " << identifier_ms << " ms" << std::endl;

    std::cerr << "Compressing basecalls (" << stats.basecalls_size
              << " bytes)..." << std::endl;
    const auto basecall_start = Clock::now();
    result.basecalls = compress_basecalls_device(
        fields.basecalls, stats.basecalls_size, field_slice_size, chunk_size,
        stream, bsc_config);
    const uint64_t compressed_basecall_size =
        result.basecalls.packed_bases.payload.size() +
        result.basecalls.n_counts.payload.size() +
        result.basecalls.n_positions.payload.size();
    uint64_t total_n_count = 0;
    const uint64_t basecall_block_count =
        result.basecalls.original_size == 0
            ? 0
            : (result.basecalls.original_size + result.basecalls.n_block_size -
               1) /
                  result.basecalls.n_block_size;
    if (result.basecalls.n_counts.original_size !=
        basecall_block_count * sizeof(uint16_t)) {
      throw std::runtime_error(
          "Compressed N-count payload has an unexpected size");
    }
    if (result.basecalls.n_positions.original_size % sizeof(uint16_t) != 0) {
      throw std::runtime_error(
          "Compressed N-position payload has an unexpected size");
    }
    total_n_count =
        result.basecalls.n_positions.original_size / sizeof(uint16_t);
    std::cerr << "  Packed bases: "
              << result.basecalls.packed_bases.original_size
              << " bytes, N positions: " << total_n_count << std::endl;
    std::cerr << "  -> " << compressed_basecall_size << " bytes" << std::endl;
    basecall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      Clock::now() - basecall_start)
                      .count();
    std::cerr << "  Time: " << basecall_ms << " ms" << std::endl;

    std::cerr << "Compressing quality scores (" << stats.quality_scores_size
              << " bytes)..." << std::endl;
    const auto quality_start = Clock::now();
    bool do_transpose = data.quality_layout == QualityLayoutKind::FixedLength &&
                        data.fixed_quality_length != 0 &&
                        bsc_config.quality_codec == QualityCodec::Zstd &&
                        bsc_config.zstd_transpose_quality;
    result.quality_transposed = do_transpose;

    if (data.quality_layout == QualityLayoutKind::FixedLength &&
        data.fixed_quality_length != 0) {
      if (do_transpose) {
        std::cerr << "  Layout: column-major fixed-length (transposed)"
                  << std::endl;
      } else {
        std::cerr << "  Layout: row-major fixed-length (column-major disabled)"
                  << std::endl;
      }
    } else {
      std::cerr << "  Layout: row-major variable-length" << std::endl;
    }
    std::cerr << "  Codec: " << quality_codec_name(bsc_config.quality_codec)
              << std::endl;
    result.compressed_quality_chunk_sizes.clear();
    result.uncompressed_quality_chunk_sizes.clear();
    if (bsc_config.quality_codec == QualityCodec::Zstd) {
      std::vector<uint8_t> transposed_host;
      if (do_transpose) {
        std::vector<uint8_t> host_quality(stats.quality_scores_size);
        CUDA_CHECK(cudaMemcpyAsync(host_quality.data(), fields.quality_scores,
                                   stats.quality_scores_size,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        transposed_host = transpose_fixed_length_quality_scores(
            host_quality, data.num_records, data.fixed_quality_length);
        CUDA_CHECK(cudaMemcpyAsync(
            fields.quality_scores, transposed_host.data(),
            stats.quality_scores_size, cudaMemcpyHostToDevice, stream));
      }

      auto qual_chunks = gpu_compress_device_chunked(
          fields.quality_scores, stats.quality_scores_size, field_slice_size,
          chunk_size, stream);
      result.quality_scores.payload = std::move(qual_chunks.data);
      result.compressed_quality_chunk_sizes =
          std::move(qual_chunks.chunk_sizes);
      result.uncompressed_quality_chunk_sizes.reserve(
          result.compressed_quality_chunk_sizes.size());
      for (size_t offset = 0; offset < stats.quality_scores_size;
           offset += field_slice_size) {
        result.uncompressed_quality_chunk_sizes.push_back(
            std::min(field_slice_size,
                     static_cast<size_t>(stats.quality_scores_size - offset)));
      }
    } else {
      const size_t quality_chunk_count =
          stats.quality_scores_size == 0
              ? 0
              : (stats.quality_scores_size + BSC_QUALITY_CHUNK_SIZE - 1) /
                    BSC_QUALITY_CHUNK_SIZE;
      const auto resolved_bsc =
          resolve_bsc_config(bsc_config, quality_chunk_count);
      std::cerr << "  BSC backend: " << bsc_backend_name(resolved_bsc.backend)
                << ", jobs: " << resolved_bsc.parallelism
                << ", chunks: " << quality_chunk_count << std::endl;
      auto qual_chunks = bsc_compress_device_chunked(
          fields.quality_scores, stats.quality_scores_size,
          BSC_QUALITY_CHUNK_SIZE, bsc_config, stream);
      result.quality_scores.payload = std::move(qual_chunks.data);
      result.compressed_quality_chunk_sizes =
          std::move(qual_chunks.compressed_chunk_sizes);
      result.uncompressed_quality_chunk_sizes =
          std::move(qual_chunks.uncompressed_chunk_sizes);
    }
    std::cerr << "  -> " << result.quality_scores.payload.size() << " bytes"
              << std::endl;
    quality_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     Clock::now() - quality_start)
                     .count();
    std::cerr << "  Time: " << quality_ms << " ms" << std::endl;

    std::cerr << "Compressing line lengths (" << stats.line_length_size
              << " bytes)..." << std::endl;
    const auto line_length_start = Clock::now();
    auto index_chunks = gpu_compress_device_chunked(
        reinterpret_cast<const uint8_t *>(d_line_lengths),
        stats.line_length_size, field_slice_size, chunk_size, stream);
    result.line_lengths.payload = std::move(index_chunks.data);
    result.compressed_line_length_chunk_sizes =
        std::move(index_chunks.chunk_sizes);
    std::cerr << "  -> " << result.line_lengths.payload.size() << " bytes"
              << std::endl;
    line_length_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         Clock::now() - line_length_start)
                         .count();
    std::cerr << "  Time: " << line_length_ms << " ms" << std::endl;

    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              Clock::now() - compression_start)
                              .count();
    if (bsc_config.stat_mode) {
      std::cerr << "Compression stage timings:" << std::endl;
      std::cerr << "  Field prep:      " << field_prep_ms << " ms" << std::endl;
      std::cerr << "  Identifiers:     " << identifier_ms << " ms" << std::endl;
      std::cerr << "  Basecalls:       " << basecall_ms << " ms" << std::endl;
      std::cerr << "  Quality scores:  " << quality_ms << " ms" << std::endl;
      std::cerr << "  Line lengths:    " << line_length_ms << " ms"
                << std::endl;
      std::cerr << "  Total compress:  " << total_ms << " ms" << std::endl;
    }

    if (!bsc_config.log_stat_path.empty()) {
      std::ofstream log(bsc_config.log_stat_path, std::ios::app);
      if (log.is_open()) {
        log << "Compressor:\n";
        log << "  Field prep:      " << field_prep_ms << " ms\n";
        log << "  Identifiers:     " << identifier_ms << " ms, "
            << stats.identifiers_size << " B\n";
        log << "  Basecalls:       " << basecall_ms << " ms, "
            << stats.basecalls_size << " B\n";
        log << "  Quality scores:  " << quality_ms << " ms, "
            << stats.quality_scores_size << " B\n";
        log << "  Line lengths:    " << line_length_ms << " ms, "
            << stats.line_length_size << " B\n";
        log << "  Total compress:  " << total_ms << " ms\n";
      }
    }
  } catch (...) {
    cuda_free_if_set(d_line_lengths);
    cuda_free_if_set(fields.identifiers);
    cuda_free_if_set(fields.basecalls);
    cuda_free_if_set(fields.quality_scores);
    cuda_free_if_set(d_quality_offsets);
    cuda_free_if_set(d_basecall_offsets);
    cuda_free_if_set(d_identifier_offsets);
    cuda_free_if_set(d_quality_lengths);
    cuda_free_if_set(d_basecall_lengths);
    cuda_free_if_set(d_identifier_lengths);
    cuda_free_if_set(d_line_offsets);
    cuda_free_if_set(d_raw_bytes);
    cudaStreamDestroy(stream);
    throw;
  }

  cuda_free_if_set(d_line_lengths);
  cuda_free_if_set(fields.identifiers);
  cuda_free_if_set(fields.basecalls);
  cuda_free_if_set(fields.quality_scores);
  cuda_free_if_set(d_quality_offsets);
  cuda_free_if_set(d_basecall_offsets);
  cuda_free_if_set(d_identifier_offsets);
  cuda_free_if_set(d_quality_lengths);
  cuda_free_if_set(d_basecall_lengths);
  cuda_free_if_set(d_identifier_lengths);
  cuda_free_if_set(d_line_offsets);
  cuda_free_if_set(d_raw_bytes);
  cudaStreamDestroy(stream);
  return result;
}

FastqData decompress_fastq(const CompressedFastqData &compressed,
                           const BscConfig &bsc_config) {
  using Clock = std::chrono::high_resolution_clock;
  const auto decompress_start = Clock::now();

  std::cerr << "Decompressing identifiers..." << std::endl;
  const auto identifier_start = Clock::now();
  const auto identifiers =
      decompress_identifiers(compressed.identifiers, compressed.num_records);
  const auto identifier_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                            identifier_start)
          .count();
  std::cerr << "  -> " << identifiers.size() << " bytes" << std::endl;

  std::cerr << "Decompressing basecalls..." << std::endl;
  const auto basecall_start = Clock::now();
  const auto n_count_bytes =
      nvcomp_zstd_decompress(compressed.basecalls.n_counts);
  std::vector<uint8_t> packed_bases;
  if (compressed.basecalls.packed_codec == BasecallPackedCodec::Bsc) {
    packed_bases = bsc_decompress_chunked(
        compressed.basecalls.packed_bases.payload,
        compressed.basecalls.compressed_packed_chunk_sizes,
        compressed.basecalls.uncompressed_packed_chunk_sizes,
        compressed.basecalls.packed_bases.original_size, bsc_config);
  } else {
    packed_bases = gpu_decompress_chunked(
        compressed.basecalls.packed_bases.payload,
        compressed.basecalls.compressed_packed_chunk_sizes,
        compressed.basecalls.packed_bases.original_size);
  }
  const auto n_position_bytes = gpu_decompress_chunked(
      compressed.basecalls.n_positions.payload,
      compressed.basecalls.compressed_n_position_chunk_sizes,
      compressed.basecalls.n_positions.original_size);
  const auto basecalls = decode_basecalls(compressed.basecalls, n_count_bytes,
                                          packed_bases, n_position_bytes);
  const auto basecall_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                            basecall_start)
          .count();
  std::cerr << "  -> " << basecalls.size() << " bytes" << std::endl;

  std::cerr << "Decompressing quality scores..." << std::endl;
  const auto quality_start = Clock::now();
  std::cerr << "  Codec: " << quality_codec_name(compressed.quality_codec)
            << std::endl;
  std::vector<uint8_t> quality_scores;
  if (compressed.quality_codec == QualityCodec::Zstd) {
    quality_scores =
        gpu_decompress_chunked(compressed.quality_scores.payload,
                               compressed.compressed_quality_chunk_sizes,
                               compressed.quality_scores.original_size);
  } else {
    const auto resolved_bsc = resolve_bsc_config(
        bsc_config, compressed.compressed_quality_chunk_sizes.size());
    std::cerr << "  BSC backend: " << bsc_backend_name(resolved_bsc.backend)
              << ", jobs: " << resolved_bsc.parallelism << ", chunks: "
              << compressed.compressed_quality_chunk_sizes.size() << std::endl;
    quality_scores = bsc_decompress_chunked(
        compressed.quality_scores.payload,
        compressed.compressed_quality_chunk_sizes,
        compressed.uncompressed_quality_chunk_sizes,
        compressed.quality_scores.original_size, bsc_config);
  }
  std::vector<uint8_t> reordered_quality_scores;
  const std::vector<uint8_t> *quality_scores_for_rebuild = &quality_scores;
  if (compressed.quality_layout == QualityLayoutKind::FixedLength &&
      compressed.fixed_quality_length != 0) {
    if (compressed.quality_transposed) {
      reordered_quality_scores = inverse_transpose_fixed_length_quality_scores(
          quality_scores, compressed.num_records,
          compressed.fixed_quality_length);
      quality_scores_for_rebuild = &reordered_quality_scores;
      std::cerr << "  Layout: column-major fixed-length (transposed, "
                << compressed.fixed_quality_length << " bases)" << std::endl;
    } else {
      std::cerr << "  Layout: row-major fixed-length (column-major disabled, "
                << compressed.fixed_quality_length << " bases)" << std::endl;
    }
  } else {
    std::cerr << "  Layout: row-major variable-length" << std::endl;
  }
  const auto quality_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              Clock::now() - quality_start)
                              .count();
  std::cerr << "  -> " << quality_scores.size() << " bytes" << std::endl;

  std::cerr << "Decompressing line lengths..." << std::endl;
  const auto line_length_start = Clock::now();
  const auto line_offset_bytes =
      gpu_decompress_chunked(compressed.line_lengths.payload,
                             compressed.compressed_line_length_chunk_sizes,
                             compressed.line_lengths.original_size);
  const auto line_length_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                            line_length_start)
          .count();
  std::cerr << "  -> " << line_offset_bytes.size() << " bytes" << std::endl;

  if (compressed.line_offset_count == 0) {
    throw std::runtime_error("Decoded line-offset count is invalid");
  }
  if (line_offset_bytes.size() !=
      compressed.line_offset_count * sizeof(uint32_t)) {
    throw std::runtime_error(
        "Decoded line-length payload has an unexpected size");
  }

  std::vector<uint64_t> line_offsets(compressed.line_offset_count);
  uint32_t *d_line_lengths = nullptr;
  uint64_t *d_line_offsets = nullptr;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  try {
    CUDA_CHECK(cudaMalloc(&d_line_lengths, line_offset_bytes.size()));
    CUDA_CHECK(
        cudaMalloc(&d_line_offsets, line_offsets.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_line_lengths, line_offset_bytes.data(),
                               line_offset_bytes.size(), cudaMemcpyHostToDevice,
                               stream));
    delta_decode_lengths_to_offsets(d_line_lengths, d_line_offsets,
                                    line_offsets.size(), stream);
    CUDA_CHECK(cudaMemcpyAsync(line_offsets.data(), d_line_offsets,
                               line_offsets.size() * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (...) {
    cuda_free_if_set(d_line_offsets);
    cuda_free_if_set(d_line_lengths);
    cudaStreamDestroy(stream);
    throw;
  }

  cuda_free_if_set(d_line_offsets);
  cuda_free_if_set(d_line_lengths);
  cudaStreamDestroy(stream);

  std::cerr << "Rebuilding FASTQ..." << std::endl;
  const auto rebuild_start = Clock::now();
  auto result =
      rebuild_fastq(line_offsets, identifiers, basecalls,
                    *quality_scores_for_rebuild, compressed.num_records);
  const auto rebuild_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              Clock::now() - rebuild_start)
                              .count();

  const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            Clock::now() - decompress_start)
                            .count();
  if (bsc_config.stat_mode) {
    std::cerr << "Decompression stage timings:" << std::endl;
    std::cerr << "  Identifiers:     " << identifier_ms << " ms" << std::endl;
    std::cerr << "  Basecalls:       " << basecall_ms << " ms" << std::endl;
    std::cerr << "  Quality scores:  " << quality_ms << " ms" << std::endl;
    std::cerr << "  Line lengths:    " << line_length_ms << " ms" << std::endl;
    std::cerr << "  Rebuild:         " << rebuild_ms << " ms" << std::endl;
    std::cerr << "  Total decompress:" << total_ms << " ms" << std::endl;
  }

  if (!bsc_config.log_stat_path.empty()) {
    std::ofstream log(bsc_config.log_stat_path, std::ios::app);
    if (log.is_open()) {
      log << "Decompressor:\n";
      log << "  Identifiers:     " << identifier_ms << " ms, "
          << identifiers.size() << " B\n";
      log << "  Basecalls:       " << basecall_ms << " ms, " << basecalls.size()
          << " B\n";
      log << "  Quality scores:  " << quality_ms << " ms, "
          << quality_scores.size() << " B\n";
      log << "  Line lengths:    " << line_length_ms << " ms, "
          << line_offset_bytes.size() << " B\n";
      log << "  Rebuild:         " << rebuild_ms << " ms\n";
      log << "  Total decompress:" << total_ms << " ms\n";
    }
  }

  return result;
}

} // namespace gpufastq
