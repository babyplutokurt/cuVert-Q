#pragma once

#include "fastq_record.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpufastq {

struct QualityLengthAnalysis {
  std::vector<uint32_t> lengths;
  QualityLayoutKind layout = QualityLayoutKind::FixedLength;
  uint32_t fixed_length = 0;
};

// Build FASTQ line-start offsets from the raw byte stream on GPU.
// The returned vector always starts with 0 and ends with raw_bytes.size().
std::vector<uint64_t>
build_line_offsets_gpu(const std::vector<uint8_t> &raw_bytes,
                       cudaStream_t stream = 0);

// Delta-encode a monotonic offset vector into same-length uint32 deltas.
// The first output element is copied from the first offset, which is expected
// to be 0 for FASTQ line offsets.
void delta_encode_offsets_to_lengths(const uint64_t *d_offsets,
                                     uint32_t *d_lengths, size_t count,
                                     cudaStream_t stream = 0);

// Decode same-length uint32 deltas back into uint64 offsets via inclusive sum.
void delta_decode_lengths_to_offsets(const uint32_t *d_lengths,
                                     uint64_t *d_offsets, size_t count,
                                     cudaStream_t stream = 0);

// Compute per-record quality lengths from FASTQ line offsets and classify
// the file as fixed-length or variable-length on GPU.
QualityLengthAnalysis
analyze_quality_lengths(const std::vector<uint64_t> &line_offsets,
                        uint64_t num_records, cudaStream_t stream = 0);

} // namespace gpufastq
