#pragma once

#include "codec_bsc.hpp"

#include <string>

namespace gpufastq::workflow {

int compress(const std::string &input_path, const std::string &output_path,
             const BscConfig &bsc_config = {});
int roundtrip(const std::string &input_path, const BscConfig &bsc_config = {});

} // namespace gpufastq::workflow
