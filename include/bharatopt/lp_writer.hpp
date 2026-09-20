#pragma once

#include "bharatopt/model.hpp"
#include <string>

namespace bharatopt {

void write_lp_format(const LPModel& model, const std::string& path);
std::string to_lp_string(const LPModel& model);

} // namespace bharatopt
