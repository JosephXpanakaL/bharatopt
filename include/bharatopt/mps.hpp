#pragma once
#include <string>
#include "bharatopt/model.hpp"
namespace bharatopt {
LPModel parse_mps(const std::string& path);
void write_mps(const LPModel& model, const std::string& path);
LPModel make_refinery_demo();
LPModel make_milp_demo();
}