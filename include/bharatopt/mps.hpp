#pragma once
#include <string>
#include "bharatopt/model.hpp"
namespace bharatopt {
LPModel parse_mps(const std::string& path);
LPModel make_refinery_demo();
}