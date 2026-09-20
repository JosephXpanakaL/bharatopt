#include "bharatopt/lp_writer.hpp"
#include <fstream>
#include <sstream>
#include <cmath>

namespace bharatopt {

std::string to_lp_string(const LPModel& m) {
    std::ostringstream ss;
    ss << (m.maximize ? "Maximize\n obj: " : "Minimize\n obj: ");

    bool first = true;
    for (std::size_t j = 0; j < m.var_names.size(); ++j) {
        double c = m.objective[j];
        if (std::abs(c) < 1e-12) continue;
        if (!first && c >= 0.0) ss << " + ";
        else if (c < 0.0) ss << " - ";
        if (std::abs(std::abs(c) - 1.0) > 1e-12) ss << std::abs(c) << " ";
        ss << m.var_names[j];
        first = false;
    }
    if (first) ss << "0";
    ss << "\n\nSubject To\n";

    for (std::size_t i = 0; i < m.A.rows; ++i) {
        std::string rname = i < m.rows.size() ? m.rows[i].name : ("c" + std::to_string(i));
        ss << " " << rname << ": ";
        bool rfirst = true;
        for (int k = m.A.row_ptr[i]; k < m.A.row_ptr[i + 1]; ++k) {
            int j = m.A.col_index[k];
            double a = m.A.values[k];
            if (std::abs(a) < 1e-12) continue;
            if (!rfirst && a >= 0.0) ss << " + ";
            else if (a < 0.0) ss << " - ";
            if (std::abs(std::abs(a) - 1.0) > 1e-12) ss << std::abs(a) << " ";
            ss << m.var_names[j];
            rfirst = false;
        }
        if (rfirst) ss << "0";

        if (std::isfinite(m.row_lower[i]) && std::isfinite(m.row_upper[i]) && std::abs(m.row_lower[i] - m.row_upper[i]) < 1e-12) {
            ss << " = " << m.row_lower[i] << "\n";
        } else if (std::isfinite(m.row_upper[i])) {
            ss << " <= " << m.row_upper[i] << "\n";
        } else if (std::isfinite(m.row_lower[i])) {
            ss << " >= " << m.row_lower[i] << "\n";
        } else {
            ss << " <= +inf\n";
        }
    }

    ss << "\nBounds\n";
    for (std::size_t j = 0; j < m.var_names.size(); ++j) {
        bool has_lo = std::isfinite(m.lower[j]);
        bool has_hi = std::isfinite(m.upper[j]);
        if (has_lo && has_hi) {
            ss << " " << m.lower[j] << " <= " << m.var_names[j] << " <= " << m.upper[j] << "\n";
        } else if (has_hi) {
            ss << " -inf <= " << m.var_names[j] << " <= " << m.upper[j] << "\n";
        } else if (has_lo && m.lower[j] != 0.0) {
            ss << " " << m.var_names[j] << " >= " << m.lower[j] << "\n";
        }
    }

    std::vector<std::string> binaries;
    for (std::size_t j = 0; j < m.var_names.size(); ++j) {
        if (j < m.integer.size() && m.integer[j] && m.lower[j] == 0.0 && m.upper[j] == 1.0) {
            binaries.push_back(m.var_names[j]);
        }
    }
    if (!binaries.empty()) {
        ss << "\nBinary\n";
        for (const auto& b : binaries) ss << " " << b << "\n";
    }

    ss << "End\n";
    return ss.str();
}

void write_lp_format(const LPModel& model, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open LP output file: " + path);
    out << to_lp_string(model);
}

} // namespace bharatopt
