#include "bharatopt/mps.hpp"
#include "bharatopt/interior_point.hpp"
#include "bharatopt/qp.hpp"
#include "bharatopt/solver.hpp"
#include "bharatopt/cuda_backend.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

static std::string js(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '\\' || c == '"') o += '\\';
        o += c;
    }
    o += '"';
    return o;
}

static void jnum(std::ostream& o, double v) {
    if (std::isfinite(v)) o << v;
    else o << "null";
}

static std::string integration_status(const bharatopt::SolverResult& r) {
    if (r.status.find("INFEASIBLE") != std::string::npos ||
        r.status == "MIP_NO_FEASIBLE_INCUMBENT") {
        return "infeasible";
    }
    if (r.converged) return "optimal";
    if (!r.x.empty()) return "feasible";
    return "unknown";
}

static double integration_gap(const bharatopt::SolverResult& r) {
    if (std::isfinite(r.mip_gap) && r.mip_gap >= 0.0) return r.mip_gap;
    if (std::isfinite(r.objective) && std::isfinite(r.best_bound)) {
        return std::abs(r.objective - r.best_bound) /
               (1.0 + std::abs(r.objective));
    }
    return 0.0;
}

static void write_contract_json(
    const std::string& path,
    const bharatopt::SolverResult& r) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path);

    out << "{";
    out << "\"status\":" << js(integration_status(r));
    out << ",\"objective\":"; jnum(out, r.objective);
    out << ",\"certified_lower_bound\":"; jnum(out, r.best_bound);
    out << ",\"relative_gap\":"; jnum(out, integration_gap(r));
    out << ",\"farkas_multipliers\":[]";
    out << "}\n";

    if (!out) throw std::runtime_error("Failed writing output file: " + path);
}

int main(int argc, char** argv) {
    bool demo = false, mip_demo = false, qp_demo = false, ip = false;
    bool use_cuda = false, json = false, lp_only = false;
    std::string mps, output_path, mode;

    bharatopt::SolverOptions opt;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "solve") {
            continue;
        } else if (a == "--demo") {
            demo = true;
        } else if (a == "--mip-demo") {
            mip_demo = true;
        } else if (a == "--qp-demo") {
            qp_demo = true;
        } else if (a == "--ip") {
            ip = true;
        } else if (a == "--cuda") {
            use_cuda = true;
        } else if (a == "--json") {
            json = true;
        } else if (a == "--lp-only") {
            lp_only = true;
        } else if ((a == "--mps" || a == "--input") && i + 1 < argc) {
            mps = argv[++i];
        } else if (a == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (a == "--mode" && i + 1 < argc) {
            mode = argv[++i];
            if (mode != "certified") {
                std::cerr << "Unsupported mode: " << mode << "\n";
                return 2;
            }
            json = true;
        } else if (a == "--max-iters" && i + 1 < argc) {
            opt.max_iterations = std::stoi(argv[++i]);
        } else if (a == "--tol" && i + 1 < argc) {
            opt.tolerance = std::stod(argv[++i]);
        } else if (a == "--max-nodes" && i + 1 < argc) {
            opt.max_nodes = std::stoi(argv[++i]);
        } else if (a == "--mip-gap" && i + 1 < argc) {
            opt.mip_gap = std::stod(argv[++i]);
        } else if (a == "--time-limit" && i + 1 < argc) {
            opt.time_limit_sec = std::stod(argv[++i]);
        } else if (a == "--no-presolve") {
            opt.presolve = false;
        } else if (a == "--scaling-passes" && i + 1 < argc) {
            opt.scaling_passes = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return 2;
        }
    }

    try {
        bharatopt::BharatOptSolverCore solver;
        opt.use_cuda = use_cuda;

        if (qp_demo) {
            auto q = bharatopt::make_qp_demo();
            auto r = solver.solve_qp(q, opt);

            if (!output_path.empty()) {
                write_contract_json(output_path, r);
            } else if (json) {
                std::cout << "{\"status\":" << js(integration_status(r))
                          << ",\"objective\":"; jnum(std::cout, r.objective);
                std::cout << ",\"certified_lower_bound\":"; jnum(std::cout, r.best_bound);
                std::cout << ",\"relative_gap\":"; jnum(std::cout, integration_gap(r));
                std::cout << ",\"farkas_multipliers\":[]}\n";
            } else {
                std::cout << "BharatOpt | model=" << q.name
                          << " rows=" << q.linear.A.rows
                          << " cols=" << q.linear.A.cols
                          << " nnz=" << q.linear.A.values.size() << "\n";
                std::cout << "status=" << r.status
                          << " converged=" << r.converged
                          << " iterations=" << r.iterations
                          << " objective=" << r.objective
                          << " primal_residual=" << r.primal_residual
                          << " dual_residual=" << r.dual_residual
                          << " time_sec=" << r.solve_time_sec
                          << " backend=" << r.backend << "\n";
            }
            return r.converged ? 0 : 3;
        }

        bharatopt::LPModel m;
        if (demo) {
            m = bharatopt::make_refinery_demo();
        } else if (mip_demo) {
            m = bharatopt::make_milp_demo();
        } else if (!mps.empty()) {
            m = bharatopt::parse_mps(mps);
        } else {
            std::cerr << "Use --demo, --mip-demo, --qp-demo, --ip or --input file.mps\n";
            return 2;
        }

        auto r = ip
            ? bharatopt::solve_interior_point(
                  m, bharatopt::InteriorPointOptions{
                         opt.max_iterations, opt.tolerance, 400, 400})
            : (lp_only ? solver.solve_lp(m, opt) : solver.solve(m, opt));

        if (!output_path.empty()) {
            write_contract_json(output_path, r);
        } else if (json) {
            std::cout << "{\"status\":" << js(integration_status(r))
                      << ",\"objective\":"; jnum(std::cout, r.objective);
            std::cout << ",\"certified_lower_bound\":"; jnum(std::cout, r.best_bound);
            std::cout << ",\"relative_gap\":"; jnum(std::cout, integration_gap(r));
            std::cout << ",\"farkas_multipliers\":[]}\n";
        } else {
            std::cout << "BharatOpt | model=" << m.name
                      << " rows=" << m.A.rows
                      << " cols=" << m.A.cols
                      << " nnz=" << m.A.values.size() << "\n";
            if (use_cuda)
                std::cout << "CUDA device: "
                          << bharatopt::cuda_backend::cuda_device_name() << "\n";
            std::cout << "status=" << r.status
                      << " converged=" << r.converged
                      << " iterations=" << r.iterations
                      << " nodes=" << r.nodes << "\n";
            std::cout << "objective=" << r.objective
                      << " best_bound=" << r.best_bound
                      << " mip_gap=" << r.mip_gap
                      << " primal_residual=" << r.primal_residual
                      << " dual_residual=" << r.dual_residual
                      << " time_sec=" << r.solve_time_sec
                      << " backend=" << r.backend << "\n";
        }

        return r.converged ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
