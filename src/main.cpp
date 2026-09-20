#include "bharatopt/mps.hpp"
#include "bharatopt/pooling.hpp"
#include "bharatopt/refinery_model.hpp"
#include "bharatopt/interior_point.hpp"
#include "bharatopt/qp.hpp"
#include "bharatopt/solver.hpp"
#include "bharatopt/cuda_backend.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <limits>

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

static void write_refinery_json(
    const std::string& path,
    const bharatopt::RefineryModel& refinery,
    const bharatopt::GlobalPoolingResult& global) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path);

    const double objective = global.objective;
    const double bound = global.global_bound;
    const double gap = global.optimality_gap;
    const bool audited = global.feasible && !global.x.empty() &&
        std::isfinite(objective) &&
        bharatopt::pooling_max_constraint_violation(refinery.pooling, global.x) <= 1e-6;

    out << "{";
    out << "\"status\":" << js(global.status);
    out << ",\"model\":" << js(refinery.pooling.name);
    out << ",\"objective\":"; jnum(out, objective);
    out << ",\"global_upper_bound\":"; jnum(out, bound);
    out << ",\"mccormick_global_upper_bound\":"; jnum(out, bound);
    out << ",\"nonlinear_gap\":"; jnum(out, gap);
    out << ",\"global_optimality_certified\":" << (global.certified ? "true" : "false");
    out << ",\"constraint_audit_passed\":" << (audited ? "true" : "false");
    out << ",\"constraint_max_violation\":";
    if (global.x.empty()) out << "null";
    else jnum(out, bharatopt::pooling_max_constraint_violation(refinery.pooling, global.x));
    out << ",\"nodes_explored\":" << global.nodes_explored;
    out << ",\"nodes_pruned\":" << global.nodes_pruned;
    out << ",\"slp_iterations\":" << global.incumbent.iterations;
    out << ",\"slp_accepted_steps\":" << global.incumbent.accepted_steps;
    out << ",\"slp_rejected_steps\":" << global.incumbent.rejected_steps;
    out << ",\"variables\":[";
    for (std::size_t i = 0; i < refinery.pooling.linear.var_names.size(); ++i) {
        if (i) out << ",";
        out << "{\"name\":" << js(refinery.pooling.linear.var_names[i])
            << ",\"value\":";
        if (i < global.x.size()) jnum(out, global.x[i]); else out << "null";
        out << "}";
    }
    out << "]}\n";
    if (!out) throw std::runtime_error("Failed writing refinery output: " + path);
}

static void write_pooling_json(
    const std::string& path,
    const bharatopt::PoolingSLPResult& slp,
    const bharatopt::McCormickRelaxation& mc) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path);

    const double bound = mc.solve_result.objective;
    double gap = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(bound) && std::isfinite(slp.objective)) {
        if (slp.objective >= 0.0) {
            gap = std::max(0.0, bound - slp.objective) /
                  (1.0 + std::abs(slp.objective));
        } else {
            gap = std::max(0.0, slp.objective - bound) /
                  (1.0 + std::abs(slp.objective));
        }
    }

    out << "{";
    out << "\"status\":" << js(slp.status);
    out << ",\"slp_objective\":"; jnum(out, slp.objective);
    out << ",\"mccormick_bound\":"; jnum(out, bound);
    out << ",\"nonlinear_gap\":"; jnum(out, gap);
    out << ",\"slp_iterations\":" << slp.iterations;
    out << ",\"accepted_steps\":" << slp.accepted_steps;
    out << ",\"rejected_steps\":" << slp.rejected_steps;
    out << ",\"trust_radius\":"; jnum(out, slp.trust_radius);
    out << ",\"improvement_ratio\":"; jnum(out, slp.improvement_ratio);
    out << ",\"mccormick_status\":" << js(mc.solve_result.status);
    out << ",\"mccormick_converged\":" << (mc.solve_result.converged ? "true" : "false");
    out << "}\n";

    if (!out) throw std::runtime_error("Failed writing pooling output: " + path);
}

static bharatopt::PoolingModel make_mrpl_pooling_model() {
    bharatopt::PoolingModel model;
    model.name = "MRPL-pooling-integration-model";
    model.linear.name = model.name;
    model.linear.var_names = {"light_crude", "heavy_crude"};
    model.linear.objective = {100.0, 80.0};
    model.linear.lower = {2.0, 2.0};
    model.linear.upper = {10.0, 10.0};
    model.linear.integer = {0, 0};
    model.linear.maximize = true;
    model.linear.rows = {
        {"throughput", bharatopt::RowSense::LessEqual, 12.0}
    };
    model.linear.A.rows = 1;
    model.linear.A.cols = 2;
    model.linear.A.row_ptr = {0, 2};
    model.linear.A.col_index = {0, 1};
    model.linear.A.values = {1.0, 1.0};
    model.linear.row_lower = {-bharatopt::INF};
    model.linear.row_upper = {12.0};
    model.objective_bilinear.push_back({0, 1, 5.0});
    model.constraint_bilinear.resize(1);
    return model;
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
    bool demo = false, mip_demo = false, qp_demo = false, ip = false, pooling_demo = false, refinery_json = false;
    bool use_cuda = false, json = false, lp_only = false;
    std::string mps, refinery_json_path, output_path, mode;
    std::size_t global_nodes = 200;
    double global_gap = 1e-5;
    double global_time_limit = 30.0;

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
        } else if (a == "--pooling-demo") {
            pooling_demo = true;
        } else if (a == "--refinery-json" && i + 1 < argc) {
            refinery_json = true;
            refinery_json_path = argv[++i];
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
        } else if (a == "--global-nodes" && i + 1 < argc) {
            global_nodes = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (a == "--global-gap" && i + 1 < argc) {
            global_gap = std::stod(argv[++i]);
        } else if (a == "--global-time-limit" && i + 1 < argc) {
            global_time_limit = std::stod(argv[++i]);
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

        if (refinery_json || (!mps.empty() && std::filesystem::path(mps).extension() == ".json")) {
            const std::string path = refinery_json ? refinery_json_path : mps;
            auto refinery = bharatopt::parse_refinery_json(path);
            bharatopt::validate_refinery_model(refinery);

            bharatopt::GlobalPoolingOptions global_options;
            global_options.slp_options.lp_options = opt;
            global_options.slp_options.max_iterations = 30;
            global_options.slp_options.initial_trust_radius = 1.0;
            global_options.slp_options.minimum_trust_radius = 1e-8;
            global_options.slp_options.maximum_trust_radius = 8.0;
            global_options.relaxation_options = opt;
            global_options.max_nodes = global_nodes;
            global_options.time_limit_sec = std::max(1.0, opt.time_limit_sec > 0.0 ? opt.time_limit_sec : global_time_limit);
            global_options.absolute_gap = global_gap;
            global_options.relative_gap = global_gap;

            auto global = bharatopt::solve_global_pooling(
                refinery.pooling, solver, global_options);

            if (!output_path.empty()) {
                write_refinery_json(output_path, refinery, global);
            } else {
                write_refinery_json("/tmp/bharatopt_refinery_result.json", refinery, global);
                std::cout << "BharatOpt | model=" << refinery.pooling.name << "\n";
                std::cout << "status=" << global.status
                          << " objective=" << global.objective
                          << " global_upper_bound=" << global.global_bound
                          << " gap=" << global.optimality_gap
                          << " nodes=" << global.nodes_explored
                          << "\n";
            }
            return global.feasible ? 0 : 3;
        }

        if (pooling_demo) {
            auto model = make_mrpl_pooling_model();

            bharatopt::PoolingSLPOptions pooling_options;
            pooling_options.lp_options = opt;
            pooling_options.max_iterations = 20;
            pooling_options.initial_trust_radius = 1.0;
            pooling_options.minimum_trust_radius = 1e-8;
            pooling_options.maximum_trust_radius = 8.0;

            auto slp = bharatopt::solve_pooling_slp(
                model, solver, pooling_options);
            auto mc = bharatopt::solve_mccormick_relaxation(
                model, solver, opt);

            if (!output_path.empty()) {
                write_pooling_json(output_path, slp, mc);
            } else if (json) {
                const double bound = mc.solve_result.objective;
                double gap = std::numeric_limits<double>::quiet_NaN();
                if (std::isfinite(bound) && std::isfinite(slp.objective)) {
                    gap = std::max(0.0, bound - slp.objective) /
                          (1.0 + std::abs(slp.objective));
                }
                std::cout << "{\"status\":" << js(slp.status)
                          << ",\"slp_objective\":"; jnum(std::cout, slp.objective);
                std::cout << ",\"mccormick_bound\":"; jnum(std::cout, bound);
                std::cout << ",\"nonlinear_gap\":"; jnum(std::cout, gap);
                std::cout << ",\"slp_iterations\":" << slp.iterations;
                std::cout << ",\"accepted_steps\":" << slp.accepted_steps;
                std::cout << ",\"rejected_steps\":" << slp.rejected_steps;
                std::cout << ",\"trust_radius\":"; jnum(std::cout, slp.trust_radius);
                std::cout << ",\"improvement_ratio\":"; jnum(std::cout, slp.improvement_ratio);
                std::cout << ",\"mccormick_status\":" << js(mc.solve_result.status);
                std::cout << ",\"mccormick_converged\":"
                          << (mc.solve_result.converged ? "true" : "false")
                          << "}\n";
            } else {
                std::cout << "BharatOpt | model=" << model.name << "\n";
                std::cout << "SLP status=" << slp.status
                          << " objective=" << slp.objective
                          << " iterations=" << slp.iterations
                          << " accepted=" << slp.accepted_steps
                          << " rejected=" << slp.rejected_steps
                          << " trust_radius=" << slp.trust_radius << "\n";
                std::cout << "McCormick status=" << mc.solve_result.status
                          << " bound=" << mc.solve_result.objective
                          << " converged=" << mc.solve_result.converged << "\n";
            }

            return (slp.feasible && std::isfinite(slp.objective) &&
                    std::isfinite(mc.solve_result.objective)) ? 0 : 3;
        }

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
            std::cerr << "Use --demo, --mip-demo, --qp-demo, --pooling-demo, --refinery-json file.json, --global-nodes N, --global-gap G, --global-time-limit S, --ip or --input file.mps\n";
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
