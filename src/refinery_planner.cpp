// src/refinery_planner.cpp
//
// Implements everything declared in refinery_planner.hpp. The validate()
// path is fully self-contained and needs nothing from your solver — it is
// safe to compile and unit-test this file today, before PoolingModel exists
// or is wired in. The build() path at the bottom is written against the
// ASSUMED PoolingModel API documented in the header; adjust method names
// there to match your real class, everything else in this file is solver-
// independent.
//
// Dependency: nlohmann/json (single header, MIT licensed) for JSON parsing.
// If your project uses a different JSON library, only load_json_file() and
// the parse_* helpers below need to change — the domain logic does not.

#include "bharatopt/refinery_planner.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace bharatopt::refinery {

using json = nlohmann::json;

namespace {

RefineryModel parse_model(const json& j) {
    RefineryModel m;
    m.name = j.value("name", std::string("unnamed"));
    m.synthetic = j.value("synthetic", false);

    for (const auto& f : j.at("feedstocks")) {
        Feedstock fs;
        fs.name = f.at("name").get<std::string>();
        fs.availability = f.value("availability", 0.0);
        fs.cost_per_bbl = f.value("cost_per_bbl", 0.0);
        if (f.contains("properties"))
            for (auto& [k, v] : f.at("properties").items()) fs.properties[k] = v.get<double>();
        m.feedstocks.push_back(std::move(fs));
    }

    for (const auto& u : j.at("units")) {
        Unit un;
        un.name = u.at("name").get<std::string>();
        un.type = u.value("type", std::string(""));
        un.capacity_min = u.value("capacity_min", 0.0);
        un.capacity_max = u.value("capacity_max", 0.0);
        un.energy_cost_per_bbl = u.value("energy_cost_per_bbl", 0.0);
        if (u.contains("feed_streams"))
            for (const auto& s : u.at("feed_streams")) un.feed_streams.push_back(s.get<std::string>());
        if (u.contains("yields"))
            for (auto& [feed, outs] : u.at("yields").items())
                for (auto& [out_name, frac] : outs.items())
                    un.yields[feed][out_name] = frac.get<double>();
        if (u.contains("output_properties"))
            for (auto& [stream, props] : u.at("output_properties").items())
                for (auto& [pname, pval] : props.items())
                    un.output_properties[stream][pname] = pval.get<double>();
        m.units.push_back(std::move(un));
    }

    for (const auto& s : j.at("streams")) {
        Stream st;
        st.name = s.at("name").get<std::string>();
        st.source = s.value("source", std::string(""));
        if (s.contains("properties"))
            for (auto& [k, v] : s.at("properties").items()) st.properties[k] = v.get<double>();
        m.streams.push_back(std::move(st));
    }

    for (const auto& p : j.at("pools")) {
        Pool pl;
        pl.name = p.at("name").get<std::string>();
        pl.blended_property = p.value("blended_property", std::string(""));
        pl.bilinear = p.value("bilinear", true);
        if (p.contains("inlet_streams"))
            for (const auto& s : p.at("inlet_streams")) pl.inlet_streams.push_back(s.get<std::string>());
        m.pools.push_back(std::move(pl));
    }

    for (const auto& pr : j.at("products")) {
        ProductSpec ps;
        ps.name = pr.at("name").get<std::string>();
        ps.from_pool = pr.value("from_pool", std::string(""));
        ps.from_stream = pr.value("from_stream", std::string(""));
        ps.minimum_production = pr.value("minimum_production", 0.0);
        ps.maximum_production = pr.value("maximum_production", 0.0);
        ps.price_per_bbl = pr.value("price_per_bbl", 0.0);
        if (pr.contains("specifications"))
            for (auto& [k, v] : pr.at("specifications").items()) ps.specifications[k] = v.get<double>();
        m.products.push_back(std::move(ps));
    }

    return m;
}

}  // namespace

RefineryPlanner::RefineryPlanner(const std::string& json_path) {
    std::ifstream in(json_path);
    if (!in) throw std::runtime_error("refinery_planner: cannot open " + json_path);
    json j;
    try {
        in >> j;
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("refinery_planner: malformed JSON: ") + e.what());
    }
    for (const char* required : {"feedstocks", "units", "streams", "pools", "products"}) {
        if (!j.contains(required))
            throw std::runtime_error(std::string("refinery_planner: missing required key '") + required + "'");
    }
    model_ = parse_model(j);
}

// ---------------------------------------------------------------- validate

ValidationResult RefineryPlanner::validate() const {
    ValidationResult r;
    check_duplicate_names(r);
    check_unknown_references(r);
    check_bounds(r);
    check_yield_sums(r);
    check_graph_cycles(r);
    check_orphans(r);
    return r;
}

void RefineryPlanner::check_duplicate_names(ValidationResult& r) const {
    auto check = [&](const char* kind, const std::vector<std::string>& names) {
        std::unordered_set<std::string> seen;
        for (const auto& n : names) {
            if (!seen.insert(n).second) {
                r.add(ValidationIssue::Severity::Error, "DUPLICATE_NAME",
                      std::string(kind) + " name used more than once: " + n);
            }
        }
    };
    std::vector<std::string> feed_names, unit_names, stream_names, pool_names, product_names;
    for (auto& f : model_.feedstocks) feed_names.push_back(f.name);
    for (auto& u : model_.units) unit_names.push_back(u.name);
    for (auto& s : model_.streams) stream_names.push_back(s.name);
    for (auto& p : model_.pools) pool_names.push_back(p.name);
    for (auto& p : model_.products) product_names.push_back(p.name);
    check("feedstock", feed_names);
    check("unit", unit_names);
    check("stream", stream_names);
    check("pool", pool_names);
    check("product", product_names);
}

void RefineryPlanner::check_unknown_references(ValidationResult& r) const {
    std::unordered_set<std::string> feed_names, stream_names, pool_names;
    for (auto& f : model_.feedstocks) feed_names.insert(f.name);
    for (auto& s : model_.streams) stream_names.insert(s.name);
    for (auto& p : model_.pools) pool_names.insert(p.name);

    for (auto& u : model_.units) {
        for (auto& feed : u.feed_streams) {
            if (!feed_names.count(feed) && !stream_names.count(feed)) {
                r.add(ValidationIssue::Severity::Error, "UNKNOWN_FEED_REF",
                      "Unit '" + u.name + "' references unknown feed/stream '" + feed + "'");
            }
        }
        for (auto& [feed, outs] : u.yields) {
            if (!feed_names.count(feed) && !stream_names.count(feed)) {
                r.add(ValidationIssue::Severity::Error, "UNKNOWN_YIELD_FEED",
                      "Unit '" + u.name + "' has a yields entry for unknown feed '" + feed + "'");
            }
            for (auto& [out_name, frac] : outs) {
                if (!stream_names.count(out_name)) {
                    r.add(ValidationIssue::Severity::Error, "UNKNOWN_YIELD_OUTPUT",
                          "Unit '" + u.name + "' yields to undeclared stream '" + out_name + "'");
                }
                if (frac < 0.0) {
                    r.add(ValidationIssue::Severity::Error, "NEGATIVE_YIELD",
                          "Unit '" + u.name + "' has a negative yield fraction for '" + out_name + "'");
                }
            }
        }
    }
    for (auto& p : model_.pools) {
        for (auto& s : p.inlet_streams) {
            if (!stream_names.count(s)) {
                r.add(ValidationIssue::Severity::Error, "UNKNOWN_POOL_INLET",
                      "Pool '" + p.name + "' references unknown stream '" + s + "'");
            }
        }
    }
    for (auto& pr : model_.products) {
        bool has_pool = !pr.from_pool.empty();
        bool has_stream = !pr.from_stream.empty();
        if (has_pool == has_stream) {  // both set, or neither set
            r.add(ValidationIssue::Severity::Error, "AMBIGUOUS_PRODUCT_SOURCE",
                  "Product '" + pr.name + "' must set exactly one of from_pool / from_stream");
            continue;
        }
        if (has_pool && !pool_names.count(pr.from_pool)) {
            r.add(ValidationIssue::Severity::Error, "UNKNOWN_PRODUCT_POOL",
                  "Product '" + pr.name + "' references unknown pool '" + pr.from_pool + "'");
        }
        if (has_stream && !stream_names.count(pr.from_stream)) {
            r.add(ValidationIssue::Severity::Error, "UNKNOWN_PRODUCT_STREAM",
                  "Product '" + pr.name + "' references unknown stream '" + pr.from_stream + "'");
        }
    }
}

void RefineryPlanner::check_bounds(ValidationResult& r) const {
    for (auto& f : model_.feedstocks) {
        if (f.availability < 0.0)
            r.add(ValidationIssue::Severity::Error, "NEGATIVE_AVAILABILITY",
                  "Feedstock '" + f.name + "' has negative availability");
    }
    for (auto& u : model_.units) {
        if (u.capacity_min > u.capacity_max)
            r.add(ValidationIssue::Severity::Error, "IMPOSSIBLE_CAPACITY",
                  "Unit '" + u.name + "' has capacity_min > capacity_max");
        if (u.capacity_min < 0.0)
            r.add(ValidationIssue::Severity::Error, "NEGATIVE_CAPACITY",
                  "Unit '" + u.name + "' has a negative capacity_min");
    }
    for (auto& p : model_.products) {
        if (p.minimum_production > p.maximum_production)
            r.add(ValidationIssue::Severity::Error, "IMPOSSIBLE_PRODUCTION_BOUNDS",
                  "Product '" + p.name + "' has minimum_production > maximum_production");
    }
}

void RefineryPlanner::check_yield_sums(ValidationResult& r) const {
    for (auto& u : model_.units) {
        for (auto& [feed, outs] : u.yields) {
            double total = 0.0;
            for (auto& [out_name, frac] : outs) total += frac;
            if (total > 1.0 + 1e-9) {
                r.add(ValidationIssue::Severity::Error, "YIELD_SUM_EXCEEDS_ONE",
                      "Unit '" + u.name + "' yields for feed '" + feed + "' sum to " +
                          std::to_string(total) + " > 1.0");
            } else if (total < 1.0 - 1e-6) {
                // Not necessarily an error — an explicit loss/coke term may account for
                // the remainder — but worth a warning if nothing named "loss" absorbs it.
                bool has_loss_term = outs.count("Loss_" + u.name) > 0;
                for (auto& [out_name, frac] : outs) {
                    if (out_name.find("Loss") != std::string::npos) has_loss_term = true;
                }
                if (!has_loss_term) {
                    r.add(ValidationIssue::Severity::Warning, "YIELD_SUM_INCOMPLETE",
                          "Unit '" + u.name + "' yields for feed '" + feed + "' sum to only " +
                              std::to_string(total) + " with no loss/coke term — is this intentional?");
                }
            }
        }
    }
}

void RefineryPlanner::check_graph_cycles(ValidationResult& r) const {
    // Build a directed graph: feedstock/stream -> unit -> stream -> pool -> product.
    // A cycle here means a unit or pool transitively feeds itself, which breaks the
    // mass-balance DAG assumption. DFS with a recursion-stack set detects it.
    std::unordered_map<std::string, std::vector<std::string>> adj;
    for (auto& u : model_.units) {
        for (auto& feed : u.feed_streams) adj[feed].push_back(u.name);
        for (auto& [feed, outs] : u.yields)
            for (auto& [out_name, frac] : outs) adj[u.name].push_back(out_name);
    }
    for (auto& p : model_.pools) {
        for (auto& s : p.inlet_streams) adj[s].push_back(p.name);
    }
    for (auto& pr : model_.products) {
        if (!pr.from_pool.empty()) adj[pr.from_pool].push_back(pr.name);
        if (!pr.from_stream.empty()) adj[pr.from_stream].push_back(pr.name);
    }

    std::unordered_set<std::string> visited, on_stack;
    std::function<bool(const std::string&)> dfs = [&](const std::string& node) -> bool {
        visited.insert(node);
        on_stack.insert(node);
        for (auto& next : adj[node]) {
            if (on_stack.count(next)) return true;              // back edge -> cycle
            if (!visited.count(next) && dfs(next)) return true;
        }
        on_stack.erase(node);
        return false;
    };
    for (auto& [node, _] : adj) {
        if (!visited.count(node) && dfs(node)) {
            r.add(ValidationIssue::Severity::Error, "GRAPH_CYCLE",
                  "The feed/unit/stream/pool/product graph contains a cycle reachable from '" +
                      node + "' — check unit feed_streams and pool inlet_streams for a reference "
                             "that points back at its own output.");
            break;  // one report is enough to act on; re-run after fixing to find more
        }
    }
}

void RefineryPlanner::check_orphans(ValidationResult& r) const {
    std::unordered_set<std::string> consumed_streams;
    for (auto& u : model_.units)
        for (auto& feed : u.feed_streams) consumed_streams.insert(feed);
    for (auto& p : model_.pools)
        for (auto& s : p.inlet_streams) consumed_streams.insert(s);
    for (auto& pr : model_.products)
        if (!pr.from_stream.empty()) consumed_streams.insert(pr.from_stream);

    for (auto& s : model_.streams) {
        if (!consumed_streams.count(s.name)) {
            r.add(ValidationIssue::Severity::Warning, "ORPHAN_STREAM",
                  "Stream '" + s.name + "' is produced but never consumed by any unit, pool, or product");
        }
    }
    std::unordered_set<std::string> pools_used;
    for (auto& pr : model_.products)
        if (!pr.from_pool.empty()) pools_used.insert(pr.from_pool);
    for (auto& p : model_.pools) {
        if (!pools_used.count(p.name)) {
            r.add(ValidationIssue::Severity::Warning, "ORPHAN_POOL",
                  "Pool '" + p.name + "' is defined but no product draws from it");
        }
    }
}

std::string ValidationResult::to_json() const {
    json j;
    j["valid"] = valid;
    j["errors"] = json::array();
    j["warnings"] = json::array();
    for (auto& e : errors) j["errors"].push_back({{"code", e.code}, {"message", e.message}});
    for (auto& w : warnings) j["warnings"].push_back({{"code", w.code}, {"message", w.message}});
    return j.dump(2);
}

// ------------------------------------------------------------------ build
//
// Sketch only — uncomment and adjust once pooling_model.hpp's real method
// names are known. The translation itself (one variable per feed/unit-feed/
// stream/pool-inlet/product, one equality row per mass balance, one bilinear
// term per pool) does not depend on solver internals, only on method names.
//
// PoolingModel RefineryPlanner::build() const {
//     PoolingModel pm;
//     std::unordered_map<std::string, int> var;
//
//     for (auto& f : model_.feedstocks)
//         var[f.name] = pm.add_variable(f.name, 0.0, f.availability);
//
//     for (auto& u : model_.units) {
//         for (auto& [feed, outs] : u.yields) {
//             std::string feed_var_name = u.name + "::" + feed;
//             var[feed_var_name] = pm.add_variable(feed_var_name, 0.0, u.capacity_max);
//             for (auto& [out_name, frac] : outs) {
//                 std::string out_var_name = out_name;  // one var per stream, summed over sources
//                 if (!var.count(out_var_name))
//                     var[out_var_name] = pm.add_variable(out_var_name, 0.0, 1e12);
//                 // out_volume - frac * feed_volume == 0  (per contributing unit; summed if
//                 // multiple units feed the same stream — extend coeffs accordingly)
//                 pm.add_linear_constraint(u.name + "->" + out_name,
//                     {{var[out_var_name], 1.0}, {var[feed_var_name], -frac}}, '=', 0.0);
//             }
//         }
//         // capacity: sum of this unit's feed vars within [capacity_min, capacity_max]
//     }
//
//     for (auto& p : model_.pools) {
//         // one quality variable + one volume variable per inlet, bilinear term per inlet,
//         // pool balance: sum(inlet volumes) == pool volume,
//         // quality balance: sum(quality_i * volume_i) == blended_property * pool_volume
//         // (this is exactly the bilinear term add_bilinear_term exists for)
//     }
//
//     for (auto& pr : model_.products) {
//         // demand bounds via add_variable bounds; specifications via
//         // add_linear_constraint against the relevant pool's blended property variable
//     }
//
//     return pm;
// }

}  // namespace bharatopt::refinery
