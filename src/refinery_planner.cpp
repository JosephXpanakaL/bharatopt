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

    if (j.contains("streams")) {
        for (const auto& s : j.at("streams")) {
            Stream st;
            st.name = s.at("name").get<std::string>();
            st.source = s.value("source", std::string(""));
            if (s.contains("properties"))
                for (auto& [k, v] : s.at("properties").items()) st.properties[k] = v.get<double>();
            m.streams.push_back(std::move(st));
        }
    }

    for (const auto& p : j.at("pools")) {
        Pool pl;
        pl.name = p.at("name").get<std::string>();
        pl.blended_property = p.value("blended_property", std::string(""));
        pl.bilinear = p.value("bilinear", true);
        if (p.contains("inlet_streams"))
            for (const auto& s : p.at("inlet_streams")) pl.inlet_streams.push_back(s.get<std::string>());
        else if (p.contains("inlets"))
            for (const auto& s : p.at("inlets")) pl.inlet_streams.push_back(s.get<std::string>());
        if (p.contains("tracked_properties"))
            for (const auto& s : p.at("tracked_properties")) pl.tracked_properties.push_back(s.get<std::string>());
        pl.inlets = pl.inlet_streams;
        if (pl.blended_property.empty() && !pl.tracked_properties.empty()) {
            pl.blended_property = pl.tracked_properties.front();
        }
        m.pools.push_back(std::move(pl));
    }

    for (const auto& pr : j.at("products")) {
        ProductSpec ps;
        ps.name = pr.at("name").get<std::string>();
        ps.from_pool = pr.value("from_pool", std::string(""));
        ps.from_stream = pr.value("from_stream", std::string(""));
        ps.pool = pr.value("pool", ps.from_pool);
        ps.minimum_production = pr.value("minimum_production", pr.value("demand_min", 0.0));
        ps.maximum_production = pr.value("maximum_production", pr.value("demand_max", 0.0));
        ps.demand_min = ps.minimum_production;
        ps.demand_max = ps.maximum_production;
        ps.price_per_bbl = pr.value("price_per_bbl", 0.0);
        if (pr.contains("specifications")) {
            for (auto& [k, v] : pr.at("specifications").items()) {
                PropertyLimit limit;
                if (v.is_object()) {
                    if (v.contains("min")) limit.min = v.at("min").get<double>();
                    if (v.contains("max")) limit.max = v.at("max").get<double>();
                } else {
                    const double value = v.get<double>();
                    limit.min = value;
                    limit.max = value;
                }
                ps.specifications[k] = limit;
            }
        }
        if (ps.from_pool.empty() && !ps.pool.empty()) ps.from_pool = ps.pool;
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
    for (const auto* required : {"feedstocks", "units", "pools", "products"}) {
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
        for (auto& [feed, outs] : u.yields) {
            for (auto& [out_name, frac] : outs) {
                stream_names.insert(out_name);
            }
        }
    }

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
bharatopt::PoolingModel RefineryPlanner::build() const {
    bharatopt::PoolingModel pm;
    pm.name = model_.name;
    pm.linear.name = model_.name;
    pm.linear.maximize = true;

    std::unordered_map<std::string, int> var_idx;
    auto add_var = [&](const std::string& name, double lb, double ub, double obj_coeff = 0.0) -> int {
        int idx = static_cast<int>(pm.linear.var_names.size());
        pm.linear.var_names.push_back(name);
        pm.linear.lower.push_back(lb);
        pm.linear.upper.push_back(ub);
        pm.linear.objective.push_back(obj_coeff);
        pm.linear.integer.push_back(0);
        var_idx[name] = idx;
        return idx;
    };

    // 1. Feedstock variables (cost is negative for maximization)
    for (const auto& f : model_.feedstocks) {
        add_var(f.name, 0.0, f.availability, -f.cost_per_bbl);
    }

    // 2. Unit feeds & intermediate stream variables
    for (const auto& u : model_.units) {
        for (const auto& [feed, outs] : u.yields) {
            std::string feed_var = u.name + "::" + feed;
            add_var(feed_var, 0.0, u.capacity_max, -u.energy_cost_per_bbl);
            for (const auto& [out_name, _] : outs) {
                if (!var_idx.count(out_name)) {
                    add_var(out_name, 0.0, 1e9, 0.0);
                }
            }
        }
    }

    // 3. Pool variables (volumes and tracked properties)
    for (const auto& p : model_.pools) {
        std::string pool_vol_name = "pool::" + p.name + "::vol";
        add_var(pool_vol_name, 0.0, 1e9, 0.0);
        const auto tracked = p.tracked_properties.empty() ? std::vector<std::string>{p.blended_property} : p.tracked_properties;
        for (const auto& prop : tracked) {
            if (prop.empty()) continue;
            std::string pool_prop_name = "pool::" + p.name + "::" + prop;
            add_var(pool_prop_name, 0.0, 1000.0, 0.0);
        }
    }

    // 4. Product variables (revenue is positive for maximization)
    for (const auto& pr : model_.products) {
        std::string pr_name = "product::" + pr.name;
        const double lower = pr.demand_min > 0.0 ? pr.demand_min : pr.minimum_production;
        const double upper = pr.demand_max > 0.0 ? pr.demand_max : pr.maximum_production;
        if (!std::isfinite(upper) || upper <= 0.0) {
            add_var(pr_name, lower, 1e9, pr.price_per_bbl);
        } else {
            add_var(pr_name, lower, upper, pr.price_per_bbl);
        }
    }

    struct Coeff { int col; double val; };
    struct RowSpec {
        std::string name;
        double lower;
        double upper;
        std::vector<Coeff> entries;
    };
    std::vector<RowSpec> rows;

    // Constraint: Feedstock allocation to units (sum of draws <= availability)
    for (const auto& f : model_.feedstocks) {
        RowSpec row;
        row.name = "alloc::" + f.name;
        row.lower = -1e9;
        row.upper = 0.0;
        row.entries.push_back({var_idx[f.name], -1.0});
        for (const auto& u : model_.units) {
            std::string feed_var = u.name + "::" + f.name;
            if (var_idx.count(feed_var)) {
                row.entries.push_back({var_idx[feed_var], 1.0});
            }
        }
        if (row.entries.size() > 1) {
            rows.push_back(row);
        }
    }

    // Constraint: Unit capacity limits
    for (const auto& u : model_.units) {
        RowSpec row;
        row.name = "cap::" + u.name;
        row.lower = u.capacity_min;
        row.upper = u.capacity_max;
        for (const auto& [feed, _] : u.yields) {
            std::string feed_var = u.name + "::" + feed;
            if (var_idx.count(feed_var)) {
                row.entries.push_back({var_idx[feed_var], 1.0});
            }
        }
        if (!row.entries.empty()) {
            rows.push_back(row);
        }
    }

    // Constraint: Stream yields from units (out_stream - sum(yield * unit_feed) == 0)
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> stream_sources;
    for (const auto& u : model_.units) {
        for (const auto& [feed, outs] : u.yields) {
            std::string feed_var = u.name + "::" + feed;
            for (const auto& [out_name, frac] : outs) {
                stream_sources[out_name].push_back({feed_var, frac});
            }
        }
    }
    for (const auto& [stream_name, sources] : stream_sources) {
        RowSpec row;
        row.name = "yield::" + stream_name;
        row.lower = 0.0;
        row.upper = 0.0;
        row.entries.push_back({var_idx[stream_name], 1.0});
        for (const auto& [feed_var, frac] : sources) {
            row.entries.push_back({var_idx[feed_var], -frac});
        }
        rows.push_back(row);
    }

    // Constraint: Pool volume balance (sum(inlets) - pool_volume == 0)
    for (const auto& p : model_.pools) {
        RowSpec row;
        row.name = "pool_bal::" + p.name;
        row.lower = 0.0;
        row.upper = 0.0;
        std::string pool_vol_name = "pool::" + p.name + "::vol";
        row.entries.push_back({var_idx[pool_vol_name], -1.0});
        const auto inlets = p.inlet_streams.empty() ? p.inlets : p.inlet_streams;
        for (const auto& inlet : inlets) {
            if (var_idx.count(inlet)) {
                row.entries.push_back({var_idx[inlet], 1.0});
            }
        }
        rows.push_back(row);
    }

    // Constraint: Product draw balance (sum(products from pool) - pool_volume <= 0)
    for (const auto& p : model_.pools) {
        RowSpec row;
        row.name = "pool_draw::" + p.name;
        row.lower = -1e9;
        row.upper = 0.0;
        std::string pool_vol_name = "pool::" + p.name + "::vol";
        row.entries.push_back({var_idx[pool_vol_name], -1.0});
        for (const auto& pr : model_.products) {
            const std::string pool_name = pr.from_pool.empty() ? pr.pool : pr.from_pool;
            if (pool_name == p.name) {
                std::string pr_name = "product::" + pr.name;
                row.entries.push_back({var_idx[pr_name], 1.0});
            }
        }
        if (row.entries.size() > 1) {
            rows.push_back(row);
        }
    }

    // Constraint: Product specifications against pool properties
    for (const auto& pr : model_.products) {
        const std::string pool_name = pr.from_pool.empty() ? pr.pool : pr.from_pool;
        for (const auto& [prop, spec] : pr.specifications) {
            std::string pool_prop_name = "pool::" + pool_name + "::" + prop;
            if (var_idx.count(pool_prop_name)) {
                if (spec.max.has_value()) {
                    RowSpec row;
                    row.name = "spec_max::" + pr.name + "::" + prop;
                    row.lower = -1e9;
                    row.upper = *spec.max;
                    row.entries.push_back({var_idx[pool_prop_name], 1.0});
                    rows.push_back(row);
                }
                if (spec.min.has_value()) {
                    RowSpec row;
                    row.name = "spec_min::" + pr.name + "::" + prop;
                    row.lower = *spec.min;
                    row.upper = 1e9;
                    row.entries.push_back({var_idx[pool_prop_name], 1.0});
                    rows.push_back(row);
                }
            }
        }
    }

    // Assemble CSR matrix A
    pm.linear.A.rows = rows.size();
    pm.linear.A.cols = pm.linear.var_names.size();
    pm.linear.A.row_ptr.assign(rows.size() + 1, 0);

    for (std::size_t i = 0; i < rows.size(); ++i) {
        pm.linear.rows.push_back({rows[i].name, RowSense::Equal, rows[i].upper});
        pm.linear.row_lower.push_back(rows[i].lower);
        pm.linear.row_upper.push_back(rows[i].upper);
        for (const auto& c : rows[i].entries) {
            pm.linear.A.col_index.push_back(c.col);
            pm.linear.A.values.push_back(c.val);
        }
        pm.linear.A.row_ptr[i + 1] = static_cast<int>(pm.linear.A.values.size());
    }

    // Bilinear pooling terms
    pm.constraint_bilinear.resize(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (const auto& p : model_.pools) {
            if (rows[i].name == "pool_bal::" + p.name) {
                const auto tracked = p.tracked_properties.empty() ? std::vector<std::string>{p.blended_property} : p.tracked_properties;
                for (const auto& prop : tracked) {
                    if (prop.empty()) continue;
                    std::string vol_var = "pool::" + p.name + "::vol";
                    std::string prop_var = "pool::" + p.name + "::" + prop;
                    if (var_idx.count(vol_var) && var_idx.count(prop_var)) {
                        pm.constraint_bilinear[i].push_back(
                            BilinearTerm{var_idx[vol_var], var_idx[prop_var], -1.0}
                        );
                    }
                }
            }
        }
    }

    return pm;
}

}  // namespace bharatopt::refinery
