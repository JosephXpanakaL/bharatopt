#pragma once
// include/bharatopt/refinery_model_types.hpp
//
// Plain data structures mirroring examples/refinery_demo.json.
// No solver logic here on purpose — this file only describes the refinery
// domain. refinery_planner.{hpp,cpp} is what translates these into the
// generic PoolingModel your solver already consumes.

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace bharatopt::refinery {

struct PropertyLimit {
    std::optional<double> min;
    std::optional<double> max;
};

struct Feedstock {
    std::string name;
    double availability = 0.0;      // volume units per period
    double cost_per_bbl = 0.0;
    double carbon_intensity_kg_co2_bbl = 0.0;  // Upstream Scope-3 carbon footprint
    std::unordered_map<std::string, double> properties;  // e.g. "sulfur_wt_pct", "density"
};

struct Unit {
    std::string name;
    std::string type;               // "distillation" | "cracking" | "hydrodesulfurization" | ...
    std::vector<std::string> feed_streams;   // feedstock names OR upstream stream names
    double capacity_min = 0.0;
    double capacity_max = 0.0;
    double energy_cost_per_bbl = 0.0;
    double direct_emission_kg_co2_per_bbl = 0.0; // Scope 1 direct emissions
    // yields[feed_name][output_stream_name] = fraction of feed volume routed to that output.
    // Linear per feed — see refinery_demo_readme.md "where the nonlinearity lives".
    std::unordered_map<std::string, std::unordered_map<std::string, double>> yields;
    // Optional: a unit that changes a property rather than (only) volume, e.g. HDS.
    // output_properties[output_stream_name][property_name] = value.
    std::unordered_map<std::string, std::unordered_map<std::string, double>> output_properties;
};

struct Stream {
    std::string name;
    std::string source;             // producing unit's name
    double carbon_intensity_kg_co2_bbl = 0.0;
    std::unordered_map<std::string, double> properties;
};

struct Pool {
    std::string name;
    std::vector<std::string> inlet_streams;
    std::vector<std::string> inlets; // compatibility alias for older JSON contracts
    std::string blended_property;   // property name this pool blends on, e.g. "sulfur_wt_pct"
    std::vector<std::string> tracked_properties; // compatibility alias
    bool bilinear = true;
};

struct ProductSpec {
    std::string name;
    std::string from_pool;          // set iff the product is drawn from a pool
    std::string from_stream;        // set iff the product is drawn directly from a stream
    std::string pool;               // compatibility alias for older JSON contracts
    double minimum_production = 0.0;
    double maximum_production = 0.0;
    double demand_min = 0.0;        // compatibility alias
    double demand_max = 0.0;        // compatibility alias
    double price_per_bbl = 0.0;
    // specification limits for a property, e.g. {"sulfur": {"max": 10.0}}
    std::unordered_map<std::string, PropertyLimit> specifications;
};

struct PlanningPeriod {
    int period_id{0};
    std::string name;
    double duration_days{30.0};
    double discount_factor{1.0};
};

struct BlendCompatibilityPenalty {
    std::string stream_a;
    std::string stream_b;
    double max_fraction_together{1.0};
    double penalty_cost_per_bbl{0.0};
};

struct RefineryModel {
    std::string name;
    bool synthetic = false;
    std::vector<Feedstock> feedstocks;
    std::vector<Unit> units;
    std::vector<Stream> streams;
    std::vector<Pool> pools;
    std::vector<ProductSpec> products;
    std::vector<PlanningPeriod> periods;
    std::vector<BlendCompatibilityPenalty> compatibility_rules;
    double max_carbon_emissions_mt = 0.0;   // Emission cap (metric tons CO2 eq)
    double carbon_tax_per_ton = 0.0;        // Carbon tax / penalty ($/MT CO2)
};

struct EmissionAuditResult {
    double scope1_emissions_mt{0.0};
    double scope3_upstream_emissions_mt{0.0};
    double total_emissions_mt{0.0};
    double total_carbon_tax_cost{0.0};
    bool satisfies_emission_cap{true};
};

}  // namespace bharatopt::refinery
