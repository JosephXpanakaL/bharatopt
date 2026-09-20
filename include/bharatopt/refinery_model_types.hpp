#pragma once
// include/bharatopt/refinery_model_types.hpp
//
// Plain data structures mirroring examples/refinery_demo.json.
// No solver logic here on purpose — this file only describes the refinery
// domain. refinery_planner.{hpp,cpp} is what translates these into the
// generic PoolingModel your solver already consumes.

#include <string>
#include <unordered_map>
#include <vector>

namespace bharatopt::refinery {

struct Feedstock {
    std::string name;
    double availability = 0.0;      // volume units per period
    double cost_per_bbl = 0.0;
    std::unordered_map<std::string, double> properties;  // e.g. "sulfur_wt_pct", "density"
};

struct Unit {
    std::string name;
    std::string type;               // "distillation" | "cracking" | "hydrodesulfurization" | ...
    std::vector<std::string> feed_streams;   // feedstock names OR upstream stream names
    double capacity_min = 0.0;
    double capacity_max = 0.0;
    double energy_cost_per_bbl = 0.0;
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
    std::unordered_map<std::string, double> properties;
};

struct Pool {
    std::string name;
    std::vector<std::string> inlet_streams;
    std::string blended_property;   // property name this pool blends on, e.g. "sulfur_wt_pct"
    bool bilinear = true;
};

struct ProductSpec {
    std::string name;
    std::string from_pool;          // set iff the product is drawn from a pool
    std::string from_stream;        // set iff the product is drawn directly from a stream
    double minimum_production = 0.0;
    double maximum_production = 0.0;
    double price_per_bbl = 0.0;
    // specifications["sulfur_max"] = 0.0010, specifications["RON_min"] = 91, etc.
    std::unordered_map<std::string, double> specifications;
};

struct RefineryModel {
    std::string name;
    bool synthetic = false;
    std::vector<Feedstock> feedstocks;
    std::vector<Unit> units;
    std::vector<Stream> streams;
    std::vector<Pool> pools;
    std::vector<ProductSpec> products;
};

}  // namespace bharatopt::refinery
