#include "bharatopt/refinery_planner.hpp"

#include <cassert>
#include <iostream>
#include <fstream>

int main() {
    // Create a temporary JSON refinery flowsheet for testing
    const std::string test_json_path = "test_planner_model.json";
    {
        std::ofstream out(test_json_path);
        out << R"({
  "name": "Unit_Test_Refinery",
  "feedstocks": [
    {
      "name": "Arab_Light",
      "availability": 50000.0,
      "cost_per_bbl": 82.5,
      "properties": {"sulfur": 1.7, "api": 33.0}
    },
    {
      "name": "Arab_Heavy",
      "availability": 30000.0,
      "cost_per_bbl": 71.0,
      "properties": {"sulfur": 2.9, "api": 27.5}
    }
  ],
  "units": [
    {
      "name": "CDU",
      "type": "distillation",
      "capacity_min": 10000.0,
      "capacity_max": 80000.0,
      "feed_streams": ["Arab_Light", "Arab_Heavy"],
      "yields": {
        "Arab_Light": {"naphtha": 0.25, "diesel_cut": 0.35, "residue": 0.40},
        "Arab_Heavy": {"naphtha": 0.15, "diesel_cut": 0.30, "residue": 0.55}
      }
    }
  ],
  "pools": [
    {
      "name": "Diesel_Pool",
      "inlets": ["diesel_cut"],
      "tracked_properties": ["sulfur"]
    }
  ],
  "products": [
    {
      "name": "Ultra_Low_Sulfur_Diesel",
      "pool": "Diesel_Pool",
      "demand_min": 5000.0,
      "demand_max": 40000.0,
      "price_per_bbl": 105.0,
      "specifications": {
        "sulfur": {"max": 10.0}
      }
    }
  ]
})";
    }

    try {
        bharatopt::refinery::RefineryPlanner planner(test_json_path);
        auto validation = planner.validate();
        std::string json_report = validation.to_json();
        
        assert(validation.valid == true);
        assert(validation.errors.empty());
        assert(planner.model().name == "Unit_Test_Refinery");
        assert(planner.model().feedstocks.size() == 2);
        assert(planner.model().units.size() == 1);
        assert(planner.model().pools.size() == 1);
        assert(planner.model().products.size() == 1);
        assert(!json_report.empty());

        auto pm = planner.build();
        assert(pm.linear.A.cols > 0);
        assert(pm.linear.A.rows > 0);
        assert(!pm.linear.var_names.empty());

        std::cout << "RefineryPlanner validation and build tests passed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
