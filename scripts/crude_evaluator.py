#!/usr/bin/env python3
"""
Crude Netback Margin and Economic Evaluator for MRPL Refinery Operations
Calculates gross product worth (GPW), net operating margin, and rank of crude feedstocks.
"""

import json
import sys
from pathlib import Path

def evaluate_crude_flowsheet(flowsheet_path):
    p = Path(flowsheet_path)
    if not p.exists():
        print(f"Error: {flowsheet_path} does not exist.")
        return 1

    data = json.loads(p.read_text(encoding="utf-8"))
    print(f"=== Crude Economic Netback Analysis: {data.get('name')} ===")
    
    products = {prod["name"]: prod.get("price_per_bbl", 0.0) for prod in data.get("products", [])}
    print("\nBenchmark Product Prices ($/bbl):")
    for k, v in products.items():
        print(f"  - {k}: ${v:.2f}")

    print("\nCrude Netback Valuation ($/bbl):")
    cdu = next((u for u in data.get("units", []) if "CDU" in u.get("name", "")), None)
    if not cdu:
        print("No CDU unit found in flowsheet.")
        return 0

    cdu_yields = cdu.get("yields", {})
    energy_cost = cdu.get("energy_cost_per_bbl", 1.8)

    crude_eval = []
    for fs in data.get("feedstocks", []):
        name = fs["name"]
        cost = fs.get("cost_per_bbl", 70.0)
        yields = cdu_yields.get(name, {})
        
        # Estimate Gross Product Worth based on middle distillates and reformate potential
        gpw = 0.0
        for cut, frac in yields.items():
            price = 85.0
            if "Naphtha" in cut: price = 95.0
            elif "Kerosene" in cut or "ATF" in cut: price = 115.0
            elif "Gas_Oil" in cut or "Diesel" in cut: price = 108.0
            elif "Residue" in cut: price = 60.0
            gpw += frac * price
            
        netback = gpw - cost - energy_cost
        crude_eval.append((name, cost, gpw, netback))

    crude_eval.sort(key=lambda x: x[3], reverse=True)
    for rank, (name, cost, gpw, netback) in enumerate(crude_eval, 1):
        print(f"  #{rank} {name:<18} | Cost: ${cost:.2f} | GPW: ${gpw:.2f} | Netback Margin: ${netback:+.2f}/bbl")

    print("\nEconomic ranking successfully computed.")
    return 0

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "examples/mrpl_refinery_flowsheet.json"
    sys.exit(evaluate_crude_flowsheet(target))
