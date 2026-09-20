import streamlit as st
import json
import subprocess
import tempfile
from pathlib import Path

st.set_page_config(page_title="BharatOpt | MRPL", page_icon="🛢️", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / "build-cloud"
EXECUTABLE = BUILD_DIR / "bharatopt"


@st.cache_resource(show_spinner=False)
def ensure_bharatopt_engine():
    """Build the native CPU engine once per Streamlit runtime."""
    # Always run the incremental native build once per Streamlit runtime.
    # This prevents a persistent build directory from keeping an older CLI
    # binary after a new C++ source revision is deployed.
    configure = [
        "cmake",
        "-S", str(PROJECT_ROOT),
        "-B", str(BUILD_DIR),
        "-DBHARATOPT_ENABLE_CUDA=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    build = ["cmake", "--build", str(BUILD_DIR), "--target", "bharatopt_cli", "-j2"]

    try:
        configured = subprocess.run(
            configure,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if configured.returncode != 0:
            return None, "CMake configure failed:\n" + configured.stderr[-6000:]

        compiled = subprocess.run(
            build,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if compiled.returncode != 0:
            return None, "Native engine build failed:\n" + compiled.stderr[-6000:]

        if not EXECUTABLE.exists():
            return None, f"Build completed but {EXECUTABLE} was not produced."

        return EXECUTABLE, None
    except subprocess.TimeoutExpired as exc:
        return None, f"Native engine build timed out: {exc}"


# --- CORE CLI BRIDGE ---
def run_pooling_engine(timeout_seconds=30.0):
    exe, build_error = ensure_bharatopt_engine()
    if exe is None:
        return {"status": "CRASH", "error": build_error or "Native BharatOpt engine unavailable."}

    with tempfile.TemporaryDirectory(prefix="bharatopt_pooling_") as tmp_dir:
        output_path = Path(tmp_dir) / "pooling.json"
        command = [str(exe.resolve()), "--pooling-demo", "--output", str(output_path), "--mode", "certified"]

        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "error": f"Pooling solve exceeded {timeout_seconds}s limit."}

        if not output_path.exists():
            return {"status": "CRASH", "error": f"Pooling engine failed. Exit code: {completed.returncode}\nStderr: {completed.stderr}"}

        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {"status": "CRASH", "error": f"Pooling engine returned invalid JSON: {exc}"}


def solve_with_bharatopt_cli(uploaded_file, timeout_seconds=75.0):
    exe, build_error = ensure_bharatopt_engine()
    if exe is None:
        return {"status": "CRASH", "error": build_error or "Native BharatOpt engine unavailable."}

    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.TemporaryDirectory(prefix="bharatopt_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        input_path = tmp_dir / f"input{suffix}"
        output_path = tmp_dir / "result.json"

        input_path.write_bytes(uploaded_file.getbuffer())

        command = [
            str(exe.resolve()),
            "solve",
            "--input", str(input_path),
            "--output", str(output_path),
            "--mode", "certified",
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "error": f"Exceeded {timeout_seconds}s limit."}

        if not output_path.exists():
            return {
                "status": "CRASH",
                "error": (
                    f"Engine failed. Exit code: {completed.returncode}\n"
                    f"Stderr: {completed.stderr}"
                ),
            }

        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "status": "CRASH",
                "error": f"Engine returned invalid JSON: {exc}",
            }


# --- UI LOGIC ---
st.title("BharatOpt")
st.caption("TEAM NOVAKIN • SIH26119 • Indigenous Refinery Optimization Engine")

with st.sidebar:
    st.header("Refinery Configuration")
    uploaded_file = st.file_uploader("Upload Refinery Model", type=["mps", "json"])
    solve_button = st.button("Run Optimizer", type="primary", use_container_width=True)
    pooling_button = st.button("Run Native Pooling Analysis", use_container_width=True)

    st.divider()
    st.subheader("Engine Status")
    st.info("Native C++ engine: built on demand")
    st.caption("JSON uploads use the configurable nonlinear refinery model. MPS uploads use the linear/MILP path.")

if pooling_button:
    with st.spinner("Running native SLP + McCormick benchmark..."):
        pooling_result = run_pooling_engine()

    if isinstance(pooling_result.get("slp_objective"), (int, float)) and isinstance(pooling_result.get("mccormick_bound"), (int, float)):
        st.subheader("Native Pooling Engine Check")
        st.caption("This button runs the repository's built-in regression benchmark. It is an engine test, not a refinery production result.")
        p1, p2, p3 = st.columns(3)
        p1.metric("SLP Objective", f"{pooling_result['slp_objective']:,.2f}")
        p2.metric("McCormick Bound", f"{pooling_result['mccormick_bound']:,.2f}")
        p3.metric("Nonlinear Gap", f"{pooling_result.get('nonlinear_gap', 0) * 100:.4f}%")
        st.success(f"SLP: {pooling_result.get('status')} • McCormick: {pooling_result.get('mccormick_status')}")
    else:
        st.error(f"Pooling engine check failed: {pooling_result.get('error', pooling_result.get('status'))}")

if uploaded_file and solve_button:
    with st.spinner("Preparing native BharatOpt engine and executing model..."):
        result = solve_with_bharatopt_cli(uploaded_file)

    status = result.get("status", "").lower()
    is_json_model = Path(uploaded_file.name).suffix.lower() == ".json"
    success_statuses = ["optimal", "feasible", "globally_certified_within_tolerance", "global_optimal_within_tolerance"]

    if status in success_statuses:
        if is_json_model:
            if result.get("global_optimality_certified"):
                st.success("Global optimum certified within the configured tolerance.")
            else:
                st.success("Feasible production plan found; global certification not reached within the search limits.")
        else:
            if status == "optimal":
                st.success("MPS model solved successfully.")
            else:
                st.success("Feasible MPS solution found.")

        c1, c2, c3, c4 = st.columns(4)
        objective = result.get("objective")
        if is_json_model:
            bound = result.get("mccormick_global_upper_bound", result.get("certified_lower_bound"))
            gap = result.get("nonlinear_gap", result.get("relative_gap"))
            nodes = result.get("nodes_explored", 0)
            bound_label = "Global / Solver Bound"
        else:
            bound = result.get("certified_lower_bound", result.get("mccormick_global_upper_bound"))
            gap = result.get("relative_gap", result.get("nonlinear_gap"))
            nodes = result.get("nodes_explored", 0)
            bound_label = "Solver Bound"

        c1.metric("Objective", f"{objective:,.2f}" if isinstance(objective, (int, float)) else "Not reported")
        c2.metric(bound_label, f"{bound:,.2f}" if isinstance(bound, (int, float)) else "Not reported")
        c3.metric("Gap", f"{gap * 100:.4f}%" if isinstance(gap, (int, float)) else "Not reported")
        c4.metric("B&B Nodes", f"{nodes:,}")

        if is_json_model and "variables" in result:
            st.subheader("Optimized Refinery Variables")
            st.dataframe(result["variables"], use_container_width=True, hide_index=True)
            v = result.get("constraint_max_violation")
            if isinstance(v, (int, float)):
                if v <= 1e-6:
                    st.success(f"Constraint audit passed: maximum violation = {v:.3e}")
                else:
                    st.warning(f"Constraint audit: maximum violation = {v:.3e}")
            st.caption("The displayed solution is generated from the uploaded JSON model. Spatial branch-and-bound tightens McCormick relaxations; certification is reported only when the remaining global bound is within the configured tolerance.")
        else:
            st.info("This is an MPS linear/MILP solve. Nonlinear refinery quantities are available when a JSON refinery model is supplied.")

    elif status == "infeasible":
        st.error("No feasible production plan was found for the supplied model.")
        st.caption("The current native CLI does not yet return a general Farkas certificate for every infeasible model, so no refinery-specific corrective action is invented here.")
        farkas_data = result.get("farkas_multipliers", [])
        if farkas_data:
            with st.expander("View Farkas Certificate"):
                st.json(farkas_data)
    else:
        st.error(f"Solver terminated: {result.get('status')}")
        st.write(result.get("error", "Unknown solver error."))

elif not uploaded_file:
    st.info("Upload a refinery model via the sidebar to begin.")
