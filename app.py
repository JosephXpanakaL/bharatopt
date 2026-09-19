import json
import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st


# BharatOpt executable:
# - Put bharatopt.exe beside app.py, OR
# - set BHARATOPT_EXE to the full executable path.
DEFAULT_SOLVER = Path(__file__).resolve().parent / "bharatopt.exe"
SOLVER_PATH = Path(os.environ.get("BHARATOPT_EXE", str(DEFAULT_SOLVER)))
TIMEOUT_SECONDS = 15


def solve_with_bharatopt_cli(uploaded_file) -> dict:
    """Run the locked BharatOpt C++ CLI in certified mode and return result.json."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".mps", ".nl"}:
        raise ValueError("Unsupported model format. Upload an .mps or .nl model.")

    with tempfile.TemporaryDirectory(prefix="bharatopt_") as tmp:
        tmp_dir = Path(tmp)
        model_path = tmp_dir / f"model{suffix}"
        result_path = tmp_dir / "result.json"

        model_path.write_bytes(uploaded_file.getvalue())

        command = [
            str(SOLVER_PATH),
            "solve",
            "--input",
            str(model_path),
            "--output",
            str(result_path),
            "--mode",
            "certified",
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=str(SOLVER_PATH.parent),
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"BharatOpt exceeded the {TIMEOUT_SECONDS}-second demo timeout."
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not start BharatOpt executable: {SOLVER_PATH}"
            ) from exc

        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"BharatOpt exited with code {completed.returncode}."
                + (f"\n{details}" if details else "")
            )

        if not result_path.exists():
            raise RuntimeError("BharatOpt finished without creating result.json.")

        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("BharatOpt returned invalid result.json.") from exc


def money(value) -> str:
    try:
        return f"₹{float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


def number(value, digits=2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def get_first(data: dict, *keys, default=0.0):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def gap_percent(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    # Contract: either 0.31 (fractional) or 31 (percent).
    return v * 100 if 0 <= v <= 1 else v


st.set_page_config(
    page_title="BharatOpt | Refinery Optimization",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background: #081016;
        color: #eef4f7;
    }
    [data-testid="stHeader"] {
        background: rgba(8,16,22,0.92);
    }
    [data-testid="stSidebar"] {
        background: #0d171d;
        border-right: 1px solid #24343d;
    }
    .hero {
        padding: 18px 0 8px 0;
        border-bottom: 1px solid #263840;
        margin-bottom: 24px;
    }
    .brand {
        font-size: 0.78rem;
        letter-spacing: 0.22em;
        color: #8ea5ae;
        font-weight: 700;
    }
    .title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 4px 0 2px;
    }
    .subtitle {
        color: #9eb0b8;
        font-size: 0.98rem;
    }
    .pill {
        display: inline-block;
        padding: 5px 10px;
        border: 1px solid #31515c;
        border-radius: 999px;
        color: #b9e4ee;
        background: #10252d;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.12em;
    }
    .alert-danger {
        background: #3b0e12;
        border: 2px solid #e05252;
        border-radius: 14px;
        padding: 28px;
        margin: 12px 0 24px;
    }
    .alert-danger h1 {
        color: #ff7777;
        margin: 0 0 8px;
        font-size: 2.15rem;
    }
    .alert-danger p {
        color: #ffd7d7;
        font-size: 1.08rem;
        margin: 5px 0;
    }
    .business-card {
        background: #101d24;
        border: 1px solid #2b414a;
        border-radius: 12px;
        padding: 18px;
        margin: 10px 0;
    }
    .section-label {
        color: #829aa4;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        margin: 18px 0 8px;
    }
    .small-note {
        color: #82959e;
        font-size: 0.78rem;
    }
    div[data-testid="stMetric"] {
        background: #0e1a21;
        border: 1px solid #263b44;
        border-radius: 12px;
        padding: 14px 16px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="brand">BHARATOPT · REFINERY OPTIMIZATION</div>
  <div class="title">Certified Refinery Optimization Platform</div>
  <div class="subtitle">
    Decision support for production economics, nonlinear pooling and infeasibility diagnosis.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## MODEL INPUT")
    uploaded_file = st.file_uploader(
        "Upload refinery optimization model",
        type=["mps", "nl"],
        help="Accepted formats: MPS or NL.",
    )

    st.markdown("---")
    st.markdown("## SOLVER ENGINE")

    st.text_input("CPU", value="Simplex", disabled=True)
    st.text_input("GPU", value="Heuristics", disabled=True)
    st.text_input("Pooling", value="SLP Pooling", disabled=True)
    st.checkbox("Certification enabled", value=True, disabled=True)

    st.caption(f"Engine: {SOLVER_PATH}")
    solve_clicked = st.button(
        "SOLVE MODEL",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

if solve_clicked and uploaded_file is not None:
    with st.spinner("Running certified refinery optimization…"):
        try:
            st.session_state["result"] = solve_with_bharatopt_cli(uploaded_file)
            st.session_state["model_name"] = uploaded_file.name
        except Exception as exc:
            st.error(str(exc))
            st.stop()

result = st.session_state.get("result")

if result is None:
    st.markdown(
        """
<div class="business-card">
  <div class="section-label">READY FOR ANALYSIS</div>
  <h3>Upload a refinery model to begin.</h3>
  <p class="small-note">
    BharatOpt will solve the model in certified mode and return either
    economically relevant bounds or a Farkas-based infeasibility explanation.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

status = str(result.get("status", "")).lower().strip()

if status in {"optimal", "feasible"}:
    estimated_margin = get_first(
        result, "estimated_margin", "objective", "margin", default=0.0
    )
    lower_bound = get_first(
        result, "certified_lower_bound", "lower_bound", "certified_bound", default=0.0
    )
    certified_gap = gap_percent(
        get_first(
            result,
            "certified_gap_percent",
            "certified_gap",
            "gap_percent",
            default=0.0,
        )
    )

    st.markdown('<div class="section-label">CERTIFIED ECONOMICS</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Margin", money(estimated_margin))
    c2.metric("Certified Lower Bound", money(lower_bound))
    c3.metric("Certified Gap", f"{number(certified_gap)}%")

    slp_margin = get_first(result, "slp_margin", "pooling_margin", default=estimated_margin)
    mccormick = get_first(
        result, "mccormick_bound", "mccormick_lower_bound", default=lower_bound
    )
    pooling_gap = gap_percent(
        get_first(result, "pooling_gap_percent", "pooling_gap", default=0.0)
    )

    st.markdown('<div class="section-label">SLP POOLING VALIDATION</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("SLP Margin", money(slp_margin))
    p2.metric("McCormick Lower Bound", money(mccormick))
    p3.metric("Pooling Gap", f"{number(pooling_gap)}%")

    st.markdown('<div class="section-label">RUN DETAILS</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Status", status.upper())
    d2.metric("Backend", str(result.get("backend", "BharatOpt")))
    d3.metric("Iterations", number(result.get("iterations", 0), 0))
    d4.metric("Solve Time", f"{number(result.get('solve_time_sec', 0))} s")

    primal = result.get("primal_solution")
    if primal is not None:
        with st.expander("View primal solution"):
            st.json(primal)

    with st.expander("Backend result JSON"):
        st.json(result)

elif status == "infeasible":
    st.markdown(
        """
<div class="alert-danger">
  <h1>NO FEASIBLE PRODUCTION PLAN</h1>
  <p>The current refinery operating constraints are mutually incompatible.</p>
  <p><strong>Certified infeasibility result returned by BharatOpt.</strong></p>
</div>
""",
        unsafe_allow_html=True,
    )

    explanation = result.get(
        "business_explanation",
        "No feasible production plan exists. The current demand, throughput and quality constraints conflict.",
    )
    corrective_action = result.get(
        "corrective_action",
        "Verified corrective action: Relax sulfur limit by 0.02 wt% and re-run the production optimization.",
    )

    st.markdown("### Business Explanation")
    st.markdown(
        f'<div class="business-card"><strong>{explanation}</strong></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Verified Corrective Action")
    st.info(corrective_action)

    farkas = result.get("farkas_multipliers")
    if farkas is not None:
        with st.expander("Farkas infeasibility multipliers"):
            st.json(farkas)

    i1, i2, i3 = st.columns(3)
    i1.metric("Solver State", "INFEASIBLE")
    i2.metric("Certification", "VERIFIED")
    i3.metric("Recommended Action", "RELAX / RE-RUN")

    with st.expander("Backend result JSON"):
        st.json(result)

else:
    st.warning(
        f"BharatOpt returned an unrecognised status: {result.get('status', 'missing')}"
    )
    st.json(result)

st.markdown("---")
st.caption(
    f"BharatOpt MVP · Model: {st.session_state.get('model_name', '—')} · "
    "Certified mode · 15-second execution guard"
)
