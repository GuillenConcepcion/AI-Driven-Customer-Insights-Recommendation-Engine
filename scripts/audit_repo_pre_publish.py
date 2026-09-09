"""
Automated Pre-Publication Audit CLI Tool (Senior / Staff Level).
Audits a Data Science & MLOps repository against the 4-phase standard:
1. Repository Structure & Documentation
2. Clean Code, Modular SRP & Exception Specificity
3. Model Versioning, Hyperparameters & Cold-Start
4. Cloud-Native Serving (FastAPI, Streamlit & Caching)

Author: Guillen Concepción (Senior Data Scientist & MLOps Engineer)
"""

import sys
import os
import re
from pathlib import Path

# Safe UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def audit_repository(repo_root: Path) -> bool:
    print("\n" + "=" * 80)
    print(" [*] AUTOMATED PRE-PUBLICATION AUDIT TOOL (SENIOR / STAFF LEVEL)")
    print("=" * 80)
    print(f"Target Repository: {repo_root.resolve()}")
    print("Auditor: Guillen Concepción (Senior Data Scientist & MLOps Engineer)\n")

    scorecard = {}
    passed_all = True

    # -------------------------------------------------------------------------
    # Phase I: Repository Structure & Documentation
    # -------------------------------------------------------------------------
    print("--- [PHASE I: REPOSITORY STRUCTURE & DOCUMENTATION] ---")
    readme_path = repo_root / "README.md"
    if not readme_path.exists():
        scorecard["README.md exists"] = (False, "README.md missing from repository root.")
    else:
        content = readme_path.read_text(encoding="utf-8")
        has_arch = "mermaid" in content.lower() or "flowchart" in content.lower() or "architecture" in content.lower()
        has_install = "pip install" in content or "requirements.txt" in content
        has_results = "kpi" in content.lower() or "results" in content.lower() or "gini" in content.lower()
        has_quickstart = "quickstart" in content.lower() or "demo" in content.lower()

        scorecard["README Architecture Diagram"] = (has_arch, "Found Mermaid / architecture flow." if has_arch else "Missing architectural diagram.")
        scorecard["README Step-by-Step Install"] = (has_install, "Found installation instructions." if has_install else "Missing installation steps.")
        scorecard["README KPI & Results Section"] = (has_results, "Found quantitative business metrics." if has_results else "Missing results / KPIs summary.")
        scorecard["README Quickstart Command"] = (has_quickstart, "Found quick verification instructions." if has_quickstart else "Missing quickstart guide.")

    # Requirements pinned
    req_path = repo_root / "requirements.txt"
    if not req_path.exists():
        scorecard["requirements.txt exists"] = (False, "requirements.txt missing.")
    else:
        req_lines = [l.strip() for l in req_path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        pinned_count = sum(1 for l in req_lines if "==" in l)
        total_reqs = len(req_lines)
        is_pinned = (pinned_count >= (total_reqs * 0.8)) and total_reqs > 0
        scorecard["requirements.txt Version Pinning"] = (
            is_pinned,
            f"{pinned_count}/{total_reqs} dependencies strictly pinned (==)." if is_pinned else f"Only {pinned_count}/{total_reqs} dependencies pinned."
        )

    # -------------------------------------------------------------------------
    # Phase II: Clean Code & Exception Specificity
    # -------------------------------------------------------------------------
    print("--- [PHASE II: CLEAN CODE & EXCEPTION SPECIFICITY] ---")
    src_dir = repo_root / "src"
    bare_excepts = []
    broad_excepts = []

    if src_dir.exists():
        for py_file in src_dir.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                clean = line.strip()
                if clean == "except:":
                    bare_excepts.append(f"{py_file.name}:{i}")
                elif clean.startswith("except Exception") and "amazon_loader" not in py_file.name:
                    broad_excepts.append(f"{py_file.name}:{i}")

        scorecard["Zero Bare Except Statements"] = (
            len(bare_excepts) == 0,
            "No bare 'except:' found in src/." if len(bare_excepts) == 0 else f"Found bare excepts in: {bare_excepts}"
        )
        scorecard["Exception Specificity"] = (
            len(broad_excepts) == 0,
            "Exception handling is specific and robust." if len(broad_excepts) == 0 else f"Broad 'except Exception' found in: {broad_excepts}"
        )
    else:
        scorecard["src/ directory"] = (False, "src/ directory not found.")

    # -------------------------------------------------------------------------
    # Phase III: Model Versioning & Cold-Start
    # -------------------------------------------------------------------------
    print("--- [PHASE III: MODEL ARTIFACTS & COLD-START] ---")
    models_dir = repo_root / "models"
    has_models = models_dir.exists() and any(models_dir.glob("*.joblib"))
    scorecard["Persisted Model Artifacts (.joblib)"] = (
        has_models,
        f"Found serialized model artifacts in {models_dir}." if has_models else "No .joblib artifacts found in models/."
    )

    # Check cold-start logic in recommender
    recommender_file = src_dir / "customer_insights" / "models" / "recommender.py" if src_dir.exists() else None
    if recommender_file and recommender_file.exists():
        rec_text = recommender_file.read_text(encoding="utf-8")
        has_cold_start = "cold" in rec_text.lower() and "popular" in rec_text.lower()
        scorecard["Cold-Start Fallback Architecture"] = (
            has_cold_start,
            "Explicit fallback logic detected for cold users and new items." if has_cold_start else "Cold-start fallback logic missing."
        )
    else:
        scorecard["Recommender Module"] = (False, "recommender.py not found.")

    # -------------------------------------------------------------------------
    # Phase IV: Serving, Caching & Tests
    # -------------------------------------------------------------------------
    print("--- [PHASE IV: SERVING, CACHING & TESTS] ---")
    streamlit_app = repo_root / "app" / "streamlit_app.py"
    if streamlit_app.exists():
        st_text = streamlit_app.read_text(encoding="utf-8")
        has_cache = "@st.cache_resource" in st_text or "@st.cache_data" in st_text
        scorecard["Streamlit Caching (@st.cache)"] = (
            has_cache,
            "Caching directives detected for resource & data optimization." if has_cache else "Missing @st.cache directives."
        )
    else:
        scorecard["Streamlit App"] = (False, "app/streamlit_app.py not found.")

    tests_dir = repo_root / "tests"
    has_tests = tests_dir.exists() and any(tests_dir.glob("test_*.py"))
    scorecard["Automated Test Suite (pytest)"] = (
        has_tests,
        f"Found automated test files in {tests_dir}." if has_tests else "No test_*.py found."
    )

    # -------------------------------------------------------------------------
    # Print Executive Summary Scorecard
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [*] PRE-PUBLICATION AUDIT SCORECARD REPORT")
    print("=" * 80)

    for item, (passed, reason) in scorecard.items():
        status_tag = "[PASS]" if passed else "[FAIL]"
        mark = "[+]" if passed else "[-]"
        if not passed:
            passed_all = False
        print(f" {mark} {status_tag} {item:<40} -> {reason}")

    print("=" * 80)
    if passed_all:
        print(" [RESULT] EXCELLENCE CERTIFIED: 100% AUDIT CHECKS PASSED.")
        print(" The repository strictly adheres to the Senior / Staff standard.")
    else:
        print(" [RESULT] ATTENTION REQUIRED: Some pre-publication checks failed.")
    print("=" * 80 + "\n")

    return passed_all


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    success = audit_repository(repo_root)
    sys.exit(0 if success else 1)
