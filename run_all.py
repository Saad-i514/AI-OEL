"""One command to reproduce every table/figure: python run_all.py
   python run_all.py --fast   -> reduced game counts for a quick check (full run takes several minutes)"""
import sys
import time
from members import STAMP
from src.world import run_part_a
from src.twists import run_twists
from src.search import run_search_demo, run_ids_demo
from src.study_b import run_heuristic_study
from src.game import run_node_study
from src.tune_c import main as run_part_c_experiments
from src.llm_agent import run_part_d
from src.experiments import run_experiments
from src.slip_study import run_slip_study

if __name__ == "__main__":
    fast = "--fast" in sys.argv
    print(STAMP, flush=True)

    def run_stage(name, fn):
        start = time.perf_counter()
        print(f"\n=== Starting {name} ===", flush=True)
        fn()
        print(f"=== Finished {name} in {time.perf_counter() - start:.1f}s ===", flush=True)

    run_stage("Part A", run_part_a)
    run_stage("Twists", run_twists)
    run_stage("Search demo", run_search_demo)
    run_stage("IDS demo", run_ids_demo)
    run_stage("Heuristic study", run_heuristic_study)
    run_stage("Node study", run_node_study)
    run_stage("Part C experiments", lambda: run_part_c_experiments(fast=fast))
    run_stage("Part D", run_part_d)      # backend chosen by OEL_LLM_BACKEND (default 'mock' = pipeline test only, NOT an LLM)
    run_stage("Slip study", lambda: run_slip_study(n_games=2 if fast else 30))    # Part C: cost of the deterministic-move assumption (no rival)
    run_stage("Part E experiments", lambda: run_experiments(n_scen=6 if fast else 30))   # Part E: scenarios, CIs, ablation, failure analysis
