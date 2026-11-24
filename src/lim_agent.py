"""Part D: LLM dispatcher (ReAct loop), three interchangeable backends, 20-request experiment.
Run:  python -m src.llm_agent                      (full experiment: direct planner vs LLM raw vs LLM + validation)
      python -m src.llm_agent --ask "Deliver stop 3 first"     (one live request, for the viva)
Backend (environment variables):
   OEL_LLM_BACKEND = anthropic | openai | mock      OEL_LLM_MODEL = model name      OEL_LLM_BASE_URL = for 'openai' backend
   ANTHROPIC_API_KEY / OPENAI_API_KEY
   'openai' = any OpenAI-compatible server: Ollama (http://localhost:11434/v1), Groq, OpenRouter, OpenAI...
   'mock'   = a scripted STUB, NOT an LLM. Only for testing the pipeline offline. Never report mock numbers as LLM results.
Every LLM reply is cached in results/llm_cache.json, so re-running run_all.py reproduces the same tables without new calls.
"""
import os, re, sys, json, hashlib, math
import requests
from members import STAMP
from src.dispatch_tools import Env, REQUESTS, TOOLS, ground_truth
from src.tune_c import wilson

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CACHE_FILE = os.path.join(RESULTS, "llm_cache.json")


# ------------------------------------------------------------------ backends
class Cache:
    def __init__(self):
        self.d = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}
    def get(self, k): return self.d.get(k)
    def put(self, k, v):
        self.d[k] = v; json.dump(self.d, open(CACHE_FILE, "w"))


class HTTPBackend:
    def __init__(self, name, model, cache):
        self.name, self.model, self.cache = name, model, cache
    def chat(self, system, messages):
        key = hashlib.sha256(json.dumps([self.name, self.model, system, messages]).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        reply = self._call(system, messages)
        self.cache.put(key, reply)
        return reply


class AnthropicBackend(HTTPBackend):
    def _call(self, system, messages):
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=120,
                          headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json={"model": self.model, "max_tokens": 1000, "temperature": 0, "system": system, "messages": messages})
        if r.status_code != 200:
            raise RuntimeError(f"Anthropic API error {r.status_code}: {r.text[:300]}  (check ANTHROPIC_API_KEY / OEL_LLM_MODEL)")
        return r.json()["content"][0]["text"]


class OpenAICompatBackend(HTTPBackend):
    def _call(self, system, messages):
        base = os.environ.get("OEL_LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        hdr = {"content-type": "application/json"}
        if os.environ.get("OPENAI_API_KEY"):
            hdr["authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
        r = requests.post(base + "/chat/completions", headers=hdr, timeout=300,
                          json={"model": self.model, "temperature": 0, "messages": [{"role": "system", "content": system}] + messages})
        if r.status_code != 200:
            raise RuntimeError(f"LLM server error {r.status_code}: {r.text[:300]}  (check OEL_LLM_BASE_URL / OEL_LLM_MODEL)")
        return r.json()["choices"][0]["message"]["content"]


class MockBackend:
    """NOT AN LLM. Reads the ground-truth spec of the request, answers correctly, but injects scripted mistakes on the
    first attempt for some requests so that the validation layer has something to catch. Pipeline testing only."""
    name = "mock"; model = "scripted-stub"
    def __init__(self):
        self.specs = {text: spec for _, _, text, spec in REQUESTS}
    def chat(self, system, messages):
        text = messages[0]["content"]; spec = self.specs.get(text, {})
        rid = next((r for r, _, t, _ in REQUESTS if t == text), 0)
        n_obs = len(messages) // 2
        last = messages[-1]["content"]
        if n_obs > 0 and last.startswith("OBSERVATION:"):
            obs = json.loads(last[len("OBSERVATION:"):])
            if "error" in obs and "REJECTED" in obs["error"]:
                return json.dumps({"thought": "The validator rejected my call; I will correct the arguments.",
                                   "action": {"tool": "plan_route", "args": spec}})
            if obs.get("feasible") is False:
                return json.dumps({"thought": "The planner says it is impossible.", "final": {"status": "infeasible", "message": obs["reason"]}})
            return json.dumps({"thought": "I have a valid route.", "final": {"status": "ok", "message": "route found"}})
        args = dict(spec); tool = "plan_route"; mode = rid % 5
        if rid in (7, 8, 11, 12) and mode in (2, 3, 0, 2): args["avoid_terrain"] = ["swamp" if rid != 12 else "ford"]     # invented terrain
        if rid in (5, 6, 11, 13, 18) and mode in (0, 1, 3, 3, 3): args.pop("first", None)                                  # ignored constraint
        if rid == 9: args["max_cost"] = "350 units"                                                                         # malformed value
        if rid == 2: tool = "route_planner"                                                                                 # wrong tool
        if rid == 6: args["stops"] = [2, 5, 9]                                                                              # invented stop
        return json.dumps({"thought": "I will call the route planner with the constraints from the request.",
                           "action": {"tool": tool, "args": args}})


def make_backend():
    kind = os.environ.get("OEL_LLM_BACKEND", "mock").lower()
    if kind == "mock":
        return MockBackend()
    cache = Cache()
    if kind == "anthropic":
        return AnthropicBackend("anthropic", os.environ.get("OEL_LLM_MODEL", "claude-haiku-5-5"), cache)
    if kind == "openai":
        return OpenAICompatBackend("openai", os.environ.get("OEL_LLM_MODEL", "llama3.1:8b"), cache)
    raise SystemExit(f"unknown OEL_LLM_BACKEND '{kind}' (use anthropic | openai | mock)")


# ------------------------------------------------------------------ agent (ReAct: Thought / Action / Observation)
def system_prompt(env):
    tools = "\n".join(f"- {n}({', '.join(f'{k}: {v}' for k, v in a.items())})" for n, a in TOOLS.items())
    return f"""You are a delivery dispatcher. {env.describe()}
Use the tools to answer the dispatcher's request. Tools:
{tools}
plan_route returns the optimal route as a string of moves. simulate replays a move string. fuel_report checks fuel use of a move string.
Rules: stop numbers are 1..{env.N}; terrain names are exactly road, grass, mud, ford. Never invent stops, terrains or tools.
Reply with ONE JSON object and nothing else. Either call a tool:
{{"thought": "...", "action": {{"tool": "plan_route", "args": {{...}}}}}}
or finish:
{{"thought": "...", "final": {{"status": "ok" or "infeasible", "message": "..."}}}}
Say "infeasible" only if plan_route reports that no route satisfies the request. After a successful plan_route you may finish."""


def parse_json(text):
    t = re.sub(r"```(?:json)?", "", text)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


def run_agent(env, request_text, llm, validate, max_steps=8):
    env.last_plan = None
    messages = [{"role": "user", "content": request_text}]
    sysmsg = system_prompt(env)
    rec = dict(tool_calls=0, rejected=0, wrong_tool=0, bad_args=0, format_errors=0, final=None)
    trace = []
    for step in range(1, max_steps + 1):
        reply = llm.chat(sysmsg, messages)
        messages.append({"role": "assistant", "content": reply})
        obj = parse_json(reply)
        if obj is None:
            rec["format_errors"] += 1
            trace.append((step, "(unparseable reply)", reply[:120], "reply must be ONE JSON object"))
            messages.append({"role": "user", "content": "OBSERVATION: " + json.dumps({"error": "reply must be one JSON object"})})
            continue
        thought = str(obj.get("thought", ""))
        if "final" in obj:
            rec["final"] = obj["final"] if isinstance(obj["final"], dict) else {"status": str(obj["final"])}
            trace.append((step, thought, "FINAL " + json.dumps(rec["final"]), None))
            break
        act = obj.get("action") if isinstance(obj.get("action"), dict) else {}
        name, args = act.get("tool"), act.get("args", {})
        res = env.call(name, args, validate)
        rec["tool_calls"] += 1; rec["rejected"] += res["rejected"]; rec["wrong_tool"] += res["wrong_tool"]; rec["bad_args"] += res["bad_args"]
        obs_text = json.dumps(res["obs"])
        trace.append((step, thought, f"{name} {json.dumps(args)}", obs_text))
        messages.append({"role": "user", "content": "OBSERVATION: " + obs_text})
    return rec, trace


def judge(env, spec, expect_ok, rec):
    """Independent success test + failure labels."""
    fin = rec["final"]
    status = (fin or {}).get("status")
    if fin is None:
        return False, "no final answer"
    if not expect_ok:
        return status == "infeasible", "" if status == "infeasible" else "claimed a route for an impossible request"
    if status != "ok":
        return False, "gave up although a route exists"
    plan = env.last_plan
    if plan is None:
        return False, "claimed success without a plan"
    return (True, "") if env.verify(spec, plan["actions"]) else (False, "ignored a constraint")


def format_trace(rid, text, mode, trace, ok):
    out = [f"--- request {rid} [{mode}]: {text}  ->  {'SUCCESS' if ok else 'FAIL'}"]
    for step, thought, action, obs in trace:
        out.append(f"step {step}")
        out.append(f"   REASON  (Thought):     {thought}")
        out.append(f"   ACT     (Action):      {action}")
        if obs is not None:
            out.append(f"   SENSE   (Observation): {obs[:300]}")
    return "\n".join(out)


def mean_ci(x):
    x = [float(v) for v in x]
    if not x: return float("nan"), 0.0
    m = sum(x) / len(x)
    if len(x) < 2: return m, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
    return m, 1.96 * sd / math.sqrt(len(x))


def run_part_d():
    print(STAMP)
    llm = make_backend(); env = Env()
    if llm.name == "mock":
        print("\n*** BACKEND = MOCK STUB (not an LLM). Numbers below test the pipeline only; do NOT put them in the report. ***")
    else:
        print(f"\nBackend: {llm.name}  model: {llm.model}")
    rows = []; traces = []; direct = {}
    for rid, lvl, text, spec in REQUESTS:
        gt = ground_truth(env, spec)
        direct[rid] = gt
    for mode, validate in (("raw", False), ("validated", True)):
        for rid, lvl, text, spec in REQUESTS:
            gt = direct[rid]; expect_ok = gt["feasible"]
            rec, trace = run_agent(env, text, llm, validate)
            ok, why = judge(env, spec, expect_ok, rec)
            cost = env.last_plan["cost"] if (ok and expect_ok and env.last_plan) else None
            rows.append(dict(mode=mode, rid=rid, level=lvl, expect_ok=expect_ok, success=ok, why=why, cost=cost,
                             optimal=gt.get("cost"), **{k: rec[k] for k in ("tool_calls", "rejected", "wrong_tool", "bad_args", "format_errors")}))
            traces.append(format_trace(rid, text, mode, trace, ok))
    # ---- tables
    print(f"\nPART D RESULTS: 20 dispatcher requests ({sum(direct[r]['feasible'] for r in direct)} feasible, "
          f"{sum(not direct[r]['feasible'] for r in direct)} infeasible by construction)\n{STAMP}")
    print(f"{'agent':<26}{'success':>18}{'tool calls':>14}{'rejected':>10}{'wrong tool':>11}{'bad args':>9}{'ignored constr.':>16}{'cost gap':>10}")
    summ = [("direct planner", 20, 1.0, 0, 0, 0, 0, 0.0)]
    print(f"{'direct planner':<26}{'100% [84,100]':>18}{1.0:>14.1f}{0:>10}{0:>11}{0:>9}{0:>16}{'0.0%':>10}")
    for mode in ("raw", "validated"):
        R = [r for r in rows if r["mode"] == mode]
        k = sum(r["success"] for r in R); p, lo, hi = wilson(k, len(R))
        tc, tci = mean_ci([r["tool_calls"] for r in R])
        gaps = [100 * (r["cost"] - r["optimal"]) / r["optimal"] for r in R if r["cost"] is not None and r["optimal"]]
        ign = sum(r["why"] == "ignored a constraint" for r in R)
        print(f"{'LLM ' + mode:<26}{f'{100*p:.0f}% [{100*lo:.0f},{100*hi:.0f}]':>18}{f'{tc:.1f}+-{tci:.1f}':>14}{sum(r['rejected'] for r in R):>10}"
              f"{sum(r['wrong_tool'] for r in R):>11}{sum(r['bad_args'] for r in R):>9}{ign:>16}{(f'{sum(gaps)/len(gaps):.1f}%' if gaps else '-'):>10}")
        summ.append((f"llm_{mode}", len(R), tc, sum(r["rejected"] for r in R), sum(r["wrong_tool"] for r in R),
                     sum(r["bad_args"] for r in R), ign, (sum(gaps) / len(gaps) if gaps else 0.0)))
    print("\nFAILURES (wrong tool / invented or malformed args / ignored constraint / other):")
    for r in rows:
        if not r["success"] or r["wrong_tool"] or r["bad_args"]:
            print(f"  [{r['mode']:<9}] request {r['rid']:>2} (level {r['level']}): "
                  f"{'FAIL: ' + r['why'] if not r['success'] else ('ok after repair' if r['mode'] == 'validated' else 'ok despite an invalid call')}"
                  f"{'; wrong tool x%d' % r['wrong_tool'] if r['wrong_tool'] else ''}{'; bad args x%d' % r['bad_args'] if r['bad_args'] else ''}")
    # ---- files
    import csv
    with open(os.path.join(RESULTS, "partD_results.csv"), "w", newline="") as f:
        f.write(f"# {STAMP}\n# backend={llm.name} model={llm.model}\n")
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(RESULTS, "partD_traces.txt"), "w", encoding="utf-8") as f:
        f.write(f"{STAMP}\nbackend={llm.name} model={llm.model}\nTrace lines map to the Sense-Reason-Act cycle.\n\n" + "\n\n".join(traces) + f"\n\n{STAMP}\n")
    print(f"\nSaved results/partD_results.csv and partD_traces.txt\n{STAMP}")


def ask(text):
    print(STAMP)
    llm = make_backend(); env = Env()
    for mode, validate in (("validated", True),):
        rec, trace = run_agent(env, text, llm, validate)
        print(format_trace(0, text, mode, trace, True)); print("plan:", env.last_plan)
    print(STAMP)


if __name__ == "__main__":
    if "--ask" in sys.argv:
        ask(sys.argv[sys.argv.index("--ask") + 1])
    else:
        run_part_d()