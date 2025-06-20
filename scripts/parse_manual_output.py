import re, glob, json, os, csv
import pandas as pd

JSONL_DIR = os.path.join("data", "jsonl", "manual")
# RESULT_CSV = os.path.join("data", "results.csv")
RESULT_JSONL = os.path.join("data", "manual_results.jsonl")
DATA_DIR     = "data"

def extract_block(pattern, text, group=1, flags=re.S):
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else ""

def main():
    rows = []
    for out_path in sorted(glob.glob(os.path.join(JSONL_DIR, "*.out"))):
        ex_id = os.path.basename(out_path).replace(".out", "")
        # load example to get metadata + ground_truth
        with open(os.path.join(JSONL_DIR, f"{ex_id}.jsonl")) as f:
            ex = json.loads(f.read())
        raw = open(out_path).read()

        # predicted model = first ```python…```
        pred_model = extract_block(r"```(?:python)?\n(.*?)```", raw)

        # metrics
        sol_acc = extract_block(r"Solution accuracy:\s*([\d\.]+)%", raw)
        cons_acc = extract_block(r"Constraint accuracy:\s*([\d\.]+)%", raw)
        mod_acc = extract_block(r"Model accuracy:\s*([\d\.]+)%", raw)
        avg_unc = extract_block(r"Average unchanged constraints:\s*([\d\.]+)", raw)
        ws, ts = extract_block(r"Wrong solutions:\s*(\d+),\s*total:\s*(\d+)", raw, group=0), None
        wm, tm = extract_block(r"Wrong models:\s*(\d+),\s*total:\s*(\d+)", raw, group=0), None
        # parse each separately
        ws_num, ts_num = re.search(r"Wrong solutions:\s*(\d+), total:\s*(\d+)", raw).groups()
        wc_num, tc_num = re.search(r"Wrong constraints:\s*(\d+), total:\s*(\d+)", raw).groups()
        wm_num, tm_num = re.search(r"Wrong models:\s*(\d+), total:\s*(\d+)", raw).groups()
        es_num = re.search(r"Errors: solution-level:\s*(\d+)", raw).group(1)
        ec_num = re.search(r"constraint-level:\s*(\d+)", raw).group(1)
        em_num = re.search(r"model-level:\s*(\d+)", raw).group(1)

        # num_vars_affected = |#intvar in ground_truth − #intvar in predicted|
        ovars = len(re.findall(r"\bintvar\(", ex[ex_id]["existing_model"]))
        pvars = len(re.findall(r"\bintvar\(", pred_model))
        num_vars_affected = abs(ovars - pvars)

        compiled_flag = (es_num == "0" and ec_num == "0" and em_num == "0")

        # read the modification subtype and difficulty from the ex_id
        # ex_id = APLAI_EX#S#_<subtype>_<difficulty>
        # e.g. APLAI_EX1S1_0_0
        
        # subtype can also contain a "_" in the name, e.g. "constraint_change"
        subtype = "_".join(ex_id.split("_")[2:-1])
        difficulty = ex_id.split("_")[-1].split(".")[0]
        modification_main = "structural" if subtype in ["constraint_addition", "constraint_removal", "constraint_change", "variable_addition", "variable_removal"] else "parametric"

        r = {
            "example_id": ex_id,
            "modification_main": modification_main,
            "subtype": subtype,
            "difficulty": difficulty,
            "modification_request": ex[ex_id]["modification_request"],
            "ground_truth_model": ex[ex_id]["modified_cpmpy_code"],
            "predicted_model": pred_model,
            "num_vars_affected": num_vars_affected,
            "compiled_flag": compiled_flag,
            "solution_accuracy": float(sol_acc),
            "constraint_accuracy": float(cons_acc),
            "model_accuracy": float(mod_acc),
            "avg_unchanged_constraints": float(avg_unc),
            "wrong_solutions": int(ws_num),
            "total_solutions": int(ts_num),
            "wrong_constraints": int(wc_num),
            "total_constraints": int(tc_num),
            "wrong_models": int(wm_num),
            "total_models": int(tm_num),
            "errors_solution": int(es_num),
            "errors_constraint": int(ec_num),
            "errors_model": int(em_num),
        }
        rows.append(r)
    
        ex_dir = os.path.join(DATA_DIR, ex_id)
        os.makedirs(ex_dir, exist_ok=True)

        metrics_fields = [
            "example_id","modification_main","subtype","difficulty",
            "num_vars_affected","compiled_flag",
            "solution_accuracy","constraint_accuracy","model_accuracy",
            "avg_unchanged_constraints",
            "wrong_solutions","total_solutions",
            "wrong_constraints","total_constraints",
            "wrong_models","total_models",
            "errors_solution","errors_constraint","errors_model"
        ]
        csv_path = os.path.join(ex_dir, "results.csv")
        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=metrics_fields)
            writer.writeheader()
            writer.writerow({k: r[k] for k in metrics_fields})

        with open(os.path.join(ex_dir, "modification_request.txt"), "w") as tf:
            tf.write(r["modification_request"])
        with open(os.path.join(ex_dir, "ground_truth_model.py"), "w") as tf:
            tf.write(r["ground_truth_model"].replace("```", ""))
        with open(os.path.join(ex_dir, "predicted_model.py"), "w") as tf:
            tf.write(r["predicted_model"].replace("```", ""))

    with open(RESULT_JSONL, "w") as outf:
        for r in rows:
            outf.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows to {RESULT_JSONL}")

if __name__ == "__main__":
    main()
