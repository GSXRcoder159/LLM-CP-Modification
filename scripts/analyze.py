import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def display_dataframe_to_user(title, df):
    print(f"\n{title}:\n")
    print(df.to_string(index=False))

# File paths expected (uploaded by the user)
FILES = {
    "gpt-4o-mini_zero": "4o-mini-data/4o_mini-results.jsonl",          # zero‑shot
    "gpt-4o-mini_few":  "4o_mini-in_context-data/4o_mini-in_context-results.jsonl", # few‑shot
    "gpt-4o_few":       "4o-in_context-data/4o-in_context-results.jsonl"       # GPT‑4 few‑shot
}

frames = []
missing_files = []

for cfg, path in FILES.items():
    if not os.path.exists(path):
        missing_files.append(path)
        continue
    with open(path, "r", encoding="utf-8") as fh:
        data = [json.loads(line) for line in fh]
    df = pd.DataFrame(data)
    df["config"] = cfg
    frames.append(df)

if missing_files:
    print("Warning – the following result files were not found in /mnt/data and therefore were skipped:")
    for p in missing_files:
        print("  •", p)

if not frames:
    raise RuntimeError("No result files available. Please upload the *.jsonl files to /mnt/data before running this cell.")

results = pd.concat(frames, ignore_index=True)

# --- Guess which columns hold the evaluation flags ---
# We try common field names; adapt if necessary.
candidates_bool = {
    "syntax_ok": ["compiled_flag", "syntax_valid", "syntactic_valid"]
}
candidates_float = {
    "constraint_acc": ["constraint_accuracy", "declaration_accuracy", "constraint_acc"],
    "model_ok":  ["model_accuracy", "model_success", "full_model_success"],
    "solution_ok": ["solution_accuracy", "solution_ok", "solution_success"]
}

def pick_column(df, options):
    for col in options:
        if col in df.columns:
            return col
    raise ValueError(f"None of {options} found in DataFrame columns {list(df.columns)}")


# Map actual columns
syntax_col = pick_column(results, candidates_bool["syntax_ok"])
model_col  = pick_column(results, candidates_float["model_ok"])
sol_col    = pick_column(results, candidates_float["solution_ok"])
acc_col    = pick_column(results, candidates_float["constraint_acc"])

# --- Aggregate overall metrics per configuration ---
overall = (results
           .groupby("config")
           .agg(syntactic_valid=(syntax_col, "mean"),
                constraint_accuracy=(acc_col, "mean"),
                full_model_success=(model_col, "mean"),
                solution_success=(sol_col, "mean"),
                cases=("config", "count"))
           .reset_index())

# Convert to percentages
for metric in ["syntactic_valid", "constraint_accuracy", "full_model_success", "solution_success"]:
    overall[metric] = (overall[metric] * 100).round(1) if metric == "syntactic_valid" else overall[metric].round(1)

display_dataframe_to_user("Overall LLM performance", overall)

# --- Success rate by modification type ---
if "modification_type" in results.columns:
    by_type = (results
               .groupby(["config", "modification_type"])[model_col]
               .mean()
               .round(1)
               .unstack("config")
               .sort_index())

    display_dataframe_to_user("Full‑model success by modification type (percent)", by_type)

# --- Success rate by difficulty ---
if "difficulty" in results.columns:
    by_diff = (results
               .groupby(["config", "difficulty"])[model_col]
               .mean()
               .round(1)
               .unstack("config")
               .loc[["easy", "moderate", "complex"]])

    display_dataframe_to_user("Full‑model success by difficulty (percent)", by_diff)

# --- Plot 1: overall full‑model success ---
fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.bar(overall["config"], overall["full_model_success"])
ax1.set_ylim(0, 100)
ax1.set_ylabel("Full‑model success (%)")
ax1.set_title("Overall success rate by LLM configuration")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# --- Plot 2: success by modification type ---
if "modification_type" in results.columns:
    fig2, ax2 = plt.subplots(figsize=(8,5))
    for cfg in by_type.columns:
        ax2.plot(by_type.index, by_type[cfg], marker="o", label=cfg)
    ax2.set_ylabel("Full‑model success (%)")
    ax2.set_title("Success per modification type")
    ax2.set_xticklabels(by_type.index, rotation=30, ha="right")
    ax2.set_ylim(0, 100)
    ax2.legend()
    plt.tight_layout()
    plt.show()
