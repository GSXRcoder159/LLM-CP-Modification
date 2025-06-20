# scripts/rerun_manual.py
import os, json, jsonlines
from shutil import copyfile
from pathlib import Path
from subprocess import check_call

# paths
TRAIN = Path("../data/modification/APLAI_course/train.jsonl")
MANUAL_DIR = Path("data/jsonl/manual")
MANUAL_DIR.mkdir(parents=True, exist_ok=True)

try:
    with jsonlines.open(TRAIN) as reader:
        # Process each example in the file
        for i, line in enumerate(reader.iter()):
            for key, value in line.items():
                example = value
            out = MANUAL_DIR / f"{key}.jsonl"
            example["id"] = key
            # write single-line JSON
            with open(out, "w") as wf:
                # wf.write(json.dumps(example) + "\n")
                # write it in the same format as the original - {"<id>": { … example fields … }}
                wf.write(json.dumps({key: example}) + "\n")
except FileNotFoundError:
    print(f"File not found: {TRAIN}")
except json.JSONDecodeError:
    print(f"Error decoding JSON from {TRAIN}")
except jsonlines.jsonlines.InvalidLineError:
    print(f"Invalid line in {TRAIN}")
except jsonlines.jsonlines.InvalidJsonError:
    print(f"Invalid JSON in {TRAIN}")
except jsonlines.jsonlines.InvalidFormatError:  
    print(f"Invalid format in {TRAIN}")
except jsonlines.jsonlines.InvalidStateError:
    print(f"Invalid state in {TRAIN}")
except jsonlines.jsonlines.InvalidOperationError:
    print(f"Invalid operation in {TRAIN}")
except Exception as e:
    print(f"Error reading {TRAIN}: {e}")

# 2. rerun the pipeline on manual dir
print("→ Running pipeline on manual examples…")
check_call([
    "python", "run_pipeline.py",
    "--dir", str(MANUAL_DIR),
    "--main", "python ../main.py"
])
print("Done. Inspect outputs under data/jsonl/manual")
