# scripts/run_pipeline.py
import subprocess, glob, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="data/jsonl/auto",
                    help="which jsonl folder to run")
parser.add_argument("--main", default="python main.py",
                    help="command to invoke your pipeline")
args = parser.parse_args()

JSONL_DIR = args.dir
OUT_GLOB = os.path.join(JSONL_DIR, "*.out")

# clean up old outputs
for old in glob.glob(OUT_GLOB):
    os.remove(old)

for path in sorted(glob.glob(os.path.join(JSONL_DIR, "*.jsonl"))):
    ex_id = os.path.splitext(os.path.basename(path))[0]
    out_file = os.path.join(JSONL_DIR, f"{ex_id}.out")
    cmd = f"{args.main} --input {path}"
    print(f"▶ {cmd}")
    with open(out_file, "w") as wf:
        proc = subprocess.run(cmd.split(), stdout=wf, stderr=subprocess.STDOUT)
    print(f"  ↳ exit {proc.returncode}")
