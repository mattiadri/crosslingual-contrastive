#!/usr/bin/env python3
import subprocess, shlex, json, time, os, sys
from pathlib import Path
import pandas as pd

OUT_DIR      = "webdataset"
TRAIN_SPLIT  = "full"
EVAL_SPLITS  = ""  # empty => internal holdout
LANGS        = "ar,de,en,es,fr,it,ja,pt,zh"
DEVICE       = "cuda"

# -------------------------
# 1) Wide sweep for LINEAR
# -------------------------
TRIALS_LINEAR = [
    ("lin_baseline",
     dict(head="linear", lr="3e-3", wd="5e-4", temp="0.10", warmup="120",
          prox_id="1e-3", ortho="0", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("lin_light_ortho",
     dict(head="linear", lr="3e-3", wd="5e-4", temp="0.09", warmup="90",
          prox_id="1e-3", ortho="1e-4", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("lin_no_reg_high_temp",
     dict(head="linear", lr="3e-3", wd="3e-4", temp="0.12", warmup="90",
          prox_id="0", ortho="0", max_grad_norm="0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("lin_stronger_reg",
     dict(head="linear", lr="1e-3", wd="7e-4", temp="0.10", warmup="150",
          prox_id="3e-3", ortho="0", max_grad_norm="1.0",
          epochs="16", steps_per_epoch="10", batch_per_lang="32")),
    # add a few more if you want
    ("lin_lr_1e3_temp_007",
     dict(head="linear", lr="1e-3", wd="5e-4", temp="0.07", warmup="120",
          prox_id="1e-3", ortho="0", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("lin_lr_3e4_temp_010",
     dict(head="linear", lr="3e-4", wd="5e-4", temp="0.10", warmup="120",
          prox_id="1e-3", ortho="0", max_grad_norm="1.0",
          epochs="24", steps_per_epoch="12", batch_per_lang="32")),
]

# -------------------------
# 2) MLP tries, seeded by best LINEAR
# We'll fill these after we know best linear
# -------------------------
MLP_GRID = [
    dict(mlp_hidden="512",  mlp_dropout="0.0"),
    dict(mlp_hidden="1024", mlp_dropout="0.0"),
    dict(mlp_hidden="2048", mlp_dropout="0.0"),
    dict(mlp_hidden="1024", mlp_dropout="0.1"),
]

def build_cmd(args: dict) -> str:
    cmd = [
        sys.executable, "align_text_to_image.py",
        "--out_dir", OUT_DIR,
        "--train_split", TRAIN_SPLIT,
        "--eval_splits", EVAL_SPLITS,
        "--langs", LANGS,
        "--device", DEVICE,
        "--save_report",
        "--epochs", args["epochs"],
        "--steps_per_epoch", args["steps_per_epoch"],
        "--batch_per_lang", args["batch_per_lang"],
        "--lr", args["lr"],
        "--wd", args["wd"],
        "--temp", args["temp"],
        "--warmup", args["warmup"],
        "--head", args.get("head", "linear"),
    ]
    if "prox_id" in args:        cmd += ["--prox_id", str(args["prox_id"])]
    if "ortho" in args:          cmd += ["--ortho", str(args["ortho"])]
    if "max_grad_norm" in args:  cmd += ["--max_grad_norm", str(args["max_grad_norm"])]

    # mlp-only flags
    if args.get("head") == "mlp":
        if "mlp_hidden" in args:  cmd += ["--mlp_hidden", str(args["mlp_hidden"])]
        if "mlp_dropout" in args: cmd += ["--mlp_dropout", str(args["mlp_dropout"])]

    return " ".join(shlex.quote(c) for c in cmd)

def _base_name(train_split: str, langs: str) -> str:
    return f"{train_split}_{'-'.join(langs.split(','))}"

def report_path(head: str) -> Path:
    # linear keeps old naming; mlp appends _mlp
    base = _base_name(TRAIN_SPLIT, LANGS)
    if head == "linear":
        fn = f"report_{base}.json"
    else:
        fn = f"report_{base}_mlp.json"
    return Path(OUT_DIR) / "alignment" / fn

def weight_path(head: str) -> Path:
    base = _base_name(TRAIN_SPLIT, LANGS)
    if head == "linear":
        fn = f"W_{base}.pt"
    else:
        fn = f"W_{base}_mlp.pt"
    return Path(OUT_DIR) / "alignment" / fn

def read_core_metrics(report_json: Path) -> dict:
    with open(report_json, "r", encoding="utf-8") as f:
        rpt = json.load(f)
    # pick holdout block if present, else train
    blocks = rpt["results"]
    pick = None
    for b in blocks:
        if b.get("kind") == "holdout":
            pick = b
            break
    if pick is None:
        pick = next(b for b in blocks if b.get("kind") == "train")

    mac = pick["macro_avg"]
    return {
        "block": pick["kind"] if "split" not in pick else f"{pick['kind']}:{pick['split']}",
        "R@1_before": mac["before"]["R@1"],
        "R@1_after":  mac["after"]["R@1"],
        "dR1":        mac["delta"]["R@1"],
        "MRR_before": mac["before"]["MRR"],
        "MRR_after":  mac["after"]["MRR"],
        "dMRR":       mac["delta"]["MRR"],
        "n_pairs":    pick["n_pairs"],
    }

def run_trial(run_root: Path, tag: str, args: dict) -> dict:
    head = args.get("head", "linear")
    print(f"\n=== Running: {tag} [{head}] ===")
    log_path = run_root / f"{tag}.log"
    cmd = build_cmd(args)
    print(cmd)
    with open(log_path, "w") as lf:
        subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT)

    rsrc = report_path(head)
    wsrc = weight_path(head)

    if not rsrc.exists():
        print(f"[WARN] Report not found: {rsrc}")
        return {"trial": tag, "head": head, "ok": False}

    # move to avoid overwrite
    rdst = run_root / f"{tag}.report.json"
    os.replace(rsrc, rdst)

    if wsrc.exists():
        wdst = run_root / f"{tag}.W.pt"
        os.replace(wsrc, wdst)
    else:
        wdst = None

    core = read_core_metrics(rdst)
    core.update({"trial": tag, "head": head, "ok": True, "report": str(rdst), "weights": (str(wdst) if wdst else "")})
    return core

def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(f"sweep_runs/{ts}")
    run_root.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Phase 1: LINEAR sweep
    # -------------------------
    linear_rows = []
    for tag, args in TRIALS_LINEAR:
        linear_rows.append(run_trial(run_root, tag, args))
    df_lin = pd.DataFrame([r for r in linear_rows if r.get("ok")])
    if df_lin.empty:
        print("No linear results collected.")
        return

    df_lin = df_lin.sort_values(["dR1", "dMRR"], ascending=[False, False]).reset_index(drop=True)
    best = df_lin.iloc[0].to_dict()
    print("\n=== Best LINEAR (by Δ R@1, then Δ MRR) ===")
    print(best)

    # save linear summary
    df_lin.to_csv(run_root / "linear_summary.csv", index=False)

    # -------------------------
    # Phase 2: MLP mini-sweep seeded by best linear hyperparams
    # -------------------------
    base_args = None
    # recover args for best trial from TRIALS_LINEAR
    best_name = best["trial"]
    for tag, args in TRIALS_LINEAR:
        if tag == best_name:
            base_args = dict(args)  # copy
            break
    assert base_args is not None

    mlp_trials = []
    for i, g in enumerate(MLP_GRID):
        a = dict(base_args)
        a["head"] = "mlp"
        # keep same lr/wd/temp/warmup/prox_id/max_grad_norm from best linear
        a["mlp_hidden"] = g["mlp_hidden"]
        a["mlp_dropout"] = g["mlp_dropout"]
        # ortho doesn't apply to mlp; keep but script should ignore or you can set 0
        a["ortho"] = "0"
        mlp_trials.append((f"mlp_from_best_{i}_h{g['mlp_hidden']}_d{g['mlp_dropout']}", a))

    mlp_rows = []
    for tag, args in mlp_trials:
        mlp_rows.append(run_trial(run_root, tag, args))
    df_mlp = pd.DataFrame([r for r in mlp_rows if r.get("ok")])
    df_mlp.to_csv(run_root / "mlp_summary.csv", index=False)

    # -------------------------
    # Final comparison table
    # -------------------------
    df_best_lin = df_lin.head(1).copy()
    df_best_lin["model"] = "best_linear"
    if not df_mlp.empty:
        df_mlp_sorted = df_mlp.sort_values(["dR1", "dMRR"], ascending=[False, False]).reset_index(drop=True)
        df_best_mlp = df_mlp_sorted.head(1).copy()
        df_best_mlp["model"] = "best_mlp"
        final = pd.concat([df_best_lin, df_best_mlp], ignore_index=True)
    else:
        final = df_best_lin

    final_cols = ["model", "trial", "head", "block", "n_pairs", "R@1_before", "R@1_after", "dR1", "MRR_before", "MRR_after", "dMRR", "report", "weights"]
    final = final.loc[:, final_cols]
    final.to_csv(run_root / "final_compare.csv", index=False)

    print("\n=== FINAL COMPARE ===")
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 50)
    print(final)

    print(f"\nSaved in: {run_root}")
    print("- linear_summary.csv")
    print("- mlp_summary.csv")
    print("- final_compare.csv")
    print("- logs + per-trial reports + weights")

if __name__ == "__main__":
    main()