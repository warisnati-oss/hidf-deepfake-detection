"""df_e02_zero_shot (VLM part): ask Qwen2.5-VL-7B-Instruct whether a face is a deepfake.

Runs locally with MLX (8-bit weights). For each image we read the log-probability of the
first answer token: score = log p("Yes") - log p("No"), so AUC can be computed.
At ~3 s per image the model runs on subsets, stratified by label x c/f prefix:
    500 val images  -> choose the prompt and the decision threshold
    1,000 test images -> scored once with the chosen prompt

Output: experiments/df_e02_zero_shot/{vlm_val_scores.csv, vlm_test_scores.csv, vlm_results.csv}
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from sklearn.metrics import roc_auc_score

from df_e02_zero_shot import best_threshold, metrics

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUT = PROJECT / "experiments" / "df_e02_zero_shot"
MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
N_VAL, N_TEST = 500, 1000
SEED = 0

PROMPTS = {
    "direct": "Is this photo of a face a deepfake, i.e. was the face digitally swapped or manipulated? "
              "Answer Yes or No.",
    "forensic": "Inspect this face carefully: skin texture, the boundary between face and hair or neck, "
                "lighting consistency and blending artifacts. Is the face a deepfake face swap? "
                "Answer Yes or No.",
}


def stratified(rows: pd.DataFrame, n: int) -> pd.DataFrame:
    frac = n / len(rows)
    return rows.groupby(["label", "prefix"]).sample(frac=frac, random_state=SEED)


def score(model, proc, prompt: str, paths: list[str], yes: int, no: int, tag: str) -> np.ndarray:
    out = []
    for k, p in enumerate(paths):
        r = next(stream_generate(model, proc, prompt, image=[str(DATA / p)], max_tokens=1, temperature=0.0))
        lp = np.array(r.logprobs.astype(mx.float32)).ravel()
        out.append(lp[yes] - lp[no])
        if k % 100 == 0:
            print(f"  {tag}: {k}/{len(paths)}", flush=True)
    return np.array(out)


def main():
    df = pd.read_csv(DATA / "index.csv", low_memory=False)
    df = df[~df.near_dup]  # photos that also appear in train under another id
    val = stratified(df[df.split == "val"], N_VAL)
    test = stratified(df[df.split == "test"], N_TEST)
    model, proc = load(MODEL)
    yes = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no = proc.tokenizer.encode("No", add_special_tokens=False)[0]
    chat = {k: apply_chat_template(proc, model.config, q, num_images=1) for k, q in PROMPTS.items()}

    y_val = (val.label == "fake").to_numpy().astype(int)
    val_scores = {k: score(model, proc, c, val.cache_path.to_list(), yes, no, f"val {k}") for k, c in chat.items()}
    val_auc = {k: roc_auc_score(y_val, s) for k, s in val_scores.items()}
    best = max(val_auc, key=val_auc.get)
    thr = best_threshold(y_val, val_scores[best])
    pd.DataFrame({"cache_path": val.cache_path.to_numpy(), "label": val.label.to_numpy(),
                  **{f"score_{k}": s for k, s in val_scores.items()}}).to_csv(OUT / "vlm_val_scores.csv", index=False)
    print("val AUC per prompt:", {k: round(v, 4) for k, v in val_auc.items()}, flush=True)

    y_test = (test.label == "fake").to_numpy().astype(int)
    s_test = score(model, proc, chat[best], test.cache_path.to_list(), yes, no, f"test {best}")
    pd.DataFrame({"cache_path": test.cache_path.to_numpy(), "label": test.label.to_numpy(),
                  "prefix": test.prefix.to_numpy(), "score": s_test}).to_csv(OUT / "vlm_test_scores.csv", index=False)

    res = {"model": "qwen2.5_vl_7b", "view": "full", "prompts": best, "threshold": thr,
           "val_auc": val_auc[best], "n_test": len(test),
           **{f"test_{k}": v for k, v in metrics(y_test, s_test, thr).items()}}
    pd.DataFrame([res]).round(4).to_csv(OUT / "vlm_results.csv", index=False)
    print(res)


if __name__ == "__main__":
    main()
