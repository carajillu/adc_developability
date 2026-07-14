import numpy as np
import pandas as pd


def evaluate_unmask_df(df, model_name):
    """
    Evaluate top-5 MLM predictions stored in a dataframe.

    Required columns:
        masked_residue
        pred_0 ... pred_4
        score_0 ... score_4

    Returns
    -------
    dict
        {
            "n_masked_positions",
            "top1_accuracy",
            "top5_accuracy", # float
            "mean_true_probability", # float if present in top5, NaN if not
            "MRR"
        }
    """

    true_probs = []
    reciprocal_ranks = []

    for _, row in df.iterrows():

        true_res = row["masked_residue"]

        rank = np.inf
        prob = np.nan

        for k in range(5):
            if row[f"pred_{k}"] == true_res:
                rank = k + 1
                prob = row[f"score_{k}"]
                break

        true_probs.append(prob)

        if np.isfinite(rank):
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    true_probs = np.array(true_probs)

    results = {
        "model_name": model_name,
        "n_masked_positions": len(df),
        "top1_accuracy": np.mean(
            df["pred_0"] == df["masked_residue"]
        ),
        "top1_accuracy_std": np.std(
            df["pred_0"] == df["masked_residue"]
        ),
        "top5_accuracy": np.mean(
            df[[f"pred_{i}" for i in range(5)]]
            .eq(df["masked_residue"], axis=0)
            .any(axis=1)
        ),
        "top5_accuracy_std": np.std(
            df[[f"pred_{i}" for i in range(5)]]
            .eq(df["masked_residue"], axis=0)
            .any(axis=1)
        ),
        "mean_true_probability": np.nanmean(true_probs),
        "std_true_probability": np.nanstd(true_probs),
        "MRR": np.mean(reciprocal_ranks),
        #"true_probabilities": true_probs,  # optional, useful for inspection
    }
    return results