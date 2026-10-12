#!/usr/bin/env python3
"""
06_temporal_certainty_analysis.py

Temporal certainty analysis for Experiment B (CK+ full-sequence CNN-LSTM).

Input:
    ~/Desktop/FER_project/pretrained_evaluation/temporal_probabilities.csv

Expected columns:
    subject, sequence, true_label, predicted_label_final,
    frame_index, sequence_length, normalized_time,
    p_anger, p_contempt, p_disgust, p_fear,
    p_happy, p_sadness, p_surprise

Main definition:
    Trigger Time t* = argmax_t [p_t(y) - p_{t-1}(y)]
where y is the ground-truth sequence label.

Important:
- Trigger Time is computed for both correctly and incorrectly classified sequences,
  but final correctness is explicitly recorded.
- Because CK+ sequences have different lengths, normalized trigger time is used
  for across-sequence / across-emotion comparison.
- Intermediate probabilities are analysis signals from a model trained with
  sequence-level supervision at the final valid time step.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path.home() / "Desktop" / "FER_project"
INPUT = ROOT / "pretrained_evaluation" / "temporal_probabilities.csv"
OUT = ROOT / "temporal_certainty_analysis"
REP_OUT = OUT / "representative_sequences"

OUT.mkdir(parents=True, exist_ok=True)
REP_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
EMOTIONS = [
    "anger", "contempt", "disgust", "fear",
    "happy", "sadness", "surprise"
]
PROB_COLS = [f"p_{e}" for e in EMOTIONS]

# Common normalized time grid for averaging variable-length sequences.
GRID = np.linspace(0.0, 1.0, 101)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def check_columns(df):
    required = {
        "subject", "sequence", "true_label", "predicted_label_final",
        "frame_index", "sequence_length", "normalized_time",
        *PROB_COLS
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "temporal_probabilities.csv is missing columns:\n"
            + "\n".join(missing)
        )


def add_target_probability(g):
    """Return one sequence sorted in temporal order with p_true and delta_p_true."""
    g = g.sort_values("frame_index").copy()
    true_label = str(g["true_label"].iloc[0]).strip().lower()

    if true_label not in EMOTIONS:
        raise ValueError(f"Unknown true label: {true_label}")

    target_col = f"p_{true_label}"
    g["target_probability"] = g[target_col].astype(float)

    # Trigger is defined from t=2 onward because t=1 has no preceding frame.
    g["delta_target_probability"] = g["target_probability"].diff()
    return g


def sequence_statistics(g):
    g = add_target_probability(g)

    subject = g["subject"].iloc[0]
    sequence = g["sequence"].iloc[0]
    true_label = str(g["true_label"].iloc[0]).strip().lower()
    final_pred = str(g["predicted_label_final"].iloc[0]).strip().lower()
    correct = (true_label == final_pred)

    # Trigger Time: largest increase in the ground-truth emotion probability.
    valid_delta = g["delta_target_probability"].dropna()
    if len(valid_delta) > 0:
        trigger_idx = valid_delta.idxmax()
        trigger_frame = int(g.loc[trigger_idx, "frame_index"])
        trigger_norm = float(g.loc[trigger_idx, "normalized_time"])
        trigger_probability = float(g.loc[trigger_idx, "target_probability"])
        max_probability_increase = float(g.loc[trigger_idx, "delta_target_probability"])
    else:
        trigger_frame = np.nan
        trigger_norm = np.nan
        trigger_probability = np.nan
        max_probability_increase = np.nan

    # First frame at which the true emotion becomes the current argmax.
    probs = g[PROB_COLS].to_numpy()
    argmax_labels = np.array(EMOTIONS)[np.argmax(probs, axis=1)]
    true_mask = (argmax_labels == true_label)

    if true_mask.any():
        first_dom_pos = int(np.flatnonzero(true_mask)[0])
        first_dominant_frame = int(g.iloc[first_dom_pos]["frame_index"])
        first_dominant_norm = float(g.iloc[first_dom_pos]["normalized_time"])
    else:
        first_dominant_frame = np.nan
        first_dominant_norm = np.nan

    # Stable recognition time:
    # earliest time from which the true class remains argmax until the final frame.
    stable_pos = None
    for pos in range(len(g)):
        if np.all(true_mask[pos:]):
            stable_pos = pos
            break

    if stable_pos is not None:
        stable_frame = int(g.iloc[stable_pos]["frame_index"])
        stable_norm = float(g.iloc[stable_pos]["normalized_time"])
    else:
        stable_frame = np.nan
        stable_norm = np.nan

    final_target_probability = float(g["target_probability"].iloc[-1])
    peak_target_probability = float(g["target_probability"].max())

    return {
        "subject": subject,
        "sequence": sequence,
        "true_label": true_label,
        "predicted_label_final": final_pred,
        "correct_final": correct,
        "sequence_length": int(g["sequence_length"].iloc[0]),
        "trigger_frame": trigger_frame,
        "normalized_trigger_time": trigger_norm,
        "target_probability_at_trigger": trigger_probability,
        "max_target_probability_increase": max_probability_increase,
        "first_dominant_frame": first_dominant_frame,
        "normalized_first_dominant_time": first_dominant_norm,
        "stable_recognition_frame": stable_frame,
        "normalized_stable_recognition_time": stable_norm,
        "final_target_probability": final_target_probability,
        "peak_target_probability": peak_target_probability,
    }


def interpolate_sequence(g):
    """Interpolate all emotion probabilities to a common normalized-time grid."""
    g = g.sort_values("normalized_time")
    x = g["normalized_time"].to_numpy(dtype=float)

    rows = []
    for tau in GRID:
        row = {
            "subject": g["subject"].iloc[0],
            "sequence": g["sequence"].iloc[0],
            "true_label": str(g["true_label"].iloc[0]).strip().lower(),
            "predicted_label_final": str(g["predicted_label_final"].iloc[0]).strip().lower(),
            "correct_final": (
                str(g["true_label"].iloc[0]).strip().lower()
                == str(g["predicted_label_final"].iloc[0]).strip().lower()
            ),
            "normalized_time": tau,
        }
        for col in PROB_COLS:
            row[col] = np.interp(tau, x, g[col].to_numpy(dtype=float))
        rows.append(row)
    return rows


def save_trigger_plot(summary):
    """Box/scatter plot of normalized Trigger Time by emotion."""
    data = []
    labels = []
    for emotion in EMOTIONS:
        vals = summary.loc[
            summary["true_label"] == emotion, "normalized_trigger_time"
        ].dropna().to_numpy()
        data.append(vals)
        labels.append(emotion)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.boxplot(data, tick_labels=labels, showmeans=True)

    # Add individual observations so tiny CK+ classes are visible.
    for i, vals in enumerate(data, start=1):
        if len(vals):
            x = np.full(len(vals), i, dtype=float)
            ax.scatter(x, vals, s=28, zorder=3)

    ax.set_title("Normalized Trigger Time by Emotion")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Normalized Trigger Time")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUT / "trigger_time_by_emotion.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_mean_target_curves(interpolated):
    """
    Plot mean probability of the ground-truth emotion for each true emotion class.
    This is usually easier to interpret in a thesis than plotting all 7x7 curves.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for emotion in EMOTIONS:
        sub = interpolated[interpolated["true_label"] == emotion].copy()
        if sub.empty:
            continue

        mean_curve = (
            sub.groupby("normalized_time")[f"p_{emotion}"]
            .mean()
            .reindex(GRID)
        )
        ax.plot(GRID, mean_curve.to_numpy(), label=emotion)

    ax.set_title("Mean Target-Emotion Probability over Normalized Time")
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Mean Probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mean_probability_by_emotion.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_representative_curves(df, summary):
    """
    For each emotion, choose a correctly classified sequence whose normalized
    Trigger Time is closest to the median among correctly classified sequences.
    If no correct sequence exists, use the sequence closest to the class median
    and clearly mark it as misclassified.
    """
    chosen_rows = []

    for emotion in EMOTIONS:
        s = summary[summary["true_label"] == emotion].copy()
        if s.empty:
            continue

        candidates = s[s["correct_final"]].copy()
        selection_type = "correct"

        if candidates.empty:
            candidates = s.copy()
            selection_type = "misclassified_no_correct_example"

        median_trigger = candidates["normalized_trigger_time"].median()
        candidates["distance_to_median_trigger"] = (
            candidates["normalized_trigger_time"] - median_trigger
        ).abs()

        row = candidates.sort_values(
            ["distance_to_median_trigger", "subject", "sequence"]
        ).iloc[0]

        subject = row["subject"]
        sequence = row["sequence"]

        g = df[
            (df["subject"] == subject) &
            (df["sequence"] == sequence)
        ].sort_values("frame_index").copy()

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        for e in EMOTIONS:
            ax.plot(
                g["normalized_time"],
                g[f"p_{e}"],
                label=e,
                linewidth=2 if e == emotion else 1.2
            )

        if pd.notna(row["normalized_trigger_time"]):
            ax.axvline(
                row["normalized_trigger_time"],
                linestyle="--",
                linewidth=1.5,
                label="Trigger Time"
            )

        status = "Correct" if bool(row["correct_final"]) else "Incorrect"
        ax.set_title(
            f"{emotion.capitalize()} — Representative Sequence "
            f"({subject}/{sequence}, {status})"
        )
        ax.set_xlabel("Normalized Time")
        ax.set_ylabel("Probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()

        filename = f"{emotion}_representative.png"
        fig.savefig(
            REP_OUT / filename,
            dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )
        plt.close(fig)

        chosen_rows.append({
            "emotion": emotion,
            "subject": subject,
            "sequence": sequence,
            "correct_final": bool(row["correct_final"]),
            "predicted_label_final": row["predicted_label_final"],
            "normalized_trigger_time": row["normalized_trigger_time"],
            "selection_type": selection_type,
            "figure": filename,
        })

    pd.DataFrame(chosen_rows).to_csv(
        OUT / "representative_sequences.csv", index=False
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT}\n"
            "Run 05_evaluate_pretrained.py first."
        )

    df = pd.read_csv(INPUT)
    check_columns(df)

    # Normalize text labels.
    df["true_label"] = df["true_label"].astype(str).str.strip().str.lower()
    df["predicted_label_final"] = (
        df["predicted_label_final"].astype(str).str.strip().str.lower()
    )

    # Stable sort.
    df = df.sort_values(
        ["subject", "sequence", "frame_index"]
    ).reset_index(drop=True)

    seq_keys = ["subject", "sequence"]
    n_sequences = df.groupby(seq_keys).ngroups

    print("TEMPORAL CERTAINTY ANALYSIS")
    print("=" * 60)
    print(f"Input: {INPUT}")
    print(f"Rows: {len(df)}")
    print(f"Sequences: {n_sequences}")

    # --------------------------------------------------------------
    # 1. Per-sequence Trigger Time and recognition timing
    # --------------------------------------------------------------
    records = []
    for _, g in df.groupby(seq_keys, sort=True):
        records.append(sequence_statistics(g))

    summary = pd.DataFrame(records)
    summary.to_csv(OUT / "sequence_trigger_times.csv", index=False)

    # --------------------------------------------------------------
    # 2. Emotion-wise summary
    # --------------------------------------------------------------
    emotion_rows = []

    for emotion in EMOTIONS:
        s = summary[summary["true_label"] == emotion].copy()
        if s.empty:
            continue

        correct_s = s[s["correct_final"]]

        emotion_rows.append({
            "emotion": emotion,
            "n_sequences": len(s),
            "n_correct": int(s["correct_final"].sum()),
            "final_accuracy": float(s["correct_final"].mean()),
            "mean_trigger_frame": s["trigger_frame"].mean(),
            "mean_normalized_trigger_time": s["normalized_trigger_time"].mean(),
            "median_normalized_trigger_time": s["normalized_trigger_time"].median(),
            "std_normalized_trigger_time": s["normalized_trigger_time"].std(),
            "mean_final_target_probability": s["final_target_probability"].mean(),
            "mean_peak_target_probability": s["peak_target_probability"].mean(),
            "n_stably_recognized": int(
                s["stable_recognition_frame"].notna().sum()
            ),
            "mean_normalized_stable_recognition_time": (
                s["normalized_stable_recognition_time"].mean()
            ),
            # Correct-only values are useful because a failed final prediction
            # should not be described as successful recognition.
            "mean_normalized_trigger_time_correct_only": (
                correct_s["normalized_trigger_time"].mean()
                if len(correct_s) else np.nan
            ),
            "mean_normalized_stable_time_correct_only": (
                correct_s["normalized_stable_recognition_time"].mean()
                if len(correct_s) else np.nan
            ),
        })

    emotion_summary = pd.DataFrame(emotion_rows)
    emotion_summary.to_csv(
        OUT / "emotion_trigger_summary.csv", index=False
    )

    # --------------------------------------------------------------
    # 3. Interpolate variable-length sequences onto common time grid
    # --------------------------------------------------------------
    interpolated_rows = []
    for _, g in df.groupby(seq_keys, sort=True):
        interpolated_rows.extend(interpolate_sequence(g))

    interpolated = pd.DataFrame(interpolated_rows)
    interpolated.to_csv(
        OUT / "normalized_probability_curves.csv", index=False
    )

    # Also save class-wise mean probabilities for later thesis tables/plots.
    mean_rows = []
    for emotion in EMOTIONS:
        sub = interpolated[interpolated["true_label"] == emotion]
        if sub.empty:
            continue
        means = sub.groupby("normalized_time")[PROB_COLS].mean().reset_index()
        means.insert(0, "true_emotion", emotion)
        mean_rows.append(means)

    mean_curves = pd.concat(mean_rows, ignore_index=True)
    mean_curves.to_csv(
        OUT / "emotion_mean_probability_curves.csv", index=False
    )

    # --------------------------------------------------------------
    # 4. Figures
    # --------------------------------------------------------------
    save_trigger_plot(summary)
    save_mean_target_curves(interpolated)
    save_representative_curves(df, summary)

    # --------------------------------------------------------------
    # 5. Human-readable summary
    # --------------------------------------------------------------
    overall_accuracy = summary["correct_final"].mean()

    lines = [
        "TEMPORAL CERTAINTY ANALYSIS SUMMARY",
        "=" * 60,
        f"Sequences analyzed: {len(summary)}",
        f"Final sequence accuracy: {overall_accuracy * 100:.2f}%",
        "",
        "Trigger Time definition:",
        "t* = argmax_t [p_t(y) - p_(t-1)(y)]",
        "where y is the ground-truth emotion label.",
        "",
        "Normalized time:",
        "tau = (t - 1) / (T - 1)",
        "",
        "Important interpretation note:",
        "Intermediate probabilities are analyzed as temporal model outputs,",
        "but the model was trained with sequence-level supervision at the",
        "final valid time step rather than direct supervision at every frame.",
        "",
        "Emotion-wise results:",
    ]

    for _, r in emotion_summary.iterrows():
        stable = r["mean_normalized_stable_recognition_time"]
        stable_text = "NA" if pd.isna(stable) else f"{stable:.3f}"
        lines.append(
            f"{r['emotion']:>8s}: "
            f"N={int(r['n_sequences']):2d}, "
            f"accuracy={r['final_accuracy']*100:6.2f}%, "
            f"mean trigger tau={r['mean_normalized_trigger_time']:.3f}, "
            f"mean stable tau={stable_text}"
        )

    (OUT / "analysis_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\nEmotion-wise summary:")
    print(
        emotion_summary[
            [
                "emotion",
                "n_sequences",
                "final_accuracy",
                "mean_normalized_trigger_time",
                "mean_normalized_stable_recognition_time",
            ]
        ].to_string(index=False)
    )

    print("\nSaved to:", OUT)
    print("\nMain outputs:")
    print(" - sequence_trigger_times.csv")
    print(" - emotion_trigger_summary.csv")
    print(" - normalized_probability_curves.csv")
    print(" - emotion_mean_probability_curves.csv")
    print(" - representative_sequences.csv")
    print(" - trigger_time_by_emotion.png")
    print(" - mean_probability_by_emotion.png")
    print(" - representative_sequences/*.png")
    print(" - analysis_summary.txt")


if __name__ == "__main__":
    main()
