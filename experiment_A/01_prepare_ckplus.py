import csv, random
from collections import Counter
from pathlib import Path
from statistics import mean

SEED = 42
PROJECT = Path.home() / "Desktop" / "FER_project"
ROOT = PROJECT / "CK+ dataset (Original Ver.)"
IMAGE_ROOT = ROOT / "cohn-kanade-images"
EMOTION_ROOT = ROOT / "Emotion"
OUT = PROJECT / "ckplus_full_pipeline_data"

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
LABEL_MAP = {0:"neutral",1:"anger",2:"contempt",3:"disgust",4:"fear",5:"happy",6:"sadness",7:"surprise"}
CLASSES = ["anger","contempt","disgust","fear","happy","sadness","surprise"]
FIELDS = ["subject","sequence","label_code","label","num_frames","image_dir","label_file"]

def read_label(path):
    value = float(path.read_text(encoding="utf-8").strip().split()[0])
    code = int(round(value))
    if code not in LABEL_MAP:
        raise ValueError(f"Unknown label: {value}")
    return code, LABEL_MAP[code]

def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

def class_counts(rows):
    c = Counter(r["label"] for r in rows)
    return {k: c.get(k, 0) for k in CLASSES}

def main():
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(IMAGE_ROOT)
    if not EMOTION_ROOT.exists():
        raise FileNotFoundError(EMOTION_ROOT)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []

    # IMPORTANT: scan Emotion, not all image folders.
    # Therefore unlabeled CK+ sequences are automatically excluded.
    label_files = sorted(EMOTION_ROOT.rglob("*_emotion.txt"))

    for label_file in label_files:
        try:
            rel = label_file.relative_to(EMOTION_ROOT)
            subject, sequence = rel.parts[0], rel.parts[1]
            code, label = read_label(label_file)

            # Current model is seven-class FER.
            if label == "neutral":
                continue

            image_dir = IMAGE_ROOT / subject / sequence
            if not image_dir.exists():
                raise FileNotFoundError(f"Missing image folder: {image_dir}")

            frames = sorted(
                p for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in EXTS
            )
            if not frames:
                raise RuntimeError("No frames found")

            rows.append({
                "subject": subject,
                "sequence": sequence,
                "label_code": code,
                "label": label,
                "num_frames": len(frames),
                "image_dir": str(image_dir),
                "label_file": str(label_file),
            })
        except Exception as e:
            skipped.append((str(label_file), str(e)))

    rows.sort(key=lambda r: (r["subject"], r["sequence"]))
    if not rows:
        raise RuntimeError("No usable labelled CK+ sequences found.")

    # Subject-independent 70/15/15 split.
    subjects = sorted({r["subject"] for r in rows})
    rng = random.Random(SEED)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = max(1, min(round(n * 0.70), n - 2))
    n_val = max(1, min(round(n * 0.15), n - n_train - 1))

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train+n_val])
    test_subjects = set(subjects[n_train+n_val:])

    train = [r for r in rows if r["subject"] in train_subjects]
    val = [r for r in rows if r["subject"] in val_subjects]
    test = [r for r in rows if r["subject"] in test_subjects]

    for name, data in [("all", rows), ("train", train), ("val", val), ("test", test)]:
        save_csv(OUT / f"{name}_sequences.csv", data)

    lengths = [r["num_frames"] for r in rows]
    lines = [
        "CK+ FULL-SEQUENCE SUMMARY",
        "=" * 50,
        f"Emotion label files found: {len(label_files)}",
        f"Usable labelled sequences: {len(rows)}",
        f"Unique subjects: {len(subjects)}",
        f"Sequence length: min={min(lengths)}, max={max(lengths)}, mean={mean(lengths):.2f}",
        ""
    ]

    for name, data, subs in [
        ("ALL", rows, set(subjects)),
        ("TRAIN", train, train_subjects),
        ("VAL", val, val_subjects),
        ("TEST", test, test_subjects),
    ]:
        lines.append(f"{name}: {len(data)} sequences / {len(subs)} subjects")
        for emotion, count in class_counts(data).items():
            lines.append(f"  {emotion:<10}: {count}")

    if skipped:
        lines += ["", "SKIPPED / PROBLEMATIC"]
        lines += [f"{p} -> {reason}" for p, reason in skipped]

    summary = "\n".join(lines)
    (OUT / "dataset_summary.txt").write_text(summary, encoding="utf-8")
    (OUT / "split_subjects.txt").write_text(
        "TRAIN\n" + "\n".join(sorted(train_subjects)) +
        "\n\nVAL\n" + "\n".join(sorted(val_subjects)) +
        "\n\nTEST\n" + "\n".join(sorted(test_subjects)) + "\n",
        encoding="utf-8"
    )

    print(summary)
    print("\nSaved to:", OUT)
    print("\nIMPORTANT: keep these split CSV files fixed for later experiments.")

if __name__ == "__main__":
    main()
