import csv, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms, models
import matplotlib.pyplot as plt

SEED = 42
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
HIDDEN = 256

PROJECT = Path.home() / "Desktop" / "FER_project"
DATA_DIR = PROJECT / "ckplus_full_pipeline_data"
TRAIN_CSV = DATA_DIR / "train_sequences.csv"
VAL_CSV = DATA_DIR / "val_sequences.csv"
MODEL_PATH = PROJECT / "cnn_lstm_ckplus_full.pth"
OUT = PROJECT / "full_sequence_training"

CLASSES = ["anger","contempt","disgust","fear","happy","sadness","surprise"]
CLASS_TO_IDX = {x:i for i,x in enumerate(CLASSES)}
EXTS = {".png",".jpg",".jpeg",".bmp"}

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class CKPlusDataset(Dataset):
    def __init__(self, csv_path):
        with Path(csv_path).open(encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        folder = Path(r["image_dir"])
        paths = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in EXTS)

        frames = []
        for p in paths:
            with Image.open(p) as im:
                frames.append(transform(im.convert("RGB")))

        return torch.stack(frames), len(frames), CLASS_TO_IDX[r["label"]]

def collate_fn(batch):
    lengths = torch.tensor([x[1] for x in batch], dtype=torch.long)
    labels = torch.tensor([x[2] for x in batch], dtype=torch.long)
    max_len = int(lengths.max())
    _, c, h, w = batch[0][0].shape

    padded = torch.zeros(len(batch), max_len, c, h, w)
    for i, (seq, length, _) in enumerate(batch):
        padded[i, :length] = seq
    return padded, lengths, labels

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(512, HIDDEN, batch_first=True)
        self.classifier = nn.Linear(HIDDEN, len(CLASSES))

    def forward(self, x, lengths):
        b, t, c, h, w = x.shape
        feat = self.cnn(x.reshape(b*t,c,h,w)).flatten(1).reshape(b,t,512)

        packed = pack_padded_sequence(
            feat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        # Last VALID LSTM state for each sequence.
        return self.classifier(hidden[-1])

def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0

    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        if training:
            loss.backward()
            optimizer.step()

        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()

    return loss_sum / total, correct / total

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print("Using device:", device)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError("Run 01_prepare_ckplus.py first.")

    OUT.mkdir(parents=True, exist_ok=True)

    train_ds = CKPlusDataset(TRAIN_CSV)
    val_ds = CKPlusDataset(VAL_CSV)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    model = CNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    best_acc = -1
    best_epoch = 0

    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, device
            )

        history.append([epoch,train_loss,train_acc,val_loss,val_acc])

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train acc={train_acc*100:.2f}% | "
            f"val acc={val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
            print("  -> saved best model")

    with (OUT/"training_history.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["epoch","train_loss","train_accuracy","val_loss","val_accuracy"])
        w.writerows(history)

    epochs=[r[0] for r in history]

    plt.figure(figsize=(8,5))
    plt.plot(epochs,[r[1] for r in history],label="Train")
    plt.plot(epochs,[r[3] for r in history],label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT/"training_loss.png",dpi=300); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(epochs,[r[2] for r in history],label="Train")
    plt.plot(epochs,[r[4] for r in history],label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.ylim(0,1)
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT/"training_accuracy.png",dpi=300); plt.close()

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_acc*100:.2f}%")
    print("Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
