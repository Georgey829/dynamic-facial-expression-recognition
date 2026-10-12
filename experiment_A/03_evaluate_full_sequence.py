import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

BATCH_SIZE = 8
HIDDEN = 256

PROJECT = Path.home() / "Desktop" / "FER_project"
DATA_DIR = PROJECT / "ckplus_full_pipeline_data"
TEST_CSV = DATA_DIR / "test_sequences.csv"
MODEL_PATH = PROJECT / "cnn_lstm_ckplus_full.pth"
OUT = PROJECT / "full_sequence_evaluation"

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

        frames=[]
        for p in paths:
            with Image.open(p) as im:
                frames.append(transform(im.convert("RGB")))

        return (
            torch.stack(frames), len(frames), CLASS_TO_IDX[r["label"]],
            r["subject"], r["sequence"], [p.name for p in paths]
        )

def collate_fn(batch):
    lengths=torch.tensor([x[1] for x in batch],dtype=torch.long)
    labels=torch.tensor([x[2] for x in batch],dtype=torch.long)
    max_len=int(lengths.max())
    _,c,h,w=batch[0][0].shape
    padded=torch.zeros(len(batch),max_len,c,h,w)

    for i,x in enumerate(batch):
        padded[i,:x[1]]=x[0]

    return (
        padded,lengths,labels,
        [x[3] for x in batch],
        [x[4] for x in batch],
        [x[5] for x in batch]
    )

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        resnet=models.resnet18(weights=None)
        self.cnn=nn.Sequential(*list(resnet.children())[:-1])
        self.lstm=nn.LSTM(512,HIDDEN,batch_first=True)
        self.classifier=nn.Linear(HIDDEN,len(CLASSES))

    def forward_all(self,x,lengths):
        b,t,c,h,w=x.shape
        feat=self.cnn(x.reshape(b*t,c,h,w)).flatten(1).reshape(b,t,512)

        packed=pack_padded_sequence(
            feat,lengths.cpu(),batch_first=True,enforce_sorted=False
        )
        packed_out,_=self.lstm(packed)
        lstm_out,_=pad_packed_sequence(
            packed_out,batch_first=True,total_length=t
        )

        step_logits=self.classifier(lstm_out)

        last_idx=lengths.to(x.device)-1
        batch_idx=torch.arange(b,device=x.device)
        final_logits=step_logits[batch_idx,last_idx]

        return final_logits,step_logits

def save_cm(cm,path):
    fig,ax=plt.subplots(figsize=(8,7))
    im=ax.imshow(cm)
    fig.colorbar(im,ax=ax)
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES,rotation=45,ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CNN-LSTM Confusion Matrix — Held-Out Test Subjects")

    threshold=cm.max()/2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                    color="white" if cm[i,j]>threshold else "black")

    fig.tight_layout()
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close(fig)

def main():
    device=get_device()
    print("Using device:",device)

    if not TEST_CSV.exists():
        raise FileNotFoundError("Run 01_prepare_ckplus.py first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run 02_train_full_sequence.py first.")

    OUT.mkdir(parents=True,exist_ok=True)

    ds=CKPlusDataset(TEST_CSV)
    loader=DataLoader(
        ds,batch_size=BATCH_SIZE,shuffle=False,
        collate_fn=collate_fn,num_workers=0
    )

    model=CNN_LSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
    model.eval()

    y_true,y_pred=[],[]
    sequence_rows=[]
    temporal_rows=[]

    with torch.no_grad():
        for x,lengths,y,subjects,sequences,frame_names in loader:
            final_logits,step_logits=model.forward_all(x.to(device),lengths)

            final_probs=torch.softmax(final_logits,dim=-1).cpu()
            step_probs=torch.softmax(step_logits,dim=-1).cpu()
            preds=final_probs.argmax(dim=1)

            for i in range(len(y)):
                true_idx=int(y[i])
                pred_idx=int(preds[i])
                n=int(lengths[i])

                y_true.append(true_idx)
                y_pred.append(pred_idx)

                sequence_rows.append([
                    subjects[i],sequences[i],n,
                    CLASSES[true_idx],CLASSES[pred_idx],
                    int(true_idx==pred_idx),
                    float(final_probs[i,pred_idx]),
                    float(final_probs[i,true_idx])
                ])

                for t in range(n):
                    probs=step_probs[i,t]
                    step_pred=int(probs.argmax())
                    normalized_time=0.0 if n==1 else t/(n-1)

                    temporal_rows.append([
                        subjects[i],sequences[i],CLASSES[true_idx],
                        t+1,frame_names[i][t],normalized_time,
                        *[float(v) for v in probs],
                        CLASSES[step_pred],float(probs[step_pred])
                    ])

    labels=list(range(len(CLASSES)))
    accuracy=accuracy_score(y_true,y_pred)

    macro_p,macro_r,macro_f1,_=precision_recall_fscore_support(
        y_true,y_pred,labels=labels,average="macro",zero_division=0
    )
    weighted_p,weighted_r,weighted_f1,_=precision_recall_fscore_support(
        y_true,y_pred,labels=labels,average="weighted",zero_division=0
    )

    with (OUT/"overall_metrics.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["metric","value"])
        w.writerows([
            ["accuracy",accuracy],
            ["macro_precision",macro_p],
            ["macro_recall",macro_r],
            ["macro_f1",macro_f1],
            ["weighted_precision",weighted_p],
            ["weighted_recall",weighted_r],
            ["weighted_f1",weighted_f1],
            ["num_test_sequences",len(y_true)],
        ])

    report=classification_report(
        y_true,y_pred,labels=labels,target_names=CLASSES,
        zero_division=0,digits=4
    )
    (OUT/"classification_report.txt").write_text(report,encoding="utf-8")

    with (OUT/"sequence_predictions.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow([
            "subject","sequence","num_frames","true_label",
            "predicted_label","correct","prediction_confidence",
            "true_label_probability"
        ])
        w.writerows(sequence_rows)

    with (OUT/"temporal_probabilities.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow([
            "subject","sequence","true_label","frame_index",
            "frame_name","normalized_time",*CLASSES,
            "step_prediction","step_prediction_probability"
        ])
        w.writerows(temporal_rows)

    per_class=defaultdict(lambda:[0,0])
    for true_idx,pred_idx in zip(y_true,y_pred):
        name=CLASSES[true_idx]
        per_class[name][1]+=1
        per_class[name][0]+=int(true_idx==pred_idx)

    with (OUT/"per_emotion_accuracy.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["emotion","correct","total","accuracy"])
        for name in CLASSES:
            correct,total=per_class[name]
            w.writerow([name,correct,total,correct/total if total else 0.0])

    cm=confusion_matrix(y_true,y_pred,labels=labels)
    save_cm(cm,OUT/"confusion_matrix.png")

    print("\nHELD-OUT TEST RESULTS")
    print("="*50)
    print(f"Accuracy        : {accuracy*100:.2f}%")
    print(f"Macro Precision : {macro_p*100:.2f}%")
    print(f"Macro Recall    : {macro_r*100:.2f}%")
    print(f"Macro F1        : {macro_f1*100:.2f}%")
    print("\n",report)
    print("Saved to:",OUT)
    print("\ntemporal_probabilities.csv can be used for Trigger Time analysis.")

if __name__=="__main__":
    main()
