#!/usr/bin/env python3
from pathlib import Path
import csv, numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix

P=Path.home()/"Desktop"/"FER_project"; DATA=P/"ckplus_full_pipeline_data"; TEST=DATA/"test_sequences.csv"
MODEL=P/"cnn_lstm_ckplus_pretrained.pth"; OUT=P/"pretrained_evaluation";OUT.mkdir(exist_ok=True)
C=["anger","contempt","disgust","fear","happy","sadness","surprise"]; C2I={c:i for i,c in enumerate(C)}
DEV=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
TF=transforms.Compose([transforms.Grayscale(3),transforms.Resize((224,224)),transforms.ToTensor(),
transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
def rows(p):
    with open(p,newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def v(r,ns):
    d={k.lower():x for k,x in r.items()}
    for n in ns:
        if d.get(n.lower()) not in (None,""):return d[n.lower()]
def fd(r):
    x=v(r,["frame_dir","image_dir","sequence_dir","frames_dir","path","sequence_path","image_path"])
    if x and Path(x).expanduser().exists():return Path(x).expanduser()
    s=v(r,["subject","subject_id"]);q=v(r,["sequence","sequence_id"])
    return P/"CK+ dataset (Original Ver.)"/"cohn-kanade-images"/s/q
def lab(r):
    # First try textual emotion labels
    for key in ["emotion", "class", "class_name", "label_name", "label"]:
        x = v(r, [key])

        if x is not None:
            x = str(x).strip().lower()

            if x in C2I:
                return C2I[x]

    # Then try numeric labels
    for key in ["emotion_label", "class_idx", "class_id", "label"]:
        x = v(r, [key])

        if x is not None:
            try:
                n = int(float(x))
            except (ValueError, TypeError):
                continue

            if 0 <= n <= 6:
                return n

            ck_map = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6
            }

            if n in ck_map:
                return ck_map[n]

    raise ValueError(f"Could not resolve label from CSV row: {r}")
class DS(Dataset):
    def __init__(self):self.r=rows(TEST)
    def __len__(self):return len(self.r)
    def __getitem__(self,i):
        r=self.r[i];p=fd(r);fs=sorted(p.glob("*.png")) or sorted(p.glob("*.jpg"))
        s=v(r,["subject","subject_id"]) or p.parent.name;q=v(r,["sequence","sequence_id"]) or p.name
        return torch.stack([TF(Image.open(f).convert("L")) for f in fs]),lab(r),s,q
def col(b):
    x,y,s,q=zip(*b);l=torch.tensor([len(a) for a in x])
    return pad_sequence(x,batch_first=True),l,torch.tensor(y),s,q
class Net(nn.Module):
    def __init__(self,h=256):
        super().__init__();b=models.resnet18(weights=None);self.cnn=nn.Sequential(*list(b.children())[:-1])
        self.lstm=nn.LSTM(512,h,batch_first=True);self.fc=nn.Linear(h,7)
    def forward(self,x,l):
        B,T,Cc,H,W=x.shape;f=self.cnn(x.reshape(B*T,Cc,H,W)).flatten(1).reshape(B,T,512)
        po=pack_padded_sequence(f,l.cpu(),batch_first=True,enforce_sorted=False);po,_=self.lstm(po)
        o,_=pad_packed_sequence(po,batch_first=True,total_length=T);z=self.fc(o)
        idx=(l.to(z.device)-1).view(B,1,1).expand(-1,1,7)
        return z.gather(1,idx).squeeze(1),z
def main():
    ck=torch.load(MODEL,map_location="cpu");m=Net(ck.get("hidden_size",256));m.load_state_dict(ck["model_state_dict"]);m.to(DEV).eval()
    print("Using device:",DEV);print("Best epoch:",ck.get("best_epoch"),"| Best val accuracy:",f'{100*ck.get("best_val_acc",0):.2f}%')
    ds=DS();dl=DataLoader(ds,batch_size=4,shuffle=False,num_workers=0,collate_fn=col)
    yt=[];yp=[];pr=[];tm=[]
    with torch.no_grad():
        for x,l,y,ss,qs in dl:
            fin,z=m(x.to(DEV),l);pf=torch.softmax(fin,1).cpu().numpy();pa=torch.softmax(z,2).cpu().numpy();pred=pf.argmax(1);yn=y.numpy()
            yt+=yn.tolist();yp+=pred.tolist()
            for i in range(len(yn)):
                T=int(l[i]);r={"subject":ss[i],"sequence":qs[i],"true_label":C[yn[i]],"predicted_label":C[pred[i]],"correct":int(yn[i]==pred[i]),"sequence_length":T}
                r.update({f"final_p_{c}":float(pf[i,j]) for j,c in enumerate(C)});pr.append(r)
                for t in range(T):
                    rr={"subject":ss[i],"sequence":qs[i],"true_label":C[yn[i]],"predicted_label_final":C[pred[i]],"frame_index":t+1,
                        "sequence_length":T,"normalized_time":0 if T==1 else t/(T-1)}
                    rr.update({f"p_{c}":float(pa[i,t,j]) for j,c in enumerate(C)});tm.append(rr)
    acc=accuracy_score(yt,yp);mp,mr,mf,_=precision_recall_fscore_support(yt,yp,average="macro",zero_division=0)
    wp,wr,wf,_=precision_recall_fscore_support(yt,yp,average="weighted",zero_division=0)
    metrics={"accuracy":acc,"macro_precision":mp,"macro_recall":mr,"macro_f1":mf,"weighted_precision":wp,"weighted_recall":wr,"weighted_f1":wf}
    with open(OUT/"overall_metrics.csv","w",newline="") as f:
        w=csv.writer(f);w.writerow(["metric","value"]);w.writerows(metrics.items())
    for fn,data in [("sequence_predictions.csv",pr),("temporal_probabilities.csv",tm)]:
        with open(OUT/fn,"w",newline="") as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
    rep=classification_report(yt,yp,labels=range(7),target_names=C,zero_division=0)
    (OUT/"classification_report.txt").write_text(rep)
    cm=confusion_matrix(yt,yp,labels=range(7));per=[]
    a,b=np.array(yt),np.array(yp)
    for i,c in enumerate(C):
        mask=a==i;n=int(mask.sum());cor=int(((b==i)&mask).sum());per.append({"emotion":c,"n_test":n,"correct":cor,"accuracy_recall":cor/n if n else np.nan})
    with open(OUT/"per_emotion_accuracy.csv","w",newline="") as f:w=csv.DictWriter(f,fieldnames=per[0].keys());w.writeheader();w.writerows(per)
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))

    # White background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Confusion matrix
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    # Axis labels
    ax.set_xticks(range(7), C, rotation=45, ha="right")
    ax.set_yticks(range(7), C)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=16)

    # Numbers inside each cell
    for i in range(7):
        for j in range(7):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
        )

    fig.tight_layout()

    fig.savefig(
        OUT / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print("\nEXPERIMENT B — HELD-OUT TEST")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Macro F1: {mf*100:.2f}%")
    print(f"Weighted F1: {wf*100:.2f}%")
    print(rep)
    print("Saved:", OUT)
if __name__=="__main__":main()