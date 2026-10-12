#!/usr/bin/env python3
from pathlib import Path
import csv, random, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

PROJECT = Path.home()/"Desktop"/"FER_project"
DATA = PROJECT/"ckplus_full_pipeline_data"
TRAIN_CSV, VAL_CSV = DATA/"train_sequences.csv", DATA/"val_sequences.csv"
OUT = PROJECT/"pretrained_training"; OUT.mkdir(exist_ok=True)
MODEL = PROJECT/"cnn_lstm_ckplus_pretrained.pth"
CLASSES=["anger","contempt","disgust","fear","happy","sadness","surprise"]
C2I={c:i for i,c in enumerate(CLASSES)}
SEED=42; BATCH_SIZE=8; EPOCHS=25; LR=1e-4; HIDDEN=256
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
TF=transforms.Compose([transforms.Grayscale(3),transforms.Resize((224,224)),transforms.ToTensor(),
 transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def rows(p):
    with open(p,newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def val(r,names):
    d={k.lower():v for k,v in r.items()}
    for n in names:
        if d.get(n.lower()) not in (None,""): return d[n.lower()]
def frame_dir(r):
    p=val(r,["frame_dir","image_dir","sequence_dir","frames_dir","path","sequence_path","image_path"])
    if p and Path(p).expanduser().exists(): return Path(p).expanduser()
    s=val(r,["subject","subject_id"]); q=val(r,["sequence","sequence_id"])
    p=PROJECT/"CK+ dataset (Original Ver.)"/"cohn-kanade-images"/s/q
    if p.exists(): return p
    raise FileNotFoundError(r)
def label(r):
    # First try textual emotion labels.
    for key in ["emotion", "class", "class_name", "label_name", "label"]:
        x = val(r, [key])
        if x is not None:
            x = str(x).strip().lower()
            if x in C2I:
                return C2I[x]

    # Then try numeric labels if present.
    for key in ["emotion_label", "class_idx", "class_id", "label"]:
        x = val(r, [key])
        if x is not None:
            try:
                n = int(float(x))
            except ValueError:
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
    def __init__(self,p):self.r=rows(p)
    def __len__(self):return len(self.r)
    def __getitem__(self,i):
        r=self.r[i]; p=frame_dir(r); fs=sorted(p.glob("*.png")) or sorted(p.glob("*.jpg"))
        return torch.stack([TF(Image.open(f).convert("L")) for f in fs]),label(r)
def collate(b):
    xs,ys=zip(*b); lens=torch.tensor([len(x) for x in xs])
    return pad_sequence(xs,batch_first=True),lens,torch.tensor(ys)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        b=models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for p in b.parameters():p.requires_grad=False
        for p in b.layer4.parameters():p.requires_grad=True
        self.cnn=nn.Sequential(*list(b.children())[:-1])
        self.lstm=nn.LSTM(512,HIDDEN,batch_first=True)
        self.fc=nn.Linear(HIDDEN,7)
    def forward(self,x,lens):
        B,T,C,H,W=x.shape
        f=self.cnn(x.reshape(B*T,C,H,W)).flatten(1).reshape(B,T,512)
        p=pack_padded_sequence(f,lens.cpu(),batch_first=True,enforce_sorted=False)
        _,(h,_)=self.lstm(p)
        return self.fc(h[-1])

def epoch(m,dl,lossfn,opt=None):
    train=opt is not None;m.train(train); loss_sum=ok=n=0
    for x,l,y in dl:
        x,y=x.to(DEVICE),y.to(DEVICE)
        if train:opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            z=m(x,l); loss=lossfn(z,y)
            if train:loss.backward();opt.step()
        loss_sum+=loss.item()*len(y);ok+=(z.argmax(1)==y).sum().item();n+=len(y)
    return loss_sum/n,ok/n

def main():
    print("Using device:",DEVICE)
    print("Experiment B: ImageNet-pretrained ResNet18; frozen conv1-layer3; trainable layer4 + LSTM + classifier")
    tr,va=DS(TRAIN_CSV),DS(VAL_CSV)
    print(f"Train sequences: {len(tr)} | Val sequences: {len(va)}")
    g=torch.Generator().manual_seed(SEED)
    trl=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=collate,generator=g)
    val=DataLoader(va,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,collate_fn=collate)
    m=Net().to(DEVICE); lossfn=nn.CrossEntropyLoss()
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=LR)
    hist=[];best=-1;best_ep=0
    for ep in range(1,EPOCHS+1):
        t=time.time();tl,ta=epoch(m,trl,lossfn,opt);vl,vaa=epoch(m,val,lossfn)
        hist.append(dict(epoch=ep,train_loss=tl,train_acc=ta,val_loss=vl,val_acc=vaa,seconds=time.time()-t))
        print(f"Epoch {ep:02d}/{EPOCHS} | train loss={tl:.4f} acc={ta*100:.2f}% | val loss={vl:.4f} acc={vaa*100:.2f}%")
        if vaa>best:
            best,best_ep=vaa,ep
            torch.save({"model_state_dict":m.state_dict(),"classes":CLASSES,"hidden_size":HIDDEN,
                        "best_epoch":ep,"best_val_acc":vaa,"experiment":"B_ImageNet_pretrained_layer4_finetune"},MODEL)
            print("  -> saved best model")
        with open(OUT/"training_history.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=hist[0].keys());w.writeheader();w.writerows(hist)
    import matplotlib.pyplot as plt
    e=[r["epoch"] for r in hist]
    plt.figure();plt.plot(e,[r["train_loss"] for r in hist],label="Train");plt.plot(e,[r["val_loss"] for r in hist],label="Validation")
    plt.xlabel("Epoch");plt.ylabel("Loss");plt.legend();plt.tight_layout();plt.savefig(OUT/"training_loss.png",dpi=200);plt.close()
    plt.figure();plt.plot(e,[100*r["train_acc"] for r in hist],label="Train");plt.plot(e,[100*r["val_acc"] for r in hist],label="Validation")
    plt.xlabel("Epoch");plt.ylabel("Accuracy (%)");plt.legend();plt.tight_layout();plt.savefig(OUT/"training_accuracy.png",dpi=200);plt.close()
    print(f"\nTraining complete. Best epoch: {best_ep} | Best val accuracy: {best*100:.2f}%")
    print("Best model:",MODEL)
if __name__=="__main__":main()
