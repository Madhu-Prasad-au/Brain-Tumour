"""PyTorch training and inference utilities for four-class brain MRI classification."""
from pathlib import Path
import json, random, time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMAGE_SIZE = 160

class MRIImageDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root, self.transform = Path(root) / split, transform
        self.samples = [(p, CLASSES.index(folder.name)) for folder in self.root.iterdir() if folder.is_dir() for p in folder.glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}]
        self.samples.sort(key=lambda x: str(x[0]).lower())
        if not self.samples: raise FileNotFoundError(f'No labeled images found under {self.root}')
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im: image = im.convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

def make_transforms(size=IMAGE_SIZE):
    train = transforms.Compose([transforms.Resize((size,size)), transforms.RandomRotation(12), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(p=.15), transforms.RandomAffine(degrees=0, translate=(.05,.05), scale=(.9,1.1)), transforms.ColorJitter(brightness=.12, contrast=.12), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    evaluate = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return train, evaluate

class BrainCNN(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        layers=[]; c=3
        for out in (32,64,128,192):
            layers += [nn.Conv2d(c,out,3,padding=1,bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(.12)]
            c=out
        self.features=nn.Sequential(*layers)
        self.classifier=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(c,128),nn.ReLU(inplace=True),nn.Dropout(.4),nn.Linear(128,n))
    def forward(self,x): return self.classifier(self.features(x))

def build_model(kind='cnn', pretrained=True):
    if kind == 'cnn': return BrainCNN(len(CLASSES)), False
    if kind != 'mobilenet_v2': raise ValueError('model must be cnn or mobilenet_v2')
    try:
        weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model=models.mobilenet_v2(weights=weights)
        used=weights is not None
    except Exception as e:
        print(f'Pretrained weights unavailable ({e}); initializing MobileNetV2 without ImageNet weights.')
        model=models.mobilenet_v2(weights=None); used=False
    model.classifier[1]=nn.Linear(model.last_channel,len(CLASSES))
    return model, used

def train(model_kind='cnn', data_dir='Tumour', output_dir='models', epochs=8, batch_size=32, lr=1e-3, pretrained=True, image_size=IMAGE_SIZE, workers=0):
    seed=42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_tf, eval_tf=make_transforms(image_size)
    ds={s:MRIImageDataset(data_dir,s,train_tf if s=='train' else eval_tf) for s in ('train','valid','test')}
    loaders={s:DataLoader(ds[s],batch_size=batch_size,shuffle=s=='train',num_workers=workers,pin_memory=device.type=='cuda') for s in ds}
    model, used_pretrained=build_model(model_kind,pretrained); model.to(device)
    criterion=nn.CrossEntropyLoss(); optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    best=float('inf'); history={'train_loss':[],'train_accuracy':[],'valid_loss':[],'valid_accuracy':[]}
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); best_path=out/f'{model_kind}.pt'
    for epoch in range(epochs):
        row={}
        for phase in ('train','valid'):
            model.train(phase=='train'); loss_sum=correct=total=0
            for x,y in loaders[phase]:
                x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase=='train'):
                    logits=model(x); loss=criterion(logits,y)
                    if phase=='train': loss.backward(); optimizer.step()
                loss_sum+=loss.item()*y.size(0); correct+=(logits.argmax(1)==y).sum().item(); total+=y.size(0)
            row[f'{phase}_loss']=loss_sum/total; row[f'{phase}_accuracy']=correct/total
        for k,v in row.items(): history[k].append(v)
        print(f"Epoch {epoch+1}/{epochs} - train loss {row['train_loss']:.4f}, acc {row['train_accuracy']:.3f}; valid loss {row['valid_loss']:.4f}, acc {row['valid_accuracy']:.3f}")
        if row['valid_loss']<best:
            best=row['valid_loss']; torch.save({'state_dict':model.state_dict(),'model':model_kind,'classes':CLASSES,'image_size':image_size,'pretrained':used_pretrained},best_path)
    # Evaluation on held-out test split
    ckpt=torch.load(best_path,map_location=device,weights_only=False); model.load_state_dict(ckpt['state_dict']); model.eval()
    cm=np.zeros((len(CLASSES),len(CLASSES)),dtype=int)
    with torch.no_grad():
        for x,y in loaders['test']:
            pred=model(x.to(device)).argmax(1).cpu().numpy()
            for actual,guess in zip(y.numpy(),pred): cm[actual,guess]+=1
    metrics={}
    for i,name in enumerate(CLASSES):
        tp=cm[i,i]; precision=tp/max(1,cm[:,i].sum()); recall=tp/max(1,cm[i,:].sum()); f1=2*precision*recall/max(1e-12,precision+recall)
        metrics[name]={'precision':precision,'recall':recall,'f1':f1,'support':int(cm[i,:].sum())}
    accuracy=float(cm.trace()/max(1,cm.sum()))
    results={'model':model_kind,'pretrained_imagenet':used_pretrained,'device':str(device),'test_accuracy':accuracy,'per_class':metrics,'confusion_matrix':cm.tolist(),'classes':CLASSES,'history':history,'epochs':epochs,'image_size':image_size}
    (out/f'{model_kind}_results.json').write_text(json.dumps(results,indent=2))
    print(f'Test accuracy: {accuracy:.4f}\nConfusion matrix (rows=true, columns=predicted):\n{cm}')
    return results

def load_for_inference(checkpoint, device='cpu'):
    ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
    model,_=build_model(ckpt['model'],pretrained=False); model.load_state_dict(ckpt['state_dict']); model.to(device).eval()
    return model,ckpt

def predict_image(model, image, image_size=IMAGE_SIZE, device='cpu'):
    _,tf=make_transforms(image_size)
    if isinstance(image, Image.Image):
        source = image.convert('RGB')
    else:
        with Image.open(image) as im: source = im.convert('RGB')
    x=tf(source).unsqueeze(0).to(device)
    with torch.no_grad(): probs=torch.softmax(model(x),dim=1)[0].cpu().numpy()
    order=np.argsort(probs)[::-1]
    return [(CLASSES[i],float(probs[i])) for i in order]
