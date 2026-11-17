import argparse
from brain_tumor import train

p=argparse.ArgumentParser(description="Train a brain MRI classifier")
p.add_argument('--model',choices=['cnn','mobilenet_v2'],default='cnn')
p.add_argument('--data',default='Tumour')
p.add_argument('--output',default='models')
p.add_argument('--epochs',type=int,default=8)
p.add_argument('--batch-size',type=int,default=32)
p.add_argument('--image-size',type=int,default=160)
p.add_argument('--no-pretrained',action='store_true',help='Do not request ImageNet weights for MobileNetV2')
a=p.parse_args()
train(a.model,a.data,a.output,a.epochs,a.batch_size,pretrained=not a.no_pretrained,image_size=a.image_size)
