#importacao das bibliotecas
import urllib.request
import os
from PIL import Image,ImageStat
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torch import optim 
import sklearn.metrics as metrics
import torchsummary
import json
import torch.nn.functional as F
from google.colab import drive
drive.mount('/content/drive')

#finetunning nos modelos pré-treinados
mobilenet = models.mobilenet_v3_small(pretrained=True)
mobilenet.classifier[3] = nn.Linear(1024, 22)
googlenet =models.googlenet(pretrained=True)
googlenet.fc=nn.Linear(1024,22)
resnet = models.resnet18(pretrained=True)
resnet.fc=nn.Linear(512,22)

#construcao do dataset
main_dir = "/content/drive/MyDrive/Trabalho/Dataset"

os.chdir(main_dir)

transform = transforms.Compose([transforms.Resize((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])

img_tensor = transform(img)
img_tensor.shape
dir = './'
for file_names in os.scandir('./'):
    print(file_names)
classes = [d.name for d in os.scandir(dir) if d.is_dir()]
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
class_to_idx
dataset = torchvision.datasets.DatasetFolder('./',loader = image_loader,extensions='jpg',transform=transform )
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21]),
 array([55, 55, 46, 62, 81, 54, 67, 67, 66, 46, 71, 89, 60, 49, 47, 72, 68,
        63, 55, 74, 59, 55]))

from torch.utils.data import Dataset

from typing import Any,Tuple

class Dataset(Dataset):
    def __init__(self,dir,transform=None,target_transform=None,loader = None):
        self.main_dir = dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.targets = []
        self.instances = self.make_instances()
        self.loader = loader
        
        if loader is None:
            self.loader = lambda x: Image.open(x).convert('RGB')

    def make_instances(self):
        instances = []
        targets = []
        for target_class in sorted(self.class_to_idx.keys()):
                class_index = self.class_to_idx[target_class]
                target_dir = os.path.join(self.main_dir, target_class)
                for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        item = path, class_index
                        targets.append(class_index)
                        instances.append(item)
        self.targets = torch.tensor(targets)
        return instances
    def __getitem__(self,index:int) -> Tuple[Any,Any]:
        path, target = self.instances[index]
        instance = self.loader(path)
        if self.transform is not None:
            instance = self.transform(instance)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return instance,target
    def __len__(self) -> int:
        return len(self.instances)

dataset = Dataset('./',transform=transform)
#Salvando as classes
with open('/content/drive/MyDrive/Trabalho/classes.json', 'w') as fp:
    json.dump(dataset.class_to_idx, fp)
#Treinamento das redes

ds = dataset

from sklearn.model_selection import train_test_split

bs = 64
train_idx, temp_idx = train_test_split(np.arange(len(ds)),test_size=0.3,shuffle=True,stratify=ds.targets)
valid_idx, test_idx = train_test_split(temp_idx,test_size=0.5,shuffle=True,stratify=ds.targets[temp_idx])
 
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
test_sampler  = torch.utils.data.SubsetRandomSampler(test_idx)
 
dl_train = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=train_sampler)
dl_valid = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=valid_sampler)
dl_test  = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=test_sampler)

x,y = next(iter(dl_train))

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#avaliacao do modelo
def avaliamodelo(model, device, loss_eval, loss_train):
  model.eval()
  lres = []
  ytrue = []
  with torch.no_grad():
      for data,target in dl_test:
          data = data.to(device)
          pred = model(data)
          res  = pred.argmax(dim=1).cpu().tolist()
          lres += res
          ytrue += target
  plt.ion()

  fig = plt.figure()
  plt.plot(loss_train[1:])
  plt.plot(loss_eval[1:])
  plt.xlabel('Epochs')
  plt.show()
  metrics.ConfusionMatrixDisplay.from_predictions(ytrue, lres)
  plt.title(label = f"Matriz de Confusão")
  plt.show()
  f1_macro = metrics.f1_score(y_true= ytrue,y_pred=lres,average='macro')
  f1_micro = metrics.f1_score(y_true= ytrue,y_pred=lres,average='micro')
  precision = metrics.precision_score(y_true= ytrue,y_pred=lres,average='weighted')
  recall = metrics.recall_score(y_true= ytrue,y_pred=lres,average='weighted')
  print("precision:%f" %(precision))
  print("\nrecall:%f" %(recall))
  print("\nf1_macro:%f"%(f1_macro))
  print("\nf1_micro:%f\n"%(f1_micro))
  print(metrics.classification_report(ytrue,lres))

  #Googlenet
googlenet.to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(googlenet.parameters(),lr=0.1)

epochs = 100
loss_train = []
loss_eval  = []
stop = False
epoch = 0
lowest_loss_eval = 10000
last_best_result = 0
while (not stop):
    googlenet.train()
    lloss = []
    for x,y in dl_train:
        x = x.to(device)
        y = y.to(device)
        pred = googlenet(x)
        closs = criterion(pred,y)
        closs.backward()
        opt.step()
        opt.zero_grad()
        lloss.append(closs.item())
    loss_train.append(np.mean(lloss))
    llos = []
    googlenet.eval()
    lres = []
    ytrue = []
    with torch.no_grad():
        for data,y in dl_valid:
            data = data.to(device)
            pred = googlenet(data)
            closs = criterion(pred.cpu(),y)
            lloss.append(closs.item())
            res  = pred.argmax(dim=1).cpu().tolist()
            lres += res
            ytrue += y
    avg_loss_eval = np.mean(lloss)
    loss_eval.append(avg_loss_eval)
    if avg_loss_eval < lowest_loss_eval:
        lowest_loss_eval = avg_loss_eval 
        last_best_result = 0
        print("Best model found! saving...")
        actual_state = {'model':googlenet.state_dict(),'opt':opt.state_dict(),'epoch':epoch,'loss':lowest_loss_eval,'loss_train':loss_train,'loss_eval':loss_eval}
        torch.save(actual_state,'/content/drive/MyDrive/Trabalho/best_model_google.pth')
    last_best_result += 1
    if last_best_result > 10:
        stop = True
    print("epoch %d loss_train %4.3f loss_eval %4.3f last_best %d"%(epoch,loss_train[-1],loss_eval[-1],last_best_result))
    epoch += 1

recover_googlenet = torch.load('/content/drive/MyDrive/Trabalho/best_model_google.pth')

googlenet.load_state_dict(recover_googlenet['model'],strict=False)
opt.load_state_dict(recover_googlenet['opt'])
loss_train = recover_googlenet['loss_train']
loss_eval  = recover_googlenet['loss_eval']
stop = False
epoch = recover_googlenet['epoch']
lowest_loss_eval = recover_googlenet['loss']
last_best_result = 0

avaliamodelo(googlenet, device, loss_eval, loss_train)

#MobileNet

mobilenet.to(device)

criterion = nn.CrossEntropyLoss()
opt = optim.SGD(mobilenet.parameters(),lr=0.1)

epochs = 100
loss_train = []
loss_eval  = []
stop = False
epoch = 0
lowest_loss_eval = 10000
last_best_result = 0
while (not stop):
    mobilenet.train()
    lloss = []
    for x,y in dl_train:
        x = x.to(device)
        y = y.to(device)
        pred = mobilenet(x)
        closs = criterion(pred,y)
        closs.backward()
        opt.step()
        opt.zero_grad()
        lloss.append(closs.item())
    loss_train.append(np.mean(lloss))
    llos = []
    mobilenet.eval()
    lres = []
    ytrue = []
    with torch.no_grad():
        for data,y in dl_valid:
            data = data.to(device)
            pred = mobilenet(data)
            closs = criterion(pred.cpu(),y)
            lloss.append(closs.item())
            res  = pred.argmax(dim=1).cpu().tolist()
            lres += res
            ytrue += y
    avg_loss_eval = np.mean(lloss)
    loss_eval.append(avg_loss_eval)
    if avg_loss_eval < lowest_loss_eval:
        lowest_loss_eval = avg_loss_eval 
        last_best_result = 0
        print("Best model found! saving...")
        actual_state = {'model':mobilenet.state_dict(),'opt':opt.state_dict(),'epoch':epoch,'loss':lowest_loss_eval,'loss_train':loss_train,'loss_eval':loss_eval}
        torch.save(actual_state,'/content/drive/MyDrive/Trabalho/best_model_mobile.pth')
    last_best_result += 1
    if last_best_result > 10:
        stop = True
    print("epoch %d loss_train %4.3f loss_eval %4.3f last_best %d"%(epoch,loss_train[-1],loss_eval[-1],last_best_result))
    epoch += 1

recover_mobile = torch.load('/content/drive/MyDrive/Trabalho/best_model_mobile.pth')

mobilenet.load_state_dict(recover_mobile['model'],strict=False)
opt.load_state_dict(recover_mobile['opt'])
loss_train = recover_mobile['loss_train']
loss_eval  = recover_mobile['loss_eval']
stop = False
epoch = recover_mobile['epoch']
lowest_loss_eval = recover_mobile['loss']
last_best_result = 0

avaliamodelo(mobilenet, device, loss_eval, loss_train)

#Resnet
resnet.to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(resnet.parameters(),lr=0.1)

epochs = 100
loss_train = []
loss_eval  = []
stop = False
epoch = 0
lowest_loss_eval = 10000
last_best_result = 0
while (not stop):
    resnet.train()
    lloss = []
    for x,y in dl_train:
        x = x.to(device)
        y = y.to(device)
        pred = resnet(x)
        closs = criterion(pred,y)
        closs.backward()
        opt.step()
        opt.zero_grad()
        lloss.append(closs.item())
    loss_train.append(np.mean(lloss))
    llos = []
    resnet.eval()
    lres = []
    ytrue = []
    with torch.no_grad():
        for data,y in dl_valid:
            data = data.to(device)
            pred = resnet(data)
            closs = criterion(pred.cpu(),y)
            lloss.append(closs.item())
            res  = pred.argmax(dim=1).cpu().tolist()
            lres += res
            ytrue += y
    avg_loss_eval = np.mean(lloss)
    loss_eval.append(avg_loss_eval)
    if avg_loss_eval < lowest_loss_eval:
        lowest_loss_eval = avg_loss_eval 
        last_best_result = 0
        print("Best model found! saving...")
        actual_state = {'model':resnet.state_dict(),'opt':opt.state_dict(),'epoch':epoch,'loss':lowest_loss_eval,'loss_train':loss_train,'loss_eval':loss_eval}
        torch.save(actual_state,'/content/drive/MyDrive/Trabalho/best_model_resnet.pth')
    last_best_result += 1
    if last_best_result > 10:
        stop = True
    print("epoch %d loss_train %4.3f loss_eval %4.3f last_best %d"%(epoch,loss_train[-1],loss_eval[-1],last_best_result))
    epoch += 1

recover_resnet = torch.load('/content/drive/MyDrive/Trabalho/best_model_resnet.pth')

resnet.load_state_dict(recover_resnet['model'],strict=False)
opt.load_state_dict(recover_resnet['opt'])
loss_train = recover_resnet['loss_train']
loss_eval  = recover_resnet['loss_eval']
stop = False
epoch = recover_resnet['epoch']
lowest_loss_eval = recover_resnet['loss']
last_best_result = 0

avaliamodelo(resnet, device, loss_eval, loss_train)

!pip install python-telegram-bot --upgrade

#escolhe o melhor modelo
recover = torch.load('/content/drive/MyDrive/Trabalho/best_model_mobile.pth')
model= models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(1024, 22)
model.to(device)
model.load_state_dict(recover['model'],strict=False)

#integração  com telegram
with open("/content/drive/MyDrive/Trabalho/classes.json", "r") as read_file:
  classes = json.load(read_file)

  from telegram.ext import Updater, Filters, MessageHandler, CommandHandler
import requests
import re
import torchvision.transforms as transforms
from PIL import Image,ImageStat
def classificador (path):
  model.eval()
  transform = transforms.Compose([transforms.Resize((80,80)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
  loader = lambda x: Image.open(x).convert('RGB')
  imagem_recebida = loader(f'/content/drive/MyDrive/Trabalho/imagens_recebidas/{path}')
  amostra = transform(imagem_recebida)
  amostra = amostra.to(device)
  amostra = amostra.unsqueeze(0)
  predito = model(amostra)
  predict  = predito.argmax(dim=1).cpu().tolist()
  name_predict = (list(classes.keys())[list(classes.values()).index(predict[0])])
  return name_predict.title()

def image_handler(update, context):
    file = update.message.photo[-1].file_id
    obj = context.bot.get_file(file)
    os.chdir('/content/drive/MyDrive/Trabalho/imagens_recebidas/')
    path = obj.download()
    resultado = classificador(path)
    update.message.reply_text(f'Esse é o tanque {resultado}')

def start(update, context):
  return update.message.reply_text('Classificador de tanques, envie uma foto do tanque a ser classificado')

def main():
    updater = Updater('5515332922:AAGMGKwC6_gP8CFz7HBhRRd4fOrO4RsZVkI')
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.photo, image_handler))
    dp.add_handler(CommandHandler('start', start))
    updater.start_polling()
    updater.idle()