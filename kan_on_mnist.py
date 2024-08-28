import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from KanModule import *
import pandas as pd

trial = 1
epoch = 50
norm_type = 'bn'
weight_decay = 0
dropout = 0
learning_rate = 0.005
save_path = f"record\KAN_regulation{weight_decay}_trial{trial}.csv"

transform = transforms.ToTensor()
minist = datasets.MNIST('./data', download=True, transform=transform)
train, val = torch.utils.data.random_split(minist, [50000, 10000])
train_loader = DataLoader(train, batch_size=1024, shuffle=True)
val_loader = DataLoader(val, batch_size=64, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()

model_type = 'K' # K for Kan, M for MLP

def train_epoch(model, train_loader, criterion, optimizer,scheduler=None, weight_decay=weight_decay):
    model.train()
    model.to('cuda')
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        if weight_decay > 0:
            loss += weight_decay * model.get_norm()
        loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return running_loss / len(train_loader)

def val_loss(model, val_loader, criterion):
    model.eval()
    model.to('cuda')
    running_loss = 0.0
    for data in val_loader:
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
    return running_loss / len(val_loader)

class fcKan(torch.nn.Module):
    def __init__(self):
        super(fcKan, self).__init__()
        self.kan1 = KanLayer(3,28*28,128, norm = norm_type)
        self.dropout1 = nn.Dropout(dropout)
        self.kan2 = KanLayer(3,128,32, norm = norm_type)
        self.dropout2 = nn.Dropout(dropout)
        self.kan3 = KanLayer(3,32,10, norm = norm_type)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.kan1(x)
        x = self.dropout1(x)
        x = self.kan2(x)
        x = self.dropout2(x)
        x = self.kan3(x)
        return x
    
    def get_norm(self):
        return self.kan1.get_norm() + self.kan2.get_norm() + self.kan3.get_norm()
    
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.bn_start = nn.BatchNorm1d(28*28)
        self.fc1 = torch.nn.Linear(28*28,768)
        self.bn1 = nn.BatchNorm1d(768)
        self.dropout1 = nn.Dropout(0)
        self.fc2 = torch.nn.Linear(768, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0)
        self.fc3 = torch.nn.Linear(128, 10)
        
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.bn_start(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

model = fcKan() if model_type == 'K' else MLP() 
model.to('cuda')
#count the number of parameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 

recorder = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "time"])

start = time.time()
for i in range(1, epoch+1):
    loss = train_epoch(model, train_loader, criterion, optimizer, scheduler)
    val = val_loss(model, val_loader, criterion)
    end = time.time()
    duration = (end - start) / 60  # convert to minutes
    print(f"Epoch {i} train loss {loss} val loss {val} time {duration} minutes")
    new_row = pd.DataFrame({"epoch": [i], "train_loss": [loss], "val_loss": [val], "time": [duration]})
    recorder = pd.concat([recorder, new_row], ignore_index=True)

input(f"Press Enter to continue..., recording to {save_path}")
recorder.to_csv(save_path, index=False)