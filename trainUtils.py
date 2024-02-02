import torch.nn as nn
import torch
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PDBData(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __getitem__(self, index):
        data = self.data_list[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data_list)

def train(model, trainloader, optimizer):
    model.train()
    running_loss = 0
    for data in trainloader:
         data.to(device)
         out = model(data.x, data.edge_index, data.edge_attr, data.batch)
         loss = F.mse_loss(out.squeeze(), data.y.squeeze(), reduction = 'mean')
         running_loss += loss.item()*len(data.y.squeeze())
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
    print(running_loss/len(trainloader.dataset))

def test(model, testloader):
     model.eval()
     predictions = []
     all_loss = 0
     for data in testloader:
         data.to(device)
         out = model(data.x, data.edge_index, data.edge_attr, data.batch)
         loss = F.mse_loss(out.squeeze(), data.y.squeeze(),reduction = "mean")
         all_loss += loss.item()
         predictions.append(out)
     loss = all_loss / len(testloader.dataset)
     return predictions,loss