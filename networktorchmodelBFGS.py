import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import os
from pdb import set_trace
import time
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from ARCLSR1 import ARCLSR1
# from quasiadam import quasiAdam
# import tqdm
import numpy as np



#Define the class models

class Trainer(object):
  """This class is for training and testing the model"""
  def __init__(self, dataset, optimizer, gammas):
    super(Trainer, self).__init__()   
    self.batch_size_train = batch_size_train
    self.batch_size_test = batch_size_test
    self.learning_rate = learning_rate
    self.momentum = 0.5
    self.log_interval = log_interval
    self.random_seed = random_seed
    self.dataset=dataset
    self.optim = optimizer
    self.optim_name = optimizer
    torch.backends.cudnn_enabled = False
    torch.manual_seed(self.random_seed)
    if dataset == 'fashion':
      self.train_loader, self.test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('../data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                                   ])),batch_size=batch_size_train, shuffle=True) , torch.utils.data.DataLoader(
                                  torchvision.datasets.FashionMNIST('../data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

      #Define the corresponding network for this dataset
      class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

      self.Network = Net().to(device=device)
      if self.optim_name == 'svrg':
        self.model_snapshot = Net().to(device=device)


    elif dataset == 'mnist':
      self.train_loader, self.test_loader = torch.utils.data.DataLoader(
                                  torchvision.datasets.MNIST('../data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                                   ])),batch_size=batch_size_train, shuffle=True),torch.utils.data.DataLoader(
                                    torchvision.datasets.MNIST('../data/', train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))])),
                                      batch_size=batch_size_test, shuffle=True)
      
      #Define the corresponding network for this dataset

      class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
      self.Network = Net().to(device=device)
      if self.optim_name == 'svrg':
        self.model_snapshot = Net().to(device=device)

    elif dataset == 'iris':
      iris = load_iris()
      X = iris['data']
      y = iris['target']
      names = iris['target_names']
      feature_names = iris['feature_names']
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X)
      X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=2)
      self.X_train, self.y_train = Variable(torch.from_numpy(X_train)).float(), Variable(torch.from_numpy(y_train)).long().to(device=device)
      self.X_test, self.y_test = Variable(torch.from_numpy(X_test)).float(), Variable(torch.from_numpy(y_test)).long().to(device=device)

      class Model(nn.Module):
        def __init__(self, input_dim):
          super(Model, self).__init__()
          self.layer1 = nn.Linear(input_dim, 50)
          self.layer2 = nn.Linear(50, 50)
          self.layer3 = nn.Linear(50, 3)
        
        def forward(self, x):
          x = F.relu(self.layer1(x))
          x = F.relu(self.layer2(x))
          x = F.softmax(self.layer3(x), dim=1)
          return x

      self.Network = Model(X_train.shape[1]).to(device=device)


    else:
      raise ValueError("Only iris, MNIST and FashionMNIST supported for now")

    # Do the same thing for the optimizer
    if optimizer == 'sgd':
      self.optimizer = torch.optim.SGD(params=self.Network.parameters(), lr=1e-3)

    elif optimizer == 'sdlbfgs':
      self.optimizer = torch.optim.SdLBFGS(params=self.Network.parameters(), max_iter=max_iter, history_size=history)

    elif optimizer == 'lbfgs':
      self.optimizer = torch.optim.LBFGS(params=self.Network.parameters(), max_iter=max_iter, history_size=history)

    elif optimizer == 'adam':
      self.optimizer = torch.optim.Adam(params=self.Network.parameters(), lr=1e-3)

    elif optimizer == 'RMSProp':
      self.optimizer = torch.optim.RMSprop(params=self.Network.parameters())

    elif optimizer == 'Adagrad':
      self.optimizer = torch.optim.Adagrad(params=self.Network.parameters())

    elif optimizer == 'ARCLSR1':
      # set_trace()
      self.optimizer = ARCLSR1(params=self.Network.parameters(),
               gamma1=gammas[0], 
               gamma2=gammas[1],
               eta1=0.7,
               eta2=0.9, 
               history_size=history,
               mu=1e3,
               max_iters=max_iter)
    elif  optimizer == 'svrg':
      self.optimizer_snapshot = optim.SVRG_Snapshot(self.model_snapshot.parameters())
      self.optimizer = torch.optim.SVRG_k(self.Network.parameters(), lr = 1e-3, weight_decay=0.0001)
    # elif optimizer == 'quasiAdam':
    #   self.optimizer = quasiAdam(self.Network.parameters(), lr=1e-2, q_lr=1e-2)
    
  

    else:
      raise ValueError("We do not support any other optimzers for now !")

    # Decide which optimizer you need here.
    self.train_losses = []
    self.train_counter = []
    self.test_losses = []
    self.test_counter = []


  def train(self, epoch):
    self.test()
    if self.optim_name == 'svrg':
      
      # Calculating the mean gradient

      self.optimizer_snapshot.zero_grad()
      for batch_idx, (data, target) in enumerate(self.train_loader):
        data = data.to(device=device)
        target = target.to(device=device)
        output = self.model_snapshot(data)
        snaploss = F.nll_loss(output, target)
        snaploss.backward()
      
      # Pass the current parameters of snapshot optimizer to the svrg_k optimizer
      u = self.optimizer_snapshot.get_param_groups()
      self.optimizer.set_u(u)
      for batch_idx, (data, target) in enumerate(self.train_loader):
        data = data.to(device=device)
        target = target.to(device=device)
        output = self.Network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        output2 = self.model_snapshot(data).to(device)
        loss2 = F.nll_loss(output2, target)
        self.optimizer_snapshot.zero_grad()
        loss2.backward()
        starttime = time.time() 
        self.optimizer.step(self.optimizer_snapshot.get_param_groups())
        endtime = time.time()
        durations.append(starttime-endtime)
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(self.train_loader.dataset),
            100. * batch_idx / len(self.train_loader), loss.item()))

          if loss.item() < 10:
            train_losses.append(loss.item())

          self.train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(self.train_loader.dataset)))
          if not os.path.isdir('../results/'):
            os.mkdir('../results/')
          
          self.test()

    else:
      self.Network.train()
      if self.dataset == 'iris':

        def closure1():
          if torch.is_grad_enabled():
            self.optimizer.zero_grad()
          outputs=self.Network(self.X_test.to(device=device))
          loss=nn.CrossEntropyLoss()(outputs.to(device=device),self.y_test.to(device=device))
          if loss.requires_grad:
            loss.backward()
          return loss

        EPOCHS  = 30
        for epoch in range(EPOCHS):
          y_pred = self.Network(self.X_train)
          loss = nn.CrossEntropyLoss()(y_pred, self.y_train)
          train_losses.append(loss.item())          
          # Zero gradients
          
          self.optimizer.zero_grad()
          self.optimizer_snapshot.zero_grad()
          loss.backward()
          self.optimizer.step()
          self.test()
        
      else:
        for batch_idx, (data, target) in enumerate(self.train_loader):

          data = data.to(device=device)
          target = target.to(device=device)
          def closure1():
            if torch.is_grad_enabled():
              self.optimizer.zero_grad()
            outputs=self.Network(data.to(device=device))
            loss=F.nll_loss(outputs.to(device=device),target.to(device=device))
            if loss.requires_grad:
              loss.backward()
            return loss

          self.optimizer.zero_grad()
          output = self.Network(data)
          loss = F.nll_loss(output, target)
          loss.backward()
          starttime = time.time()
          self.optimizer.step(closure1)
          endtime = time.time()
          durations.append(starttime-endtime)
          if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(self.train_loader.dataset),
              100. * batch_idx / len(self.train_loader), loss.item()))

            if loss.item() < 10:
              train_losses.append(loss.item())

            self.train_counter.append(
              (batch_idx*64) + ((epoch-1)*len(self.train_loader.dataset)))
            if not os.path.isdir('../results/'):
              os.mkdir('../results/')

            # torch.save(self.Network.state_dict(), '../results/'+self.dataset+'_'+self.optim+'_model.pth')

            #Test accuracy of prediction here
            self.test()

    


  def test(self):
    self.Network.eval()
    if self.dataset == 'iris':
      with torch.no_grad():
        y_pred = self.Network(self.X_test)
        correct = (torch.argmax(y_pred, dim=1) == self.y_test).type(torch.FloatTensor)
        test_losses.append(correct.mean())
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
          correct.mean(), len(correct[correct==1]), len(self.X_test),
          100. * correct.mean()))


    else:      
      self.Network.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
        for data, target in self.test_loader:
          data = data.to(device=device)
          target = target.to(device=device)
          output = self.Network(data.to(device=device))
          test_loss += F.nll_loss(output, target, reduction='mean').item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
        # set_trace()
      test_loss /= len(self.test_loader.dataset)
      test_losses.append(100. * correct / len(self.test_loader.dataset))
      print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(self.test_loader.dataset),
        100. * correct / len(self.test_loader.dataset)))
      

def main():
  global n_epochs, batch_size_train, batch_size_test, learning_rate, momentum, log_interval, history, max_iter, random_seed, device, durations, train_losses, test_losses
  n_epochs = 30
  batch_size_sizes = [128, 256, 512, 1024]
  batch_size_test = 128
  learning_rate = 0.01
  momentum = 0.5
  log_interval = 10

  random_seed = 1
  torch.backends.cudnn.enabled = True
  torch.manual_seed(random_seed)


  train_losses = []

  test_losses = []
  durations = []


  device = 'cuda'
  optimizers = ['ARCLSR1']
  histories =[5, 10, 15, 20, 50, 100]
  max_iters = [1, 5, 10, 15, 20, 50, 100]
  batch_size_sizes = [256, 512, 1024, 2048]
  gammas = [(1,10), (1.2,100), (10,100), (10,1000), (100, 1000), (1e2, 1e4)]


  dataset = 'mnist'
  quasiNewton = set(['lbfgs', 'sdlbfgs', 'ARCLSR1'])
  # Perform exhaustive search
  for optimizer in optimizers:

    # Check if optimizer is first order or second order method
    if optimizer in quasiNewton:

      # Try for different iterations per batch
      for max_iter in max_iters:

        # Try for different memory values for the quasi-Newton methods
        for history in histories:

          # Try for different batch sizes
          for batch_size_train in batch_size_sizes:
          
            print('performing optimization for '+optimizer)
            print('Experiment for batch-size', batch_size_train)
            print('Experiment for history-size', history)
            print('Experiment for maxiters', max_iter)
            trainer = Trainer(dataset=dataset, optimizer=optimizer, gammas = (1,10))
            for epoch in range(1, n_epochs + 1):
              trainer.train(epoch)
              trainer.test()

            with open(dataset+'_'+optimizer+'_train', 'wb') as handle:
              pkl.dump(train_losses, handle, protocol=pkl.HIGHEST_PROTOCOL)

            with open(dataset+'_'+optimizer+'_durations', 'wb') as handle:
              pkl.dump(durations, handle, protocol=pkl.HIGHEST_PROTOCOL)
            
            if not os.path.isdir('../results/'+optimizer+'/'+'history_size'+ str(history)+'/'+'max_iters'+ str(max_iter)+'/'+ str(batch_size_train)+'/Testing_losses/'):
              os.makedirs('../results/'+optimizer+'/'+'history_size'+ str(history)+'/'+'max_iters'+ str(max_iter)+'/'+ str(batch_size_train)+'/Testing_losses/')

            with open('../results/'+optimizer+'/'+'history_size'+ str(history)+'/'+'max_iters'+ str(max_iter)+'/'+ str(batch_size_train)+'/Testing_losses/'+dataset+'_'+'test', 'wb') as handle:
              pkl.dump(test_losses, handle, protocol=pkl.HIGHEST_PROTOCOL)
              print('Saved at: '+'../results/'+optimizer+'/'+'history_size'+ str(history)+'/'+'max_iters'+ str(max_iter)+'/'+ str(batch_size_train)+'/Testing_losses/'+dataset+'_'+'test')

            train_losses = []
            durations = []
            test_losses = []
    else:
      # Do not require history size of max_iters for first order methods
      for batch_size_train in batch_size_sizes:
        print('performing optimization for '+optimizer)
        print('Experiment for batch-size', batch_size_train)
        trainer = Trainer(dataset=dataset, optimizer=optimizer)
        for epoch in range(1, n_epochs + 1):
          trainer.train(epoch)
          trainer.test()

        with open(dataset+'_'+optimizer+'_train', 'wb') as handle:
          pkl.dump(train_losses, handle, protocol=pkl.HIGHEST_PROTOCOL)

        with open(dataset+'_'+optimizer+'_durations', 'wb') as handle:
          pkl.dump(durations, handle, protocol=pkl.HIGHEST_PROTOCOL)
        quasiNewton = set(['lbfgs', 'sdlbfgs', 'ARCLSR1wl'])
        if not os.path.isdir('../results/'+optimizer+'/'+ str(batch_size_train)+'/Testing_losses/'):
          os.makedirs('../results/'+optimizer+'/'+ str(batch_size_train)+'/Testing_losses/')
        
        with open('../results/'+optimizer+'/'+ str(batch_size_train)+'/Testing_losses/'+dataset+'_'+'test', 'wb') as handle:
          pkl.dump(test_losses, handle, protocol=pkl.HIGHEST_PROTOCOL)
          print('saved it in '+'../results/'+optimizer+'/'+ str(batch_size_train)+'/Testing_losses/'+dataset+'_'+'test')

        train_losses = []
        durations = []
        test_losses = []


if __name__ == '__main__':
  main()
