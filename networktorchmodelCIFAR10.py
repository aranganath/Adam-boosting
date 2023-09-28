import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pickle as pkl
from quasiadam import quasiAdam
from torchvision.models import resnet34

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

torch.manual_seed(8722110579710359886)
batch_size = 2048

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import torch.nn as nn
import torch.nn.functional as F


optimizers = ['adam', 'quasiAdam','sgd']

criterion = nn.CrossEntropyLoss()
max_iters = 10
history_size = 10

num_epochs = 30

for opt in optimizers:
  net = resnet34().to(device)
  train_losses = []
  test_losses = []
  if opt == 'sgd':
    optimizer = torch.optim.SGD(params=net.parameters(), lr=5e-1)
  
  if opt == 'sdlbfgs':
    optimizer = torch.optim.SdLBFGS(params=net.parameters(), max_iter=max_iters, history_size=history_size)

  if opt == 'lbfgs':
    optimizer = torch.optim.LBFGS(params=net.parameters())

  if opt == 'adam':
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-2, foreach=False)

  if opt == 'RMSProp':
    optimizer = torch.optim.RMSprop(params=net.parameters())

  if opt == 'Adagrad':
    optimizer = torch.optim.Adagrad(params=net.parameters())
  
  if opt == 'quasiAdam':
    optimizer = quasiAdam(params = net.parameters(), lr=1e-2, q_lr=1e-2, foreach=False)


  print('Training for '+opt+':')


  # Begin Training
  for epoch in range(num_epochs):  # loop over the dataset multiple times
      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 10 mini-batches
            testingLoss = 0.
            # Compute the testing loss
            with torch.no_grad():
              for j, testdata in enumerate(testloader, 0):
                testinputs, testlabels = testdata
                testoutputs = net(testinputs.to(device))
                testloss = criterion(testoutputs, testlabels.to(device))
                test_losses.append(testloss.item())
                testingLoss+=testloss.item()


                 
            print('[%d, %5d] avg testing loss: %.3f' %
                  (epoch + 1, i + 1, testingLoss / (j+1)))
            
            running_loss = 0.0

  with open('./results/CIFARResults/CIFAR10'+opt+'_batch_size_resnet34_'+str(batch_size), 'wb') as handle:
    print('Saving in Training Losses in:'+'./results/CIFARResults/CIFAR10_resnet34_'+opt+'_batch_size_'+str(batch_size))
    pkl.dump(train_losses, handle, protocol=pkl.HIGHEST_PROTOCOL)
  
  with open('./results/CIFARResults/CIFAR10test'+opt+'_batch_size_resnet34_'+str(batch_size), 'wb') as handle:
    print('Saving in Testing Losses in:'+'./results/CIFARResults/CIFAR10test_resnet34_'+opt+'_batch_size_'+str(batch_size))
    pkl.dump(test_losses, handle, protocol=pkl.HIGHEST_PROTOCOL)

  print('Finished Training for '+opt)