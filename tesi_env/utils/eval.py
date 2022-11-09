import torch
import numpy as np
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(epoch, loader , net, optimizer, loss_func, log_freq):

    net.train()
    running_loss=0
    correct = 0
    total = 0
    losses = []

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
        
        inputs, targets = inputs.to(device), targets.to(device)
     
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = loss_func(outputs, targets)

        running_loss += loss.item() # loss value
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1) # max per riga
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item() # elem-wise comparison from pred and labels
        
       # print statistics every log_freq mini batch
        running_loss += loss.item()
        if (batch_idx) % log_freq == 0:    # print every log_freq mini batches
            print('[Epoch : %d, Iter: %5d] loss: %.3f' %
                  (epoch + 1, batch_idx, running_loss / log_freq))
            losses.append( running_loss / log_freq)
            running_loss = 0.0
            print('Train accuracy: {} %'.format(100 * (correct / total)))
            correct=0
            total=0
        
    return losses

def validation(net, loader, loss_func):
    
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
        
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = net(inputs)
            loss = loss_func(outputs, targets)

            test_loss += loss.item()

            #accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
        
        print('Validation accuracy: {} %'.format(100 * (correct / total)))
    
    return loss

def test(net, loader):
    
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
        
            inputs, targets = inputs.to(device), targets.to(device)
            #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            
            outputs = net(inputs)

            #accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
    
        print('Test accuracy: {} %'.format(100 * (correct / total)))