import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from attacks import white_pixle as wixle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def adv_train(epoch, loader , net, optimizer, loss_func, log_freq):

    net.train()
    running_loss=0
    correct = 0
    total = 0
    losses = []

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
        
        inputs, targets = inputs.to(device), targets.to(device)

        for idx, image in enumerate(inputs):

            prob = random.randint(0, 1)
            if prob > 0.5:
                percentage_of_pixel = random.randint(5, 40)
                attack = wixle(pixels_per_iteration = ((percentage_of_pixel / 100) * (224 * 224)), restarts = 100,
                 iterations = 50, model= net)
                # attack the image
                adv_image = attack(imagwe, targets[[idx]])
                inputs[idx].data = adv_image.data

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