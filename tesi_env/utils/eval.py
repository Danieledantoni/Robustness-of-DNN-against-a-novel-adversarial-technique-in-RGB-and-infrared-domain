import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
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

def train_generative_model(generator, discriminator, binary_ce, mse_loss, double_img_loader):
    print_str = "\t".join(["{}"] + ["{:.6f}"] * 2)
    print("\t".join(["{:}"] * 3).format("Epoch", "Gen_Loss", "Dis_Loss"))
    from torch.autograd import Variable
    for epoch in range(5):
        generator.train()
        discriminator.train()

        gen_loss, dis_loss, n = 0, 0, 0
        for batch_idx, (img, adv_img, target, adv_target) in enumerate(tqdm(double_img_loader)):

            img, adv_img = img.to(device), adv_img.to(device)
            target, adv_target = target.to(device), adv_target.to(device)
            current_size = img.size(0)

            # real and fakes targets
            t_real = Variable(torch.ones(current_size).cuda())
            t_fake = Variable(torch.zeros(current_size).cuda())
            y_real = dis(img).squeeze()
            adv_fake = gen(adv_img)
            y_fake = dis(adv_fake).squeeze()

            loss_D = binary_ce(y_real, t_real) + binary_ce(y_fake, t_fake)
            opt_dis.zero_grad()
            loss_D.backward(retain_graph = True)
            opt_dis.step()

            # double train the generator in order to make faster both discriminator and generator convergence
            for _ in range(2):
                x_fake = gen(adv_img)
                x_fake = x_fake.clone()
                y_fake = dis(adv_fake).squeeze()
                y_fake = y_fake.clone()

                loss_G = 0.7 * mse_loss(adv_fake, img) + 0.3 * binary_ce(y_fake, t_real)
                opt_gen.zero_grad()
                loss_G.backward(retain_graph = True)

            opt_gen.step()

            gen_loss += loss_G.data * img.size(0)
            dis_loss += loss_D.data * img.size(0)
            n += img.size(0)
        print(print_str.format(epoch, gen_loss / n, dis_loss / n))
