import torch
def load_model(model, opt, path_name):
  if torch.cuda.is_available() == False:
    checkpoint = torch.load(path_name, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
  else:
    checkpoint = torch.load(path_name, map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
  print('The model {} is loaded successfully.'.format(path_name.split('/')[-1]))

import json
def save_model(model, opt, loss, epoch, path, name):
  info = {}
  # store some infos about the model
  info[name] = {'pre_trained' : 'yes',
                                         'loss' : type (loss).__name__,
                                         'epochs' : epoch + 1,
                                         'optimizer' : type (opt).__name__,
                                         'learn. rate' : opt.defaults['lr']}
  with open('{0}/info_{1}.json'.format(path, name[:-3]), 'w') as f:
    json.dump(info, f)
  path_name = '{0}/{1}'.format(path, name)
  torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': opt.state_dict(),
              'loss': loss,
              }, path_name)
  print('The model {} is saved successfully.'.format(name))

def load_generator(model, opt, path_name):
    checkpoint = torch.load(path_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_gen_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['bce']
    print('The model {} is loaded successfully.'.format(path_name.split('/')[-1]))
