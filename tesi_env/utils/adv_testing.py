import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm
import random
import time
import numpy as np

class adversarial_test:
  def __init__(self, model, attack, data):
    self.data = data
    self.attack = attack
    self.model = model

  def basic_attack(self, type_of_attack, max_attacks, stats = False, detection = False):

    self.model.eval()
    current_attacks = 0
    total_time_required = 0
    total_pixel_moved = []
    total_iter_required = []
    number_img = 0
    corr_class_after_attack = 0
    successful_attacks = 0

    if type_of_attack != 'black' and type_of_attack != 'white':
      raise ValueError('Type of attacks can only be: black or white')
    
    if len(next(iter(self.data))) > 2:
        raise ValueError('The function requires the dataset containing only one image domain')

    if type_of_attack == 'black':
      print('{} attacked by black-box Pixle'.format(type(self.model).__name__))

      for batch_idx, (inputs, gt_labels) in enumerate(tqdm(self.data)):
            
        inputs, gt_labels = inputs.to(device), gt_labels.to(device)
        if detection:
          outputs, to_keep = self.model(inputs)
          inputs = inputs[to_keep]
          gt_labels = gt_labels[to_keep]
        else:
          outputs = self.model(inputs)
        _, predictions = torch.max(outputs.data, 1)
        # attack only correct classified images
        corr_class_img = [i for i in range(len(inputs)) if predictions[i] == gt_labels[i]]

        # ATTACK IMAGES
        image_to_attack = []

        if current_attacks <= max_attacks:
          att_in_this_batch = random.randint(1, len(corr_class_img))
          if (current_attacks + att_in_this_batch) > max_attacks:
            att_in_this_batch = max_attacks - current_attacks
          
          current_attacks += att_in_this_batch
          image_to_attack = random.sample(corr_class_img, att_in_this_batch)

          for idx_img in image_to_attack:

            start = time.process_time_ns()
            adv_img = self.attack(inputs[idx_img].unsqueeze(0), gt_labels[[idx_img]])
            end = time.process_time_ns()
            total_time_required += (end - start)
            total_pixel_moved += self.attack.l_0_norm
            total_iter_required += self.attack.required_iterations
            inputs[idx_img] = adv_img

        else:
          pass

        if detection:
          outputs, labels_to_keep = self.model(inputs)
          if len(outputs) == 0:
            pass
          else:
            _, predictions = torch.max(outputs.data, 1)
            gt_labels = gt_labels[labels_to_keep]
            number_img += gt_labels.size(0)
            corr_class_after_attack += predictions.eq(gt_labels.data).cpu().sum().item()
            successful_attacks += (predictions[image_to_attack] != gt_labels[image_to_attack]).sum().item()

        else:
          outputs = self.model(inputs)
          _, predictions = torch.max(outputs.data, 1)
          number_img += gt_labels.size(0)
          corr_class_after_attack += predictions.eq(gt_labels.data).cpu().sum().item()
          successful_attacks += (predictions[image_to_attack] != gt_labels[image_to_attack]).sum().item()

      # evaluating stats
      if stats:
        pixel_moved = [total_pixel_moved[i].item() for i in range(len(total_pixel_moved))]
        time_required = (total_time_required / 1e+9) / current_attacks
        print('L0 norm: {} avg, {} var'.format(np.mean(pixel_moved), np.std(pixel_moved)))
        print('Queries required: {} avg, {} var'.format(np.mean(total_iter_required), np.std(total_iter_required)))
        print('Average time required: {} sec.'.format(time_required))
              
      print('Test accuracy: {} %'.format(100 * (corr_class_after_attack / number_img)))
      print('Generated attacks {}'.format(current_attacks))
      print('Success rate: {} %'.format(100 * (successful_attacks / current_attacks)))

    if type_of_attack == 'white':
      print('{} attacked by Wixle'.format(type(self.model).__name__))

      for batch_idx, (inputs, gt_labels) in enumerate(tqdm(self.data)):
            
        inputs, gt_labels = inputs.to(device), gt_labels.to(device)
        if detection:
          outputs, to_keep = self.model(inputs)
          inputs = inputs[to_keep]
          gt_labels = gt_labels[to_keep]
        else:
          outputs = self.model(inputs)
        _, predictions = torch.max(outputs.data, 1)
        # attack only correct classified images
        corr_class_img = [i for i in range(len(inputs)) if predictions[i] == gt_labels[i]]

        # ATTACK IMAGES
        image_to_attack = []

        if current_attacks <= max_attacks:
          att_in_this_batch = random.randint(1, len(corr_class_img))
          if (current_attacks + att_in_this_batch) > max_attacks:
            att_in_this_batch = max_attacks - current_attacks
          
          current_attacks += att_in_this_batch
          image_to_attack = random.sample(corr_class_img, att_in_this_batch)

          for idx_img in image_to_attack:
            start = time.process_time_ns()
            adv_img = self.attack(inputs[idx_img].unsqueeze(0), gt_labels[[idx_img]])
            end = time.process_time_ns()
            total_time_required += (end - start)
            total_iter_required += self.attack.required_iterations
            inputs[idx_img].data = adv_img.data

        else:
          pass

        if detection:
          outputs, labels_to_keep = self.model(inputs)
          if len(outputs) == 0:
            pass
          else:
            _, predictions = torch.max(outputs.data, 1)
            gt_labels = gt_labels[labels_to_keep]
            number_img += gt_labels.size(0)
            corr_class_after_attack += predictions.eq(gt_labels.data).cpu().sum().item()
            successful_attacks += (predictions[image_to_attack] != gt_labels[image_to_attack]).sum().item()

        else:
          outputs = self.model(inputs)
          _, predictions = torch.max(outputs.data, 1)
          number_img += gt_labels.size(0)
          corr_class_after_attack += predictions.eq(gt_labels.data).cpu().sum().item()
          successful_attacks += (predictions[image_to_attack] != gt_labels[image_to_attack]).sum().item()

      if stats:
        time_required = (total_time_required / 1e+9) / current_attacks
        print('Queries required: {} avg, {} var'.format(np.mean(total_iter_required), np.std(total_iter_required)))
        print('Average time required: {} sec.'.format(time_required))

      print('Test accuracy: {} %'.format(100 * (corr_class_after_attack / number_img)))
      print('Generated attacks {}'.format(current_attacks))
      print('Success rate: {} %'.format(100 * (successful_attacks / current_attacks)))