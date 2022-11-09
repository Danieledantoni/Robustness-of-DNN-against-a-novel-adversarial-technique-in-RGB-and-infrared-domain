from itertools import chain

import numpy as np
import torch
from torch.nn.functional import softmax, cross_entropy
from torchattacks.attack import Attack

class RandomWhitePixle(Attack):

    def __init__(self, model,
                 pixel_to_swap_per_image_transfer = None,
                 pixels_per_iteration=1,
                 mode='htl',
                 return_solutions = False,
                 transfer = False,
                 average_channels=True,
                 restarts=20,
                 iterations=100,
                 **kwargs):

        super().__init__("Pixle", model)

        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError('restarts must be and integer >= 0 '
                             '({})'.format(restarts))

        assert mode in ['htl', 'lth']

        self.mode = mode
        self.iterations = iterations
        self.pixels_per_iteration = pixels_per_iteration
        self.return_solutions = return_solutions

        self.restarts = restarts
        self.transfer = transfer
        self.pixel_to_swap_per_image_transfer = pixel_to_swap_per_image_transfer
        self.average_channels = average_channels

        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):

        shape = images.shape

        if len(shape) == 3:
            images = images[None]
            c, h, w = shape
        else:
            _, c, h, w = shape

        images = images.to(self.device)
        labels = labels.to(self.device)

        images.requires_grad = True

        adv_images = []
        swapped_pixels = []
        iterations = []
        statistics = []
        image_probs = []

        for img_i in range(len(images)):
            img = images[img_i]
            label = labels[img_i]

            loss_f, callback_f = self._get_fun(label, target_attack=False)

            best_adv_image = img.clone()
            image_swapped_pixels = []

            im_iterations = 0
          
            if self.transfer == False:
              for restart_i in range(self.restarts):

                  loss = cross_entropy(self.model(img[None]), label[None],
                                      reduction='none')
                  self.model.zero_grad()
                  img.grad = None

                  data_grad = torch.autograd.grad(loss, img,
                                                  retain_graph=False,
                                                  create_graph=False)[0]

                  data_grad = torch.abs(data_grad)

                  if self.average_channels:
                      data_grad = data_grad.mean(0)
                      shape = (h, w)
                  else:
                      shape = (c, h, w)

                  if isinstance(self.pixels_per_iteration, float):
                      pixels_per_iteration = int(
                          self.pixels_per_iteration * (h * w))
                  else:
                      pixels_per_iteration = self.pixels_per_iteration

                  data_grad = data_grad.view(-1)
                  probs = data_grad / data_grad.sum(-1, keepdim=True)

                  data_grad = 1 / data_grad
                  invert_probs = data_grad / data_grad.sum(-1, keepdim=True)

                  if self.mode == 'htl':
                      source_prob = probs
                      dest_prob = invert_probs
                  else:
                      source_prob = invert_probs
                      dest_prob = probs

                  indexes = np.arange(len(source_prob))

                  source_prob = source_prob.detach().cpu().numpy()
                  dest_prob = dest_prob.detach().cpu().numpy()

                  source_prob = np.nan_to_num(source_prob, posinf=0.0, neginf=0.0)
                  dest_prob = np.nan_to_num(dest_prob, posinf=0.0, neginf=0.0)

                  if dest_prob.sum() == 0.0 or source_prob.sum() == 0.0:
                      break

                  stop = False
                  best_solution = None
                  best_loss = loss_f(best_adv_image)

                  patch_i = 0

                  for patch_i in range(self.iterations):
                      pert_image = best_adv_image.clone()

                      selected_indexes1 = np.random.choice(indexes,
                                                          pixels_per_iteration,
                                                          False, source_prob)

                      selected_indexes2 = np.random.choice(indexes,
                                                          pixels_per_iteration,
                                                          False, dest_prob)

                      aa = [np.unravel_index(_a, shape) for _a in selected_indexes1]
                      bb = [np.unravel_index(_b, shape) for _b in selected_indexes2]

                      for a, b in zip(aa, bb):
                          if self.average_channels:
                              pert_image[:, b[0], b[1]] = img[:, a[0], a[1]]
                              # if self.swap:
                              #     adv_img[:, a[0], a[1]] = v
                          else:
                              pert_image[b[0], b[1], b[2]] = img[
                                  a[0], a[1], a[2]]
                              # if self.swap:
                              #     adv_img[a[0], a[1], a[2]] = v

                      l = loss_f(pert_image)

                      if l < best_loss:
                          best_loss = l
                          best_solution = (selected_indexes1, selected_indexes2)

                      image_probs.append(best_loss)

                      if callback_f(pert_image):
                          best_solution = (selected_indexes1, selected_indexes2)
                          stop = True
                          break

                  im_iterations += patch_i
                  if best_solution is not None:
                      image_swapped_pixels.append(best_solution)

                      selected_indexes1, selected_indexes2 = best_solution
                      
                      aa = [np.unravel_index(_a, shape) for _a in
                            selected_indexes1]
                      bb = [np.unravel_index(_b, shape) for _b in
                            selected_indexes2]
                      
                      for a, b in zip(aa, bb):
                          
                          if self.average_channels:
                              best_adv_image[:, b[0], b[1]] = img[:, a[0], a[1]]
                              # if self.swap:
                              #     adv_img[:, a[0], a[1]] = v
                          else:
                              best_adv_image[b[0], b[1], b[2]] = img[
                                  a[0], a[1], a[2]]
                              # if self.swap:
                              #     adv_img[a[0], a[1], a[2]] = v

                      img = best_adv_image.clone()

                  if stop:
                      break

              iterations.append(im_iterations)
              statistics.append(image_probs)

              swapped_pixels.append(image_swapped_pixels)
              adv_images.append(best_adv_image.detach())
            
            else:
              for i in range(len(self.pixel_to_swap_per_image_transfer[0])):

                selected_indexes1, selected_indexes2 = self.pixel_to_swap_per_image_transfer[0][i]
                
                
                aa = [np.unravel_index(_a, (224, 224)) for _a in
                            selected_indexes1]
                bb = [np.unravel_index(_b, (224, 224)) for _b in
                            selected_indexes2]
                
                for a, b in zip(aa, bb):
                          
                  if self.average_channels:
                      best_adv_image[:, b[0], b[1]] = img[:, a[0], a[1]]
                      # if self.swap:
                      #     adv_img[:, a[0], a[1]] = v
                  else:
                      best_adv_image[b[0], b[1], b[2]] = img[
                                  a[0], a[1], a[2]]
                      # if self.swap:
                      #     adv_img[a[0], a[1], a[2]] = v

                img = best_adv_image.clone()
              adv_images.append(best_adv_image.detach())
        self.required_iterations = iterations

        adv_images = torch.stack(adv_images, 0)

        if self.return_solutions:
            return adv_images, swapped_pixels

        return adv_images

    def _get_prob(self, image):
        out = self.model(image.to(self.device))
        prob = softmax(out, dim=1)
        return prob.detach().cpu().numpy()

    def loss(self, img, label, target_attack=False):

        p = self._get_prob(img)
        p = p[np.arange(len(p)), label]

        if target_attack:
            p = 1 - p

        return p.sum()

    def _get_fun(self, label, target_attack=False):
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        @torch.no_grad()
        def func(img, **kwargs):

            if len(img.shape) == 3:
                img = img[None, :]

            p = self._get_prob(img)
            p = p[np.arange(len(p)), label]

            if target_attack:
                p = 1 - p

            return p.sum()

        @torch.no_grad()
        def callback(img, **kwargs):

            if len(img.shape) == 3:
                img = img[None, :]

            p = self._get_prob(img)[0]
            mx = np.argmax(p)

            if target_attack:
                return mx == label
            else:
                return mx != label

        return func, callback

    def _perturb(self, source, solution, destination=None):
        if destination is None:
            destination = source

        c, h, w = source.shape[1:]

        x, y, xl, yl = solution[:4]
        destinations = solution[4:]

        source_pixels = np.ix_(range(c),
                               np.arange(y, y + yl),
                               np.arange(x, x + xl))

        indexes = torch.tensor(destinations)
        destination = destination.clone().detach().to(self.device)

        s = source[0][source_pixels].view(c, -1)

        destination[0, :, indexes[:, 0], indexes[:, 1]] = s

        return destination