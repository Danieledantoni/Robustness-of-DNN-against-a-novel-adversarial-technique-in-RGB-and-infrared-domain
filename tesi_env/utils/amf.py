from numba import njit,prange, jit
import numpy as np

@jit(parallel=True)
def amf(image, initial_window, max_window):
    """runs the Adaptive Median Filter proess on an image"""
    height, width, channels = image.shape #get the shape of the image.
    total = channels * height * width
    
    z_min, z_med, z_max, z_xy = 0, 0, 0, 0
    S_max = max_window
    S_xy = initial_window #dynamically to grow

    output_image = image.copy()
    
    if channels > 1:
      for c in prange(channels):
        for row in range(height - S_xy):
          for col in range(width - S_xy):
            filter_window = get_window(image, c, row, col, S_xy)
            z_min = np.min(filter_window) #min of intensity values
            z_max = np.max(filter_window) #max of intensity values
            z_med = np.median(filter_window) #median of intensity values
            z_xy = image[row, col, c] #current intensity

            new_intensity = first_step(image, z_min, z_med, z_max, z_xy, S_xy, S_max, row, col, c)
            output_image[row, col, c] = new_intensity
    return output_image

@jit(nopython=True)
def get_window(img, channel, row_pixel, col_pixel, window_size):
  return img[row_pixel : row_pixel + window_size, col_pixel : col_pixel + window_size, channel]

@jit(nopython=True)
def first_step(image, z_min, z_med, z_max, z_xy, S_xy, S_max, row_pix, col_pix, channel):
  if z_min < z_med < z_max:
    return second_step(z_min, z_med, z_max, z_xy)
  else:
    S_xy += 1
    if S_xy > S_max:
      return z_med
    else:
      filter_window = get_window(image, channel, row_pix, col_pix, S_xy)
      z_min = np.min(filter_window)
      z_max = np.max(filter_window)
      z_med = np.median(filter_window)
      z_xy = image[row_pix, col_pix, channel] 
      return first_step(image, z_min, z_med, z_max, z_xy, S_xy, S_max, row_pix, col_pix, channel)

@jit(nopython=True)
def second_step(z_min, z_med, z_max, z_xy):
  if z_min < z_xy < z_max:
    return z_xy
  else:
    return z_med