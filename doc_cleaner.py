import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as Im 
from PIL import ImageShow, ImageOps, ImageEnhance
import argparse

def get_arguments():  #argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputs_dir', type=str, default='./inputs', help='Input Directory Path')
	parser.add_argument('--outputs_dir', type=str, default='./outputs', help='Output Directory Path')
	parser.add_argument("--contrast_val", type= int, default = 3, help = 'Contrast Value (def=3)')
    parser.add_argument("--threshold_area", type= int, default = None, help = 'Threshold Area (def: 1/20th of image area)')
	args = parser.parse_args()
	return args
	


def unshadow(img):

  rgb_planes = cv2.split(img)
  result_planes = []
  result_norm_planes = []
  for plane in rgb_planes:
      dilated_img = cv2.dilate(plane, np.ones((3,3), np.uint8))   #small kernel size used to avoid colour bleed on the reconstructed image
      bg_img = cv2.medianBlur(dilated_img, 101)            #kernal size of 81 has been arrived at 81 after finetuning based on input image size that was set.
      #bg_img = dilated_img
      diff_img = 255 - cv2.absdiff(plane, bg_img)
      norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      result_planes.append(diff_img)
      result_norm_planes.append(norm_img)

  result = cv2.merge(result_planes)
  result_norm = cv2.merge(result_norm_planes)
  return result, result_norm
  

def contrast_enhance(image, contrast):
	im = Im.fromarray(image)

	# image contrast 'enhancer'
	enhancer = ImageEnhance.Contrast(im)

	factor = contrast  # increase contrast default 3x
	im_output = enhancer.enhance(factor)

	return im_output




def resize_img(img, target_ht):

	ratio_ = img.shape[1]/img.shape[0]
	new_wd = round(ratio_*target_ht)
	img_pil = Im.fromarray(img)
	img_resized = img_pil.resize((new_wd, target_ht))
	output_img = np.array(img_resized)
	return output_img

def get_contours(img, resize_ratio):

    tar_h, tar_w = resize_ratio
	
	new_wd = round(ratio_*target_ht)
	img_pil = Im.fromarray(img)
	img_resized = img_pil.resize((new_wd, target_ht))
	output_img = np.array(img_resized)
	return output_img




    
    
def run_enhancer():

	args = get_arguments()
	input_dir = args.inputs_dir
	output_dir = args.outputs_dir
	contrast = args.contrast_val


    img_p = input_dir + '/' + name
    img = cv2.imread(img_p, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_img(img, target_ht=1200)

    _,unshadow_img = unshadow(img)   #remove image shadow


    sharpen = cv2.GaussianBlur(unshadow_img, (0,0), 3)             #sharpen text/edges
    sharpen = cv2.addWeighted(unshadow_img, 1.5, sharpen, -0.5, 0)


    im_output = contrast_enhance(sharpen, contrast)   #increase contrast
    final_im = np.array(im_output)

        
    return final_im




def resize_image_ratio(img, height, width, h_scale, w_scale):
  img = cv2.resize(img, dsize=(width*(w_scale),height*(h_scale)), interpolation=cv2.INTER_CUBIC)
  return img

def get_contours_from_image(img, lower_thresh = 127):
  ret,thresh = cv2.threshold(np.uint8(img),lower_thresh,255,cv2.THRESH_BINARY_INV)
  
  kernel = np.ones((2,2), np.uint8)
  img_dilation = cv2.dilate(thresh, kernel, iterations=1)
  gsblur=cv2.GaussianBlur(img_dilation,(5,5),0)

  ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return ctrs





def cleanup(img, resize_ratio = (4,5)):
  
  args = get_arguments()
  area_thr = args.threshold_area
  h_scale, w_scale = resize_ratio
  orig_h = img.shape[0]
  orig_w = img.shape[1]
  img = resize_image_ratio(img, orig_h, orig_w, h_scale, w_scale)
  ctrs = get_contours_from_image(img)
  sorted_c = sorted(ctrs, key= lambda x: cv2.contourArea(x), reverse= True)

  t_area_new = (h_scale*orig_h)*(orig_w * w_scale)
  if not area_thr:
    area_thr = t_area_new*(1/20)
  else:
    area_thr = area_thr*(h_scale)*(w_scale)

  for i in range(len(sorted_c)):
    x,y,w,h = cv2.boundingRect(sorted_c[i])
    if w*h<=area_thr:
      for jk in range(x,x+w):
        for kk in range(y, y+h):
          img[kk, jk] = 255

  img = resize_image_ratio (img, orig_h, orig_w, 1,1)
  
  return img


def save_im(img, f_name):
	args = get_arguments()
    input_dir = args.inputs_dir
	output_dir = args.outputs_dir
    files = os.listdir(input_dir)
    f_name = files[0]
    save_p = output_dir + '/' + 'cleaned_' + name    #save outputs
    cv2.imwrite(save_p, img) 

def main():
    im = run_enhancer()
    

if __name__ == '__main__':
    main()
