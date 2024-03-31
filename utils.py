import matplotlib.pyplot as plt
import cv2
import copy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation



def read_image(image_path):
  image = cv2.imread(f"{image_path}" )
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = torch.tensor(image)
  plt.figure(figsize=(5,5))
  plt.imshow(image)
  plt.axis('on')
  plt.show()
  return image

def merge_images(image, shifted_image, shifted_mask):
    binary_mask = (shifted_mask > 0).float()
    merged_image = image.clone()

    for channel in range(3):  # Loop through each color channel
        merged_image[..., channel] = image[..., channel] * (1 - binary_mask) + shifted_image[..., channel]/255 * binary_mask
    return merged_image



def resize_image(image, shape, n_channel):
  output_size = shape
  if n_channel == 1:
    image = image.unsqueeze(-1)

  image_permute = image.permute(2,0,1)
  resized_image_3d_tensor = torch.nn.functional.interpolate(image_permute.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False).squeeze(0)
  resized_image_3d_tensor = resized_image_3d_tensor.permute(1,2,0)

  if n_channel == 1:
    resized_image_3d_tensor = resized_image_3d_tensor.squeeze(-1)
  return resized_image_3d_tensor
def move_content(image,mask, x_shift, y_shift):
    height, width = mask.shape
    shifted_image = image
    shifted_mask = torch.zeros_like(mask)
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 1:
                new_x = min(width - 1, max(0, x + x_shift))  # Ensure new_x is within bounds
                new_y = min(height - 1, max(0, y - y_shift))  # Ensure new_y is within bounds

                shifted_mask[new_y, new_x] = 1
                shifted_image[new_y,new_x,0] = shifted_image[y,x,0]
                shifted_image[new_y,new_x,1] = shifted_image[y,x,1]
                shifted_image[new_y,new_x,2] = shifted_image[y,x,2]
                shifted_image[y,x,0] = 0
                shifted_image[y,x,1] = 0
                shifted_image[y,x,2] = 0

    return shifted_image, shifted_mask



def add_padding_mask(mask, k):
    # Create a kernel representing the neighborhood around each one
    kernel_size = 2 * k + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32).to(mask.device)

    # Use 2D convolution to surround each one with ones
    surrounded_tensor = F.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel, padding=k).squeeze().byte()
    surrounded_tensor = (surrounded_tensor > 0).to(float)

    return surrounded_tensor



def apply_mask_to_image(image, mask, opacity=0.4):
    # Convert mask to red overlay
    red_overlay = torch.zeros_like(image, dtype=torch.float32)
    red_overlay[mask >= .9] = torch.tensor([1.0, 0.0, 0], dtype=torch.float32)  # Set red color where mask is 1
    
    # Blend the original image and the red overlay using alpha blending
    masked_image =  image/255 + opacity * red_overlay
    # masked_image = torch.tensor(masked_image*255, dtype = torch.uint8)
    return masked_image

    # # Example usage
    # image_tensor = image  # Example 3D image tensor
    # binary_mask = discrete_mask  # Example 2D binary mask

    # # Apply mask to image
    # masked_image = apply_mask_to_image(image_tensor, binary_mask)

    # # Display the masked image (you can use any method to display or save tensor as image)
    # plt.imshow(masked_image)


def tensor_to_pil(tensor):
  to_pil = ToPILImage()
  temp_image = None
  if tensor.dim() == 3:
    temp_image = tensor.permute(2,0,1)
    temp_image = to_pil(temp_image)
  elif tensor.dim() ==2:
    temp_image = to_pil(tensor)
  return temp_image



def load_segment_pipeline():
  processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
  model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
  return processor, model

def get_mask(image, prompts, processor, model):
  inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
  # predict
  with torch.no_grad():
    outputs = model(**inputs)
    print(outputs.logits.size())

  preds = outputs.logits
  if outputs.logits.dim() == 2:
    preds = outputs.logits.unsqueeze(0)
  return preds[0]

def load_inpaint_pipeline():
  # stabilityai/stable-diffusion-2-inpainting
  # pipeline = AutoPipelineForInpainting.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16").to("cuda")
  pipeline = AutoPipelineForInpainting.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16, variant="fp16").to("cuda")

  pipeline.enable_model_cpu_offload()
  return pipeline

def inpaint_image(pipeline, prompt, image, mask, temperature, strength, guidance_scale, negative_prompt):
  pipeline.config.temperature = temperature
  inpaint_image = pipeline(prompt=prompt, image=image, mask_image=mask, strength = strength, guidance_scale = guidance_scale, negative_prompt = negative_prompt).images[0]
  return inpaint_image
