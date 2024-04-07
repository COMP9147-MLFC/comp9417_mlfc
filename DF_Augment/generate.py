import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from os import listdir
from os.path import isfile, join

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

def get_image(url):
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

prompt = "same image"

images_path = "./landscape_orig"
files =  [f for f in listdir(images_path) if isfile(join(images_path, f))]

for f in files[130:]:
  image_final_path = images_path + "/" + f
  image_output_path = "./landscape_gen" + "/" + f + "" 
  im = get_image(image_final_path)
  images = pipe(prompt, image=im, num_inference_steps=10, image_guidance_scale=1).images
  images[0].save(image_output_path.replace(".jpg", "_ai.jpg"))