
import torch
import numpy as np
from PIL import Image

def resize_image(image)->Image.Image:
    pixel_number = 1024*1024
    granularity_val = 64
    ratio = image.size[0] / image.size[1]
    width = int((pixel_number * ratio) ** 0.5)
    width = width - (width % granularity_val)
    height = int(pixel_number / width)
    height = height - (height % granularity_val)
    return image.resize((width, height))

def get_masked_background_image(image, image_mask)->tuple:
    image_mask_pil = image_mask.resize(image.size) # fg is white
    image = np.array(image.convert("RGB")).transpose(2, 0, 1).astype(np.float32) / 255.0
    image_mask = np.array(image_mask_pil.convert("L")).astype(np.float32) / 255.0
    image[:,image_mask < 0.5] = 0  # mask background
    return image, image_mask

def get_control_image_tensor(vae, image, mask)->torch.Tensor:
        masked_image, image_mask = get_masked_background_image(image, mask)
        masked_image_tensor = torch.from_numpy(masked_image)
        masked_image_tensor = (masked_image_tensor - 0.5) / 0.5 # normalize for vae
        masked_image_tensor = masked_image_tensor.unsqueeze(0).to(device="cuda:0")
        # encode the image to get the control latents
        control_latents = vae.encode(  
                masked_image_tensor[:, :3, :, :].to(vae.dtype)
            ).latent_dist.sample()   
        control_latents = control_latents * vae.config.scaling_factor 

        mask_tensor = torch.tensor(image_mask, dtype=torch.float32)[None, None, ...].to(device="cuda:0")
        mask_tensor = torch.where(mask_tensor > 0.5, 1.0, 0) # binarize the mask
        mask_resized = torch.nn.functional.interpolate(mask_tensor, size=(control_latents.shape[2], control_latents.shape[3]), mode='nearest')
        control_tensor = torch.cat([control_latents, mask_resized], dim=1)
        return control_tensor

def remove_bg_from_image(image_path: str)->Image.Image:
    from transformers import pipeline
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    mask = pipe(image_path, return_mask = True) # outputs a pillow mask
    return mask

def paste_fg_over_image(gen_image: Image.Image, orig_image: Image.Image, fg_mask: Image.Image)->Image.Image:
    fg_mask = fg_mask.convert("L")
    fg_mask = fg_mask.resize(orig_image.size, Image.NEAREST)
    gen_image = gen_image.convert("RGBA")
    orig_image = orig_image.convert("RGBA")
    gen_image.paste(orig_image, (0, 0), fg_mask)
    return gen_image.convert("RGB")