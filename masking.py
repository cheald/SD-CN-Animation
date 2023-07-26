import torch
import functools
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from torchvision import transforms
from PIL import Image, ImageFilter, ImageChops, ImageOps, ImageFilter, ImageEnhance

transform = transforms.ToPILImage()

@functools.cache
def get_new_clip_models():
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to("cuda")
    return (processor, model)


def make_masks(pos_texts, neg_texts, images):
    processor, model = get_new_clip_models()
    texts = pos_texts + neg_texts
    inputs = processor(text=texts * len(images), images=images * len(texts), padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return [get_single_mask(outputs[0], idx * len(texts), pos_texts, neg_texts) for idx, _ in enumerate(images)]

def get_single_mask(output, offset, pos_texts, neg_texts):
    # Add the positive texts
    for i, txt in enumerate(pos_texts):
        img = transform(torch.sigmoid(output[offset+i]))
        img = img.point( lambda p: min(p*1.25, 255) if p > 64 else 0 )
        add_blur = img.filter(ImageFilter.GaussianBlur(radius = 3))
        img = ImageChops.lighter(img, add_blur)

        # w = img.width
        # h = img.height
        # if (mask_padding > 0):
        #     aspect_ratio = w / float(h)
        #     new_width = w+mask_padding*2
        #     new_height = round(new_width / aspect_ratio)
        #     img = img.resize((new_width,new_height))
        #     img = center_crop(img, w, h)

        if (i > 0):
            img = ImageChops.lighter(img, final_img)
        final_img = img


    # Subtract out the negative texts
    for i, txt in enumerate(neg_texts):
        img = transform(torch.sigmoid(output[offset+i+len(pos_texts)]))
        add_blur = img.filter(ImageFilter.GaussianBlur(radius = 3))
        img = ImageChops.lighter(img, add_blur)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)

        img = ImageChops.invert(img)
        final_img = ImageChops.darker(img, final_img)

    return final_img

def get_masks(images, mask_prompt, negative_mask_prompt, width, height):
    return [img.resize((width, height)) for img in make_masks(
            mask_prompt.split(","),
            negative_mask_prompt.split(","),
            images
        )]

def apply(p, mask_prompt, negative_mask_prompt, mask_precision):
    width = p.width
    height = p.height

    masks = get_masks(
        p.init_images,
        mask_prompt, negative_mask_prompt,
        width, height
    )

    p.mode = 1
    p.mask_mode = 1
    p.inpainting_fill = 1
    p.image_mask = mask
    p.mask_for_overlay = p.image_mask
    p.latent_mask = None
