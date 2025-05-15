from abc import ABC, abstractmethod
import base64
from datetime import datetime
import os

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from huggingface_hub import login
from openai import OpenAI
from PIL import Image
import torch

from src.common import utils, config


login(config.HUGGING_FACE_TOKEN)


class ImageModel(ABC):
    @abstractmethod
    def generate_image(self, **kwargs):
        pass


class DummyImageModel(ImageModel):
    def generate_image(self, **kwargs):
        return Image.new("RGBA", (100, 100), "red")


class StableDiffusion1_5_Txt2ImgModel(ImageModel):
    def __init__(self):
        MODEL = "sd-legacy/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(MODEL)
        self.pipe = self.pipe.to("cuda")

    def generate_image(self, **kwargs):
        pipe = self.pipe(**kwargs)
        image = pipe.images[0]
        return image


class StableDiffusion2_1_Txt2ImgModel(ImageModel):
    def __init__(self):
        MODEL = "stabilityai/stable-diffusion-2-1"
        self.pipe = StableDiffusionPipeline.from_pretrained(MODEL)
        self.pipe = self.pipe.to("cuda")

    def generate_image(self, **kwargs):
        pipe = self.pipe(**kwargs)
        image = pipe.images[0]
        return image


class StableDiffusionInpaintingImageModel(ImageModel):
    def __init__(self):
        MODEL = "sd-legacy/stable-diffusion-inpainting"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    def generate_image(self, **kwargs):
        pipe = self.pipe(**kwargs)
        image = pipe.images[0]
        return image


class OpenAIDallE3ImageModel(ImageModel):
    def __init__(self):
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
        )
        self.model = "dall-e-3"
        self.response_format = "b64_json"

    def generate_image(self, **kwargs):
        response = self.client.images.generate(
            prompt=kwargs["prompt"],
            model=self.model,
            n=1,
            response_format=self.response_format
        )
        image_data = response.data[0].b64_json
        image_bytes = base64.b64decode(image_data)
        with open(os.path.join("out",f"openai_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.png"), "wb") as f:
            f.write(image_bytes)
            
        image = utils.base64_to_image(image_data)
        return image


if __name__ == "__main__":
    """
    image_model = StableDiffusion1_5_Txt2ImgModel()
    image = image_model.generate_image(
        prompt="Minimalist T-shirt design of planet Earth, modern aesthetic",
        negative_prompt="blurry, messy, over-detailed, extra elements, complex, photorealistic",
        height=768,
        width=768,
        num_inference_steps=50,
        num_images_per_prompt=1
    )
    """

    image = None
    if config.GPU == 0:
        if config.PAYMENT_ACTIVE:
            image_model = OpenAIDallE3ImageModel()
        else:
            image_model = DummyImageModel()
        image = image_model.generate_image(prompt="Minimalist planet Earth T-shirt design, modern aesthetic")

    elif config.GPU == 1:
        empty_tshirt_image_path = os.path.join("data", "images", "EmptyTshirt2.png")
        empty_tshirt_image = Image.open(empty_tshirt_image_path).resize((512, 512), Image.Resampling.LANCZOS).convert("RGB")
        mask_path = os.path.join("data", "images", "EmptyTshirt2Mask.png")
        mask = Image.open(mask_path).resize((512, 512), Image.Resampling.LANCZOS).convert("1")

        image_model = StableDiffusionInpaintingImageModel()
        image = image_model.generate_image(
            prompt="Minimalist planet Earth T-shirt, modern aesthetic",
            negative_prompt="blurry, messy, over-detailed, extra elements, complex, photorealistic",
            image=empty_tshirt_image,
            mask_image=mask,
            strength=0.95,
        )

    try:
        image.show()
    except Exception as e:
        print(f"Image could not be shown: {e}")
