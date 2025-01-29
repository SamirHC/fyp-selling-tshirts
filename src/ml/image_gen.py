from abc import ABC, abstractmethod

from diffusers import StableDiffusionPipeline
from dotenv import dotenv_values
from huggingface_hub import login
from PIL import Image


config = dotenv_values(".env")
HUGGING_FACE_TOKEN = config["HUGGING_FACE_TOKEN"]
login(HUGGING_FACE_TOKEN)


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


if __name__ == "__main__":
    image_model = StableDiffusion1_5_Txt2ImgModel()
    image = image_model.generate_image(
        prompt="Minimalist T-shirt design of planet Earth, modern aesthetic",
        negative_prompt="blurry, messy, over-detailed, extra elements, complex, photorealistic",
        height=768,
        width=768,
        num_inference_steps=50,
        num_images_per_prompt=1
    )
    image.show()
