from abc import ABC, abstractmethod

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from dotenv import dotenv_values
from huggingface_hub import login
from PIL import Image
import torch


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


class StableDiffusionInpaintingImageModel(ImageModel):
    def __init__(self):
        MODEL = "sd-legacy/stable-diffusion-inpainting"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    def generate_image(self, **kwargs):
        pipe = self.pipe(**kwargs)
        image = pipe.images[0]
        return image


def generate_print_design(model, noun):
    image = model.generate_image(
        prompt=f"(({noun})) Large, bold, simple minimalist design modern aesthetic, (vector graphics), pinterest",
        negative_prompt="background, photorealistic, blurry, over-detailed, extra elements",
        height=512,
        width=512,
        num_inference_steps=20,
        num_images_per_prompt=1
    )
    return image


if __name__ == "__main__":
    import random

    tshirt_design_nouns = [
        "Moon", "Tiger", "Galaxy", "Mountain", "Wave", "Skull", "Sunset", "Lion", "Ocean", "Butterfly",
        "Eagle", "Tree", "Rose", "Desert", "Star", "Volcano", "Compass", "Pineapple", "Dragon", "Dreamcatcher",
        "Lightning", "Palm", "Lightning Bolt", "Unicorn", "Owl", "Cityscape", "Rocket", "Mandala", "Heart",
        "Skull", "Mermaid", "Peacock", "Abstract", "Fish", "Whale", "Bear", "Rainbow", "Cat", "Music Notes",
        "Leaf", "Mountain Range", "Wolf", "Shark", "Compass Rose", "Flower", "Feather", "Jellyfish", "Pyramid",
        "Cactus", "Rocket Ship", "Astronaut", "Nebula", "Octopus", "Lighthouse", "Guitar", "Skateboard", 
        "Campfire", "Bird", "Mushroom", "Castle", "Dragonfly", "Sunflower", "Anchor", "Crown", "Fox", "Tornado", 
        "Cloud", "Turtle", "Penguin", "Koala", "Lightning Strike", "Zebra", "Giraffe", "Whale Shark", 
        "Iceberg", "Venus", "Crescent", "Sun", "Hummingbird", "Lynx", "Panda", "Bamboo", "Raven", "Waves", 
        "Pirate Ship", "Phoenix", "Firefly", "Marijuana Leaf", "Tribal", "Giraffe Head", "Lioness", "Moth", 
        "Seahorse", "Sailboat", "T-rex", "Coyote", "Bison", "Hawk", "Peacock Feather", "Moonlit Forest", 
        "Dragon Skull", "Flower Crown", "Caveman", "Sunrise", "Wavecrest", "Horizon", "Astral", "Meteor", "Compass Rose", "Zodiac", 
        "Dream", "Butterfly Wings", "Starlight", "Palm Tree", "Vine", "Feathered Arrow", "Peach", 
        "Citrus", "Ice Cream", "Fireworks", "Rocket Launch", "Tornado", "Galaxy Swirl", "Kite", 
        "Labyrinth", "Pyramid Eye", "Crown Jewel", "Viking", "Mandalorian", "Autumn Leaves", "Neon Lights", 
        "Circuit Board", "Ancient Ruins", "Celtic Knot", "Pine Forest", "Skull Mask", "Wings of Freedom", 
        "Eclipse", "Zombie", "Vampire", "Venom", "Electric Guitar", "Gothic Cross", "Sand Dunes", 
        "Skull and Crossbones", "Street Art", "Graffiti", "Boho", "Phoenix Feather", "Jaguar", 
        "Glitch Art", "Triceratops", "Chameleon", "Shaman", "Camp Tent"
    ]
    image_model = StableDiffusion1_5_Txt2ImgModel()

    for i in range(5):
        noun = random.choice(tshirt_design_nouns)
        print(noun)
        image = generate_print_design(image_model, noun)
        image.show()
    
    """
    import os

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
    """
