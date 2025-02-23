import random

import pandas as pd
from PIL import Image, ImageOps

from src.common import utils
from src.ml import image_gen
from src.design_generation import image_edit


def generate_central_mockup(image_data: pd.Series, design: Image.Image) -> tuple[Image.Image, Image.Image]:
    mockup = utils.get_image_from_url(image_data["Mockup Image"]).convert("RGBA")

    x = image_data["x"]
    y = image_data["y"]
    width = image_data["width"]
    height = image_data["height"]
    size = (width, height)

    mask = Image.new("1", mockup.size)
    scaled_design = ImageOps.contain(design.convert("RGBA"), size)

    paste_x = x + width//2 - scaled_design.width//2
    paste_y = y + height//2 - scaled_design.height//2

    mockup.paste(scaled_design, (paste_x, paste_y), scaled_design)
    mask.paste(Image.new("1", scaled_design.size, color=1), (paste_x, paste_y), scaled_design)

    return mockup, mask


def generate_plain_mockup(image_data: pd.Series) -> tuple[Image.Image, Image.Image]:
    mockup = utils.get_image_from_url(image_data["Mockup Image"]).convert("RGBA")
    mask = Image.new("1", mockup.size)

    return mockup, mask


def generate_stable_diffusion_print_design_mockup(image_data: pd.Series):
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
    image_model = image_gen.StableDiffusion1_5_Txt2ImgModel()

    noun = random.choice(tshirt_design_nouns)
    design = image_gen.generate_print_design(image_model, noun)
    design = image_edit.remove_bg(design)

    return generate_central_mockup(image_data, design)


def procure_data_from_few_designs():
    import os

    mockup_data_path = os.path.join("data", "images", "T-Shirt Mockups - Sheet1.csv")
    df = pd.read_csv(mockup_data_path)

    mask_image_path = os.path.join("data", "images", "tshirt_segmentation", "masks")
    mockup_image_path = os.path.join("data", "images", "tshirt_segmentation", "mockups")
    design_image_path = os.path.join("data", "images", "tshirt_segmentation", "designs")
    designs = dict(
        (design_path[:-4], Image.open(os.path.join(design_image_path, design_path))) 
        for design_path in os.listdir(design_image_path)
    )

    augmentations = {"mirror": ImageOps.mirror}

    for i in range(len(df)):
        print(i, df.iloc[i]["Mockup Image"])
        try:
            # Plain, no mask
            mockup, mask = generate_plain_mockup(df.iloc[i])
            mockup = mockup.resize((256, 256))
            mask = mask.resize((256, 256))
            mockup.save(os.path.join(mockup_image_path, f"{i}_plain_mockup.png"), "PNG")
            mask.save(os.path.join(mask_image_path, f"{i}_plain_mask.png"), "PNG")

            for design_name, design in designs.items():
                # Central
                mockup, mask = generate_central_mockup(df.iloc[i], design)
                mockup = mockup.resize((256, 256))
                mask = mask.resize((256, 256))
                mockup.save(os.path.join(mockup_image_path, f"{i}_{design_name}_synthetic_mockup.png"), "PNG")
                mask.save(os.path.join(mask_image_path, f"{i}_{design_name}_synthetic_mask.png"), "PNG")
                
                # Augmented Data
                for aug_name, aug_fun in augmentations.items():
                    mockup, mask = aug_fun(mockup), aug_fun(mask)
                    mockup = mockup.resize((256, 256))
                    mask = mask.resize((256, 256))
                    mockup.save(os.path.join(mockup_image_path, f"{i}_{design_name}_{aug_name}_synthetic_mockup.png"), "PNG")
                    mask.save(os.path.join(mask_image_path, f"{i}_{design_name}_{aug_name}_synthetic_mask.png"), "PNG")

        except Exception as e:
            print(f"failed {i}: {e}")


def procure_dataset_from_stable_diffusion():
    import os

    mockup_data_path = os.path.join("data", "images", "T-Shirt Mockups - Sheet1.csv")
    df = pd.read_csv(mockup_data_path)

    mask_image_path = os.path.join("data", "images", "tshirt_segmentation", "masks")
    mockup_image_path = os.path.join("data", "images", "tshirt_segmentation", "mockups")

    augmentations = {"mirror": ImageOps.mirror}

    for i in range(1000):
        try:
            mockup, mask = generate_stable_diffusion_print_design_mockup(df.iloc[random.randrange(0, len(df))])
            mockup = mockup.resize((256, 256))
            mask = mask.resize((256, 256))
            mockup.save(os.path.join(mockup_image_path, f"{i}_synthetic_mockup.png"), "PNG")
            mask.save(os.path.join(mask_image_path, f"{i}_synthetic_mockup.png"), "PNG")

            # Augmented Data
            for aug_name, aug_fun in augmentations.items():
                mockup, mask = aug_fun(mockup), aug_fun(mask)
                mockup = mockup.resize((256, 256))
                mask = mask.resize((256, 256))
                mockup.save(os.path.join(mockup_image_path, f"{i}_{aug_name}_synthetic_mockup.png"), "PNG")
                mask.save(os.path.join(mask_image_path, f"{i}_{aug_name}_synthetic_mask.png"), "PNG")
        except Exception as e:
            print(f"failed {i}: {e}")

if __name__ == "__main__":
    procure_dataset_from_stable_diffusion()
