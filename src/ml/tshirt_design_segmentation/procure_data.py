import pandas as pd
from PIL import Image, ImageOps

from src.common import utils


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


if __name__ == "__main__":
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
