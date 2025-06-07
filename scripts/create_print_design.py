import os
import webbrowser

from src.data_collection import fonts
from src.common import utils
from src.design_generation.internal_repr import *
from src.design_generation.template import *


def save_design(design: Design, save_path: str):
    svg = design.to_svg()
    with open(save_path, "w") as f:
        f.write(svg)
    print(f"SVG Saved to {save_path}")


def view_design(svg_path: str):
    webbrowser.get("firefox").open(svg_path)


def create_grid_collage():
    image_paths = [
        os.path.join("out", "components", filename) for filename in (
            "ChatGPT Image Jun 6, 2025, 09_19_26 PM.png",
            "ChatGPT Image Jun 6, 2025, 09_20_56 PM.png",
            "ChatGPT Image Jun 6, 2025, 09_21_48 PM.png",
            "ChatGPT Image Jun 6, 2025, 09_22_27 PM.png",
        )
    ]
    images = list(map(lambda im: Image.open(im), image_paths))
    return GridCollage((512, 512), images, 2, 2)


def create_captioned_image():
    image_path = os.path.join("out", "components", "ChatGPT Image Jun 6, 2025, 10_17_38 PM.png")
    image = Image.open(image_path)
    return CaptionedImage((512, 512), image, "Coolio", 92, (255,215,0), "COOL", "VIBES")


if __name__ == "__main__":
    # Grid Collage
    save_path = os.path.join("out", "grid_collage.svg")
    design = create_grid_collage()
    save_design(design, save_path)
    view_design(save_path)

    # Captioned Image
    save_path = os.path.join("out", "captioned_image.svg")
    design = create_captioned_image()
    save_design(design, save_path)
    view_design(save_path)
