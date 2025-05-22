import random


from src.common import constants
from src.design_generation import internal_repr as ir
from src.design_generation.template import TopBottomTextWithCenterImage
from src.ml.genai import image_gen, text_gen


def generate_design(tags: list[str], **kwargs) -> ir.Design:
    if len(tags) < 2:
        tags += ["minimalist", "vintage"]

    text_model: text_gen.TextModel = (
        kwargs.get("text_model", text_gen.DummyLLM())
    )
    image_model: image_gen.ImageModel = (
        kwargs.get("image_model", image_gen.DummyImageModel())
    )

    # TODO: Implement prompts and image generation
    prompt = text_model.generate_text(
        "Return in string format a short, well-crafted prompt for a generative AI model using the following tags: " +
        ",".join(tags)
    )
    print(f"Tags: {tags}")
    print(f"Prompt: {prompt}")
    image = image_model.generate_image(prompt=prompt).resize((256, 256))

    # TODO:
    #  - Create slogan and split
    #  - Choose font based on tags
    #  - Choose color based on bg color for readability
    design = TopBottomTextWithCenterImage(
        canvas_size=(512, 512),
        image=image,
        font="cookie",
        font_size=72,
        color=constants.Color.BLACK,
        top_text=random.choice(tags),
        bottom_text=random.choice(tags),
    ).design

    return design
