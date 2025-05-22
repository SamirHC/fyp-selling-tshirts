import random


from src.common import constants
from src.design_generation import internal_repr as ir
from src.design_generation.template import TopBottomTextWithCenterImage
from src.ml.genai import image_gen, text_gen


def generate_design(tags: list[str], **kwargs) -> ir.Design:
    if len(tags) < 2:
        tags += ["minimalist", "vintage"]
    title: str = kwargs.get("title", None)
    colours: list[str] = kwargs.get("colours", None)

    text_model: text_gen.TextModel = (
        kwargs.get("text_model", text_gen.DummyLLM())
    )
    image_model: image_gen.ImageModel = (
        kwargs.get("image_model", image_gen.DummyImageModel())
    )

    # Prompt Generating Prompt
    prompt_gen_prompt = (
        "Return ONLY a Python string of a well-crafted prompt for a generative AI model "
        f"using the tags ({",".join(tags)}) only to emphasise the colour themes, not the content."
    )
    if title:
        prompt_gen_prompt += (
            " where the nouns of the image are inspired from the contents of a "
            f"Tshirt print design titled {title}. Do not use too many different nouns, keep it simple. "
        )
    else:
        "where the nouns of the image are determined appropriately based on the colour/style tags."
    prompt_gen_prompt += " DO NOT REQUEST IN THE PROMPT TO PRODUCE TEXT."

    content_prompt = text_model.generate_text(prompt_gen_prompt)

    # Image Generating Prompt
    prompt = f"Create a T-shirt print design graphic, centered, transparent background, no text"
    if colours:
        prompt += f", using the colours ({",".join(colours)})"
    prompt += f": {content_prompt}"

    print(f"Tags: {tags}")
    print(f"Colours: {colours}")
    print(f"Title: {title}")
    print(f"Prompt: {prompt}")

    # Image Generation
    image = image_model.generate_image(
        prompt=prompt
    ).resize((256, 256))

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
