import random


from src.common import constants
from src.design_generation import internal_repr as ir
from src.design_generation.template import TopBottomTextWithCenterImage
from src.ml.genai import image_gen, text_gen


def create_prompt_for_image_prompt(tags, title, text_model: text_gen.TextModel):
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

    return text_model.generate_text(prompt_gen_prompt)


def create_prompt_for_image(tags, title, text_model, colours):
    # Prompt Generating Prompt
    content_prompt = create_prompt_for_image_prompt(tags, title, text_model)

    # Image Generating Prompt
    prompt = f"Create a T-shirt print design graphic, centered, transparent background, no text"
    if colours:
        prompt += f", using the colours ({",".join(colours)})"
    prompt += f": {content_prompt}"
    prompt += " DO NOT INCLUDE TEXT IN THE IMAGE, DO NOT INCLUDE ANY CLOTHING, ONLY EXTRACT THE FULL PRINT DESIGN."

    return prompt


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

    # Prompt Information
    prompt = create_prompt_for_image(tags, title, text_model, colours)
    print(f"Tags: {tags}")
    print(f"Colours: {colours}")
    print(f"Title: {title}")
    print(f"Prompt: {prompt}")

    # Image Generation
    image = image_model.generate_image(
        prompt=prompt
    ).resize((256, 256))

    # Get a batch of prompts and images, and compute fitness
    # Fitness: minimises colour palette distance
    #          minimises mask occurence when using segformerb3 model
    #          scores highly 

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
