from src.common import constants, image_edit
from src.data_collection import palettes
from src.design_generation import internal_repr as ir
from src.design_generation.template import CaptionedImage
from src.design_generation import font_select
from src.ml.genai import image_gen, text_gen


def create_prompt_for_image_prompt(tags, title, text_model: text_gen.TextModel):
    prompt_gen_prompt = (
        "Return ONLY a Python string of a well-crafted prompt for a generative AI model "
        f"using the tags ({",".join(tags)}) only to emphasise the colour themes, not the content."
    )
    if title:
        prompt_gen_prompt += (
            " where the contents of the image are inspired from the contents of a "
            f"Tshirt print design titled {title}. Keep it simple, only use up to 3 nouns. "
        )
    else:
        "where the nouns of the image are determined appropriately based on the colour/style tags."
    prompt_gen_prompt += " DO NOT REQUEST IN THE PROMPT TO PRODUCE TEXT. DO NOT INCLUDE WORDS RELATING TO TSHIRTS"

    return text_model.generate_text(prompt_gen_prompt)


def create_prompt_for_image(tags, title, text_model, colours=None):
    # Prompt Generating Prompt
    content_prompt = create_prompt_for_image_prompt(tags, title, text_model)

    # Image Generating Prompt
    prompt = f"Vector design centered, no background"
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
    prompt = create_prompt_for_image(tags, title, text_model)#, colours)
    print(f"Tags: {tags}")
    print(f"Colours: {colours}")
    print(f"Title: {title}")
    print(f"Prompt: {prompt}")

    # Image Generation
    image = image_model.generate_image(
        prompt=prompt
    ).resize((256, 256))
    image = image_edit.remove_bg(image)

    # Font Selection
    font = font_select.select_font(prompt, image)

    # Get a batch of prompts and images, and compute fitness
    # Fitness: minimises colour palette distance
    #          minimises mask occurence when using segformerb3 model
    #          scores highly 
    # TODO:
    #  - Create slogan and split
    #  - Choose font based on tags
    #  - Choose color based on bg color for readability
    design = CaptionedImage(
        canvas_size=(512, 512),
        image=image,
        font=font,
        font_size=72,
        color=(constants.Color.BLACK if not colours else palettes.hex_to_rgb(colours[0])),
        top_text="",
        bottom_text="",
    ).design

    return design
