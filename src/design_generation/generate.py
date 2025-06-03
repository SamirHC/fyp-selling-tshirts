import random

from src.common import constants, image_edit
from src.data_collection import palettes
from src.design_generation import internal_repr as ir
from src.design_generation.template import TopBottomTextWithCenterImage
from src.design_generation import font_select
from src.ml.genai import image_gen, text_gen


styles = ["retro", "vintage", "line art", "grunge", "psychedelic", "gothic",
    "urban", "pop art", "anime", "cartoon", "nature inspired",
    "silhouette", "cyberpunk", "surreal", "tattoo", 
    "watercolour", "kawaii", "graffiti", "glitchcore", "hand drawn", 
    "optical illusion"]


def create_prompt_for_image_prompt(tags, title, nouns, text_model: text_gen.TextModel):
    prompt_gen_prompt = (
        "Return ONLY a Python string of a well-crafted prompt for a generative AI model "
        f"using the tags ({",".join(tags + [random.choice(styles)])})."
    )
    if title:
        prompt_gen_prompt += (
            f" where the nouns of the image are {nouns} and inspired from a"
            f"vector design titled {title}. Keep it simple. "
        )
    else:
        f"where the nouns of the image are {nouns} colour/style tags."
    prompt_gen_prompt += " DO NOT REQUEST IN THE PROMPT TO PRODUCE TEXT."

    return text_model.generate_text(prompt_gen_prompt)


def create_prompt_for_image(tags, title, nouns, text_model, colours):
    # Prompt Generating Prompt
    content_prompt = create_prompt_for_image_prompt(tags, title, nouns, text_model)

    # Image Generating Prompt
    prompt = f"Create a vector design, centered"
    if colours:
        prompt += f", using the colours ({",".join(colours)})"
    prompt += f": {content_prompt}"

    return prompt


def get_nouns(title, text_model: text_gen.TextModel) -> list[str]:
    response: str = text_model.generate_text(f"""
        Return a comma separated list of object nouns from the provided string
        that are interesting subjects for a graphic design: '{title}'.
    """)
    nouns = [x for x in response.strip().split(",") if 3 <= len(x) < 10]
    print(f"Nouns: {nouns}")
    random.shuffle(nouns)
    n = min(len(nouns), 3)
    return nouns[:n]


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
    if prompt is None:
        image = image_gen.DummyImageModel().generate_image()
    else:
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
    design = TopBottomTextWithCenterImage(
        canvas_size=(512, 512),
        image=image,
        font=font,
        font_size=72,
        color=(constants.Color.BLACK if not colours else palettes.hex_to_rgb(colours[0])),
        top_text="GOOD",
        bottom_text="VIBES",
    ).design

    return design
