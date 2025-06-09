from datetime import datetime
import os
import sqlite3
from PIL import Image

from src.common import constants, image_edit, config
from src.data_collection import palettes
from src.design_generation import internal_repr as ir
from src.design_generation.template import CaptionedImage
from src.design_generation import font_select
from src.ml.genai import image_gen, text_gen
from src.ml.color_analysis.color_theme_classifier import CIELabColorThemeClassifier


def add_to_population(clothes_key, image_no_bg: Image.Image, prompt):
    print("Adding to population...")
    # Save image file
    filename = f"generate_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.png"
    image_path = os.path.join("data", "ea_population", filename)
    image_no_bg.save(image_path)

    # Colour Analysis
    palette_data = CIELabColorThemeClassifier.get_palette_data(image_no_bg)
    palette_id = palette_data["row_idx"]
    palette_dist = palette_data["dist"]
    colour_score = 0 if palette_dist > 40 else (1 - palette_dist / 40)

    # Temp
    prompt_score = -1
    aesthetic_score = -1

    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    query = """
        INSERT INTO evaluate_generations (image_path, source, item_id, prompt, palette_id, palette_distance, colour_score, prompt_score, aesthetic_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (filename, *clothes_key, prompt, palette_id, palette_dist, colour_score, prompt_score, aesthetic_score)
    print(params)
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    print("Done.")


def create_prompt_for_image_prompt(tags, title, text_model: text_gen.TextModel):
    prompt_gen_prompt = (
        "Return ONLY a Python string of a well-crafted prompt for a generative AI model "
        f"using the tags ({",".join(tags)}) to modify nouns"
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
    prompt += " DO NOT INCLUDE TEXT IN THE IMAGE. KEEP IT SIMPLE"

    return prompt, content_prompt


def generate_design(tags: list[str], **kwargs) -> ir.Design:
    if len(tags) < 2:
        tags += ["minimalist", "vintage"]
    title: str = kwargs.get("title", None)
    colours: list[str] = kwargs.get("colours", None)
    clothes_key: tuple[str, str] = kwargs.get("clothes_key", None)

    text_model: text_gen.TextModel = (
        kwargs.get("text_model", text_gen.DummyLLM())
    )
    image_model: image_gen.ImageModel = (
        kwargs.get("image_model", image_gen.DummyImageModel())
    )

    # Prompt Information
    prompt, content_prompt = create_prompt_for_image(tags, title, text_model)#, colours)
    print(f"Tags: {tags}")
    print(f"Colours: {colours}")
    print(f"Title: {title}")
    print(f"Prompt: {prompt}")

    # Image Generation
    image = image_model.generate_image(
        prompt=prompt
    ).resize((256, 256))
    image_no_bg = image_edit.remove_bg(image)
    image_no_bg = image_edit.crop_to_content(image_no_bg)

    # Save generation to population
    add_to_population(clothes_key, image_no_bg, content_prompt)

    # Font Selection
    font = font_select.select_font(prompt, image)

    # Get a batch of prompts and images, and compute fitness
    # Fitness: minimises colour palette distance
    #          scores highly 
    # TODO:
    #  - Create slogan and split
    #  - Choose font based on tags
    #  - Choose color based on bg color for readability
    design = CaptionedImage(
        canvas_size=(512, 512),
        image=image_no_bg,
        font=font,
        font_size=72,
        color=(constants.Color.BLACK if not colours else palettes.hex_to_rgb(colours[0])),
        top_text="",
        bottom_text="",
    ).design

    return design


def get_tags_title_colours(cursor: sqlite3.Cursor, colour_tags=True) -> list[str]:
    source, item_id, title = cursor.execute("""
        SELECT source, item_id, title FROM clothes
        WHERE EXISTS (
            SELECT source, design_id FROM print_design_tags AS pdt
            WHERE pdt.source=clothes.source AND pdt.design_id=clothes.item_id
        )
        ORDER BY RANDOM() LIMIT 1
    """).fetchone()
    tags = [
        x[0] for x in cursor.execute("""
            SELECT tag FROM print_design_tags
            JOIN palette_tags
            ON print_design_tags.tag = palette_tags.name
            WHERE source=? AND design_id=? AND is_colour_tag=?
        """, (source, item_id, int(colour_tags))).fetchall()
    ]
    colours = [x[0] for x in cursor.execute("""
        SELECT colour FROM print_design_nearest_palette
        NATURAL JOIN palette_colours
        WHERE source=? AND design_id=?
    """, (source, item_id)).fetchall()]

    return tags, title, colours, source, item_id


def generate_random_design_from_db() -> ir.Design:
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    tags, title, colours, source, item_id = get_tags_title_colours(cursor)

    conn.close()

    image_model = image_gen.DummyImageModel()
    if config.GPU == 0 and config.PAYMENT_ACTIVE:
        image_model = image_gen.OpenAIDallE3ImageModel()
    elif config.GPU == 1:
        image_model = image_gen.StableDiffusion1_5_Txt2ImgModel()

    text_model = text_gen.GPT4LLM()  # text_gen.DeepSeekLLM()

    design = generate_design(tags, **{
        "text_model": text_model,
        "image_model": image_model,
        "title": title,
        "colours": colours,
        "clothes_key": (source, item_id)
    })
    return design
