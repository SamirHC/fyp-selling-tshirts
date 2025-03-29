import os
import random

import pandas as pd
from lxml import etree

from src.common import utils, constants
from src.design_generation import image_edit
from src.design_generation import internal_repr as ir
from src.ml.color_theme_classifier import ColorThemeClassifier
from src.ml.tshirt_design_segmentation import segmentation
from src.ml.genai import image_gen, text_gen
from src.design_generation.template import TopBottomTextWithCenterImage


def extract_design_data(df: pd.DataFrame, **kwargs):
    segmentation_model: segmentation.TshirtDesignSegmentationModel = (
        kwargs.get("tshirt_design_segmentation_model", segmentation.ProceduralSegmentation())
    )

    def _analyse_image(url):
        image = utils.get_image_from_url(url).convert("RGB")
        design_image = segmentation_model.extract_design(image)

        insights = {}

        palette_data = ColorThemeClassifier.get_palette_data(design_image)
        insights["color_tags"] = palette_data["color_tags"]
        insights["other_tags"] = palette_data["other_tags"]
        insights["colors"] = palette_data["colors"]

        # TODO: 
        #  - Resnet Image Classification or other: get key nouns in image and
        #    and collect for semantic data
        #  - Classify design as text, image, both or neither

        return pd.Series(insights)

    df[["color_tags", "other_tags", "colors"]] = df["img_url"].apply(_analyse_image)


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


if __name__ == "__main__":
    image_df_path = os.path.join("data", "dataframes", "seller_hub_data", "ebay_data.pickle")
    tshirt_df = utils.load_data(image_df_path)

    sample_df = tshirt_df.sample(n=1)

    extract_design_data(sample_df, **{
        "tshirt_design_segmentation_model": segmentation.SegformerB3ClothesSegmentation()
    })

    row = sample_df.iloc[0][["color_tags", "other_tags"]]
    tags = ["pinterest", "tshirt print design"] + row["color_tags"] + row["other_tags"]
    design = generate_design(tags, **{
        "text_model": text_gen.DeepSeekLLM(),
        "image_model": image_gen.StableDiffusion1_5_Txt2ImgModel()
    })
    xml = etree.tostring(design.to_svg(), pretty_print=True)

    temp_path = os.path.join("out", "temp.svg")
    with open(temp_path, "w") as f:
        f.write(xml.decode())
    design_image = image_edit.svg_to_png(temp_path)
    design_image.show()
