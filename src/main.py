import sqlite3

from datetime import datetime

from src.common import config, image_edit
from src.ml.genai import image_gen, text_gen
from src.design_generation import generate


if __name__ == "__main__":
    import os

    DB_PATH = os.path.join("data","db","dev_database.db")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    source, item_id =cursor.execute("""
        SELECT source, item_id FROM clothes
        WHERE EXISTS (
            SELECT source, design_id FROM print_design_tags AS pdt
            WHERE pdt.source=clothes.source AND pdt.design_id=clothes.item_id
        )
        ORDER BY RANDOM() LIMIT 1
    """).fetchone()
    tags = [
        x[0] for x in cursor.execute("""
            SELECT tag FROM print_design_tags
            WHERE source=? AND design_id=?
        """, (source, item_id)).fetchall()
    ]

    conn.close()
    print(tags)

    image_model = image_gen.DummyImageModel()
    if config.GPU == 0 and config.PAYMENT_ACTIVE:
        image_model = image_gen.OpenAIDallE3ImageModel()
    elif config.GPU == 1:
        image_model = image_gen.StableDiffusion1_5_Txt2ImgModel()

    design = generate.generate_design(tags, **{
        "text_model": text_gen.DeepSeekLLM(),
        "image_model": image_model
    })

    temp_path = os.path.join("out", f"main {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.svg")
    with open(temp_path, "w") as f:
        f.write(design.to_svg())
    design_image = image_edit.svg_to_png(temp_path)
    design_image.show()
