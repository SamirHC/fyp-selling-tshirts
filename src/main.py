import sqlite3

from datetime import datetime

from src.common import config
from src.ml.genai import image_gen, text_gen
from src.design_generation import generate


def get_tags_title_colours(cursor: sqlite3.Cursor) -> list[str]:
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
            WHERE source=? AND design_id=?
        """, (source, item_id)).fetchall()
    ]
    colours = [x[0] for x in cursor.execute("""
        SELECT colour FROM print_design_nearest_palette
        NATURAL JOIN palette_colours
        WHERE source=? AND design_id=?
    """, (source, item_id)).fetchall()]

    return tags, title, colours


if __name__ == "__main__":
    import os
    import webbrowser

    image_model = image_gen.DummyImageModel()
    if config.GPU == 0 and config.PAYMENT_ACTIVE:
        image_model = image_gen.OpenAIDallE3ImageModel()
    elif config.GPU == 1:
        image_model = image_gen.StableDiffusion1_5_Txt2ImgModel()

    text_model = text_gen.DeepSeekLLM()
    MAX_TRIES = 5

    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    for i in range(MAX_TRIES):
        tags, title, colours = get_tags_title_colours(cursor)
        nouns = generate.get_nouns(title, text_model)
        if not nouns and i < MAX_TRIES-1:
            continue

        design = generate.generate_design(tags, **{
            "text_model": text_model,
            "image_model": image_model,
            "title": title,
            "colours": colours,
        })

    conn.close()

    out_path = os.path.join("out", f"main {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.svg")
    with open(out_path, "w") as f:
        f.write(design.to_svg())
    webbrowser.get("firefox").open(out_path)
