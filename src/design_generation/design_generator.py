from typing import Type

from numpy import random

from src.design_generation import design_builder
from src.data_collection import fonts
from src.data_collection import palettes
from src.data_collection import slogans
from src.design_generation import internal_repr as ir
from src.design_generation import template
from src.design_generation import text_layout
from src.design_generation import text_splitter


slogan_data = slogans.get_slogan_data()
font_data = fonts.get_font_data()
palette_data = palettes.get_palette_data()


def get_random_kwargs(layout: Type[text_layout.TextLayout]):
    kwargs = {}

    for param in layout.render_required_kwargs:
        match param:
            case "font_path":
                value = font_data["path"].sample(n=1).values[0]
            case "font_size":
                value = random.randint(16, 256)
            case "text_color":
                value = palette_data["colors"].sample(n=1).values[0][0]
            case "align":
                value = random.choice(text_layout.Align, 1)[0]
            case "line_spacing":
                value = random.randint(30, 200)
            case _:
                value = None
        kwargs[param] = value
    
    return kwargs


def random_design(seed=None):
    if seed:
        random.seed(seed)

    # Random slogan
    text: str = slogan_data["text"].sample(n=1).values[0]

    # Random layout
    layout = random.choice(text_layout.get_text_layouts(), 1)[0]

    builder = design_builder.TextComponentBuilder()
    builder.set_layout(layout)

    match layout:
        case text_layout.Identity:
            pass
        case text_layout.Multiline:
            word_count = len(text.split(" "))
            indices = [i for i in range(1, word_count)]
            random.shuffle(indices)
            indices = [0] + indices[:random.randint(len(indices))] + [word_count]
            splitter = text_splitter.WordIndexSplit(indices)

            text = [
                text_layout.Identity.render(t, **get_random_kwargs(text_layout.Identity))
                for t in splitter.split(text)
            ]
        case _:
            pass

    builder.set_text_components(text)
    builder.layout_params = get_random_kwargs(layout)

    return builder.build()


if __name__ == "__main__":
    for k in range(1):
        img, _ = random_design()
        img.show()
