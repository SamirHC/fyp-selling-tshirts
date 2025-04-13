import cairosvg
import io
from PIL import Image
import rembg


def remove_bg(image: Image.Image) -> Image.Image:
    return rembg.remove(image)


def crop_to_content(image: Image.Image) -> Image.Image:
    """
    Crops an image with transparency to the non-transparent content area.

    :param image: Input image with transparency

    :return out: Image cropped to content
    """

    bbox = image.getbbox()
    if bbox:
        return image.crop(bbox)
    else:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))


def svg_to_png(svg_path) -> Image.Image:
    return Image.open(io.BytesIO(cairosvg.svg2png(url=svg_path, dpi=300)))


if __name__ == "__main__":
    from src.ml.genai.image_gen import StableDiffusion1_5_Txt2ImgModel

    image_model = StableDiffusion1_5_Txt2ImgModel()
    image = image_model.generate_image(
        prompt="Minimalist T-shirt design of planet Earth, modern aesthetic",
        negative_prompt="blurry, messy, over-detailed, extra elements, complex, photorealistic",
        height=768,
        width=768,
        num_inference_steps=50,
        num_images_per_prompt=1
    )
    image.show()
    image = remove_bg(image)
    image = crop_to_content(image)
    image.show()
