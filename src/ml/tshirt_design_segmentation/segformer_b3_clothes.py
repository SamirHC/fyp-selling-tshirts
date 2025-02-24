from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
from PIL import Image
import numpy as np

from src.common import utils


class SegformerB3Clothes:
    class Labels:
        Background = 0
        Hat = 1
        Hair = 2
        Sunglasses = 3
        Upperclothes = 4
        Skirt = 5
        Pants = 6
        Dress = 7
        Belt = 8
        Leftshoe = 9
        Rightshoe = 10
        Face = 11
        Leftleg = 12
        Rightleg = 13
        Leftarm = 14
        Rightarm = 15
        Bag = 16
        Scarf = 17

    processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")

    @staticmethod
    def segment_clothes(image: Image.Image):
        inputs = SegformerB3Clothes.processor(images=image, return_tensors="pt")

        outputs = SegformerB3Clothes.model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]
        return pred_seg
    
    @staticmethod
    def seg_to_pil_image(seg):
        return Image.fromarray(seg.numpy().astype(np.uint8))

    @staticmethod
    def segment_upper_clothes(image: Image.Image):
        seg = SegformerB3Clothes.seg_to_pil_image(SegformerB3Clothes.segment_clothes(image))
        return Image.fromarray(np.array(seg) == SegformerB3Clothes.Labels.Upperclothes)


if __name__ == "__main__":
    import os

    image_df_path = os.path.join("data", "dataframes", "seller_hub_data", "ebay_data.pickle")
    tshirt_df = utils.load_data(image_df_path)
        
    url = "https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/473491/item/goods_09_473491_3x4.jpg?width=600"

    for i in range(len(tshirt_df)):
        url = tshirt_df.iloc[i]["img_url"]
        image = utils.get_image_from_url(url).convert("RGB")

        pred_seg = Image.fromarray(np.array(image) * np.repeat(np.array(SegformerB3Clothes.segment_upper_clothes(image))[:,:,np.newaxis], 3, axis=2))
        pred_seg.show()
