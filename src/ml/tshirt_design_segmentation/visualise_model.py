import os

import numpy as np
import torch
from PIL import Image

from src.ml.tshirt_design_segmentation.model import TshirtPrintImageSegmentationModel


model = TshirtPrintImageSegmentationModel()
model.load_state_dict(torch.load(os.path.join("data", "models", "TshirtPrintImageSegmentationModel1.pt")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_binary_masks(images: list[Image.Image]):
    sizes = [image.size for image in images]
    images = [image.copy().resize((256, 256)) for image in images]
    images = np.array([np.transpose(image, (2, 0, 1))[:3, :, :] for image in images])

    logits = model(torch.from_numpy(images).to(device, dtype=torch.float32))
    binary_masks = (logits > 0.5).float().squeeze().cpu().numpy()

    return [Image.fromarray(binary_mask, mode="1").resize(size) for binary_mask, size in zip(binary_masks, sizes)]


if __name__ == "__main__":
    from src.common import utils

    etsy_path = os.path.join("data", "dataframes", "etsy_listing_data", "EtsyPageScraper 2025-01-08 08:53:59.pickle")
    df = utils.load_data(etsy_path)

    images = [utils.get_image_from_url(url) for url in df["img_url"]]
    images = [image for image in images if image is not None]

    for image in images:
        binary_masks = get_binary_masks([image])
        mask = binary_masks[0]
        image.putalpha(mask)
        image.show()
