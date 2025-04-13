import os

import pandas as pd
import numpy as np
import torch
from torch import optim
from torch.utils.data import IterableDataset, DataLoader
from torch import nn

from src.ml.tshirt_design_segmentation import procure_data
from src.ml.genai.image_gen import StableDiffusion1_5_Txt2ImgModel as SD15
from src.common import image_edit
from src.data_collection import fonts
from src.design_generation import render_text


class LazyGenSyntheticDataset(IterableDataset):
    def __init__(self, tshirt_image_data: pd.DataFrame, nouns: list[str]):
        self.tshirt_image_data = tshirt_image_data
        self.nouns = nouns
        self.fonts = fonts.get_font_data()
        self.image_model = SD15()

    def get_random_noun(self) -> str:
        return str(np.random.choice(self.nouns, 1)[0])

    def generate_image_design(self):
        design = self.image_model.generate_image(
            prompt=self.get_random_noun(),
            negative_prompt="blurry, messy, over-detailed, extra elements, complex, photorealistic",
            height=512,
            width=512,
            num_inference_steps=50,
            num_images_per_prompt=1
        )
        design = image_edit.remove_bg(design)
        return design

    def generate_text_design(self):
        font = self.fonts.sample(n=1).iloc[0]
        random_color = tuple(np.random.randint(0, 256, 3))
        design = render_text.render_text(self.get_random_noun(), font["path"], text_color=random_color)
        return design

    def generate_design(self):
        if np.random.random() < 0.5:
            return self.generate_image_design()
        else:
            return self.generate_text_design()

    def __iter__(self):
        while True:
            try:
                design = self.generate_design()  # Random print design like image
                image, mask = procure_data.generate_central_mockup(self.tshirt_image_data.sample(n=1).iloc[0], design)
            except Exception as e:
                print(f"Failed to produce image and mask: {e}")
                continue
            image = np.array(image)
            mask = np.array(mask)
            yield image, mask


class TshirtPrintImageSegmentationModel(nn.Module):
    def __init__(self):
        super(TshirtPrintImageSegmentationModel, self).__init__()

        input_channel = 3
        output_channel = 1

        # Encoder path
        n = 16 # 16
        self.conv1 = nn.Sequential(
        nn.Conv2d(input_channel, n, kernel_size=3, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU(),
        nn.Conv2d(n, n, kernel_size=3, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU()
        )
        n *= 2 # 32
        self.conv2 = nn.Sequential(
        nn.Conv2d(int(n / 2), n, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU(),
        nn.Conv2d(n, n, kernel_size=3, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU()
        )
        n *= 2 # 64
        self.conv3 = nn.Sequential(
        nn.Conv2d(int(n / 2), n, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU(),
        nn.Conv2d(n, n, kernel_size=3, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU()
        )
        n *= 2 # 128
        self.conv4 = nn.Sequential(
        nn.Conv2d(int(n / 2), n, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU(),
        nn.Conv2d(n, n, kernel_size=3, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU()
        )
        # Decoder path
        n //= 2 # 64
        self.up_conv4 = nn.ConvTranspose2d(n*2, n, kernel_size=2, stride=2)
        self.decode_conv3 = nn.Sequential(
        nn.Conv2d(n*2, n, kernel_size=3, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU(),
        nn.Conv2d(n, n, kernel_size=3, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU(),
        )
        n //= 2 # 32
        self.up_conv3 = nn.ConvTranspose2d(n*2, n, kernel_size=2, stride=2)
        self.decode_conv2 = nn.Sequential(
        nn.Conv2d(n*2, n, kernel_size=3, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU(),
        nn.Conv2d(n, n, kernel_size=3, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU(),
        )
        n //= 2 # 16
        self.up_conv2 = nn.ConvTranspose2d(n*2, n, kernel_size=2, stride=2)
        self.decode_conv1 = nn.Sequential(
        nn.Conv2d(n*2, n, kernel_size=3, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU(),
        nn.Conv2d(n, n, kernel_size=3, padding=1),
        nn.BatchNorm2d(n),
        nn.ReLU(),
        )
        self.out_conv = nn.Conv2d(n, output_channel, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        conv1_skip = x
        x = self.conv2(x)
        conv2_skip = x
        x = self.conv3(x)
        conv3_skip = x
        x = self.conv4(x)
        # Decoder
        x = self.up_conv4(x)
        x = torch.cat([x, conv3_skip], dim=1)
        x = self.decode_conv3(x)
        x = self.up_conv3(x)
        x = torch.cat([x, conv2_skip], dim=1)
        x = self.decode_conv2(x)
        x = self.up_conv2(x)
        x = torch.cat([x, conv1_skip], dim=1)
        x = self.decode_conv1(x)
        x = self.out_conv(x)
        return torch.sigmoid(x)

    def save(self):
        save_path = os.path.join("data", "models", "TshirtPrintImageSegmentationModel1.pt")
        torch.save(self.state_dict(), save_path)


def train_and_save_model(train_data, test_data):
    # Model
    model = TshirtPrintImageSegmentationModel()
    
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = list(model.parameters())
    optimizer = optim.Adam(params, 1e-4)
    criterion = nn.BCELoss()

    its = 10000
    train_batch_size = 16
    eval_batch_size = 16

    for i in range(1, 1+its):
        model.train()

        images, masks = train_data.get_random_batch(train_batch_size)
        images, masks = torch.from_numpy(images), torch.from_numpy(masks)
        images, masks = images.to(device, dtype=torch.float32), masks.to(device, dtype=torch.float32)
        logits = model(images)

        optimizer.zero_grad()
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            model.eval()
            
            with torch.no_grad():
                test_images, test_masks = test_data.get_random_batch(eval_batch_size)
                test_images, test_masks = torch.from_numpy(test_images), torch.from_numpy(test_masks)
                test_images, test_masks = test_images.to(device, dtype=torch.float32), test_masks.to(device, dtype=torch.float32)
                test_logits = model(test_images)
                test_loss = criterion(test_logits, test_masks)
                print(f"{i} Eval: {test_loss}")

    model.save()
    return model


if __name__ == "__main__":
    mockup_data_path = os.path.join("data", "images", "T-Shirt Mockups - Sheet1.csv")
    tshirt_image_df = pd.read_csv(mockup_data_path)

    nouns = [
        "mirror",
        "camera",
        "cassette",
        "vinyl",
        "keyboard",
        "skateboard",
        "typewriter",
        "headphones",
        "lighter",
        "backpack",
        "sunglasses",
        "lantern",
        "candle",
        "compass",
        "bottle",
        "cup",
        "umbrella",
        "radio",
        "watch",
        "clock",
        "notebook",
        "pencil",
        "pen",
        "book",
        "guitar",
        "record",
        "shoe",
        "hat",
        "mask",
        "glasses",
        "ring",
        "chain",
        "wallet",
        "coin",
        "phone",
        "drone",
        "speaker",
        "magnet",
        "blade",
        "brick",
        "chair",
        "bed",
        "mirrorball",
        "lamp",
        "tape",
        "lens",
        "flag",
        "ticket",
        "box",
        "key"
    ]

    dataset = LazyGenSyntheticDataset(tshirt_image_df, nouns)
    loader = DataLoader(dataset, batch_size=4)

    for img, mask in loader:
        print("Image shape:", img.shape)  # (B, 3, H, W)
        print("Mask shape:", mask.shape)  # (B, H, W)
        break
