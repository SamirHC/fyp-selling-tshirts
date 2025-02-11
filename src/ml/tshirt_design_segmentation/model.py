import os

import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset
from torch import nn


class TshirtPrintImageDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx] / 255
        label = self.masks[idx]
        return image, label

    def get_random_batch(self, batch_size):
        indices = np.arange(len(self))
        np.random.shuffle(indices)

        images, labels = [], []
        for idx in indices[:batch_size]:
            image, label = self[idx]
            images.append(np.transpose(image, (2, 0, 1))[:3, :, :])
            labels.append(label[np.newaxis, :, :])
        images = np.array(images)
        labels = np.array(labels)

        return images, labels


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
    import imageio

    # Load Dataset
    base_path = os.path.join("data", "images", "tshirt_segmentation")
    image_path = os.path.join(base_path, "mockups")
    label_path = os.path.join(base_path, "masks")

    images = []
    labels = []
    for image_name in os.listdir(image_path):
        image = imageio.imread(os.path.join(image_path, image_name))
        images.append(image)

        label_name = os.path.join(label_path, image_name.removesuffix("mockup.png")+"mask.png")
        label = imageio.imread(label_name)
        labels.append(label)
    
    
    train_data = TshirtPrintImageDataset(images[:-200], labels[:-200])
    test_data = TshirtPrintImageDataset(images[-200:], labels[-200:])
    # model = train_and_save_model(train_data, test_data)

    model = TshirtPrintImageSegmentationModel()
    model.load_state_dict(torch.load(os.path.join("data", "models", "TshirtPrintImageSegmentationModel1.pt")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    # Visualise
    images, labels = test_data.get_random_batch(4)
    logits = model(torch.from_numpy(images).to(device, dtype=torch.float32))
    binary_mask = (logits > 0.5).float().squeeze().cpu().numpy()
    images = images.transpose(0, 2, 3, 1)
    labels = labels.transpose(0, 2, 3, 1)
    fig, axs = plt.subplots(4, 3, figsize=(60, 120))

    for i in range(4):
        axs[i][0].imshow(images[i])
        axs[i][1].imshow(binary_mask[i], cmap=colors.ListedColormap(['black', 'white']))
        axs[i][2].imshow(labels[i], cmap=colors.ListedColormap(['black', 'white']))

    plt.show()
