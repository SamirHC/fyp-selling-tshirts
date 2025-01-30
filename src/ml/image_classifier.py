from transformers import AutoImageProcessor, ResNetForImageClassification
import torch


processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")


def classify_image(image):
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]


if __name__ == "__main__":
    from src.common import utils
    from src.data_collection.etsy_page_scrape import EtsyPageScraper
    import os

    files = os.listdir(EtsyPageScraper.BASE_SAVE_DIR)
    dataset_path = os.path.join(EtsyPageScraper.BASE_SAVE_DIR, files[0])
    df = utils.load_data(dataset_path)

    for i in range(len(df)):
        image_url = df["img_url"].iloc[i]
        image = utils.get_image_from_url(image_url)
        print(i, classify_image(image))
