import numpy as np
from PIL import ImageDraw
import torch
from torchvision.ops import box_iou

from src.ml.tshirt_design_segmentation.segmentation import (
    NoSegmentation,
    ContourSegmentation,
    ProceduralSegmentation,
    EntropySegmentation,
    SegformerB3ClothesSegmentation,
    TshirtDesignSegmentationModel
)
from src.common import utils


def rect_intersect(rect1: tuple, rect2: tuple) -> tuple:
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1 + w1, x2 + w2) - xi
    hi = min(y1 + h1, y2 + h2) - yi

    if wi < 0 or hi < 0:
        wi = hi = 0

    return xi, yi, wi, hi

def rect_area(rect: tuple) -> float:
    """
    rect is a tuple of (x, y, width, height)
    """
    return rect[2] * rect[3]

def iou(rect1: tuple, rect2: tuple) -> float:
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    box1 = torch.tensor([[x1, y1, x1+w1, y1+h1]], dtype=torch.float)
    box2 = torch.tensor([[x2, y2, x2+w2, y2+h2]], dtype=torch.float)

    return box_iou(box1, box2).item()

def modified_iou(rect1, rect2_inner, rect2_outer) -> float:
    """
    returns |A intersect B|/|(A-(A intersect C)) union B| for B <= C
    """
    print("Inside miou", rect1, rect2_inner, rect2_outer)
    assert rect_intersect(rect2_inner, rect2_outer) == rect2_inner
    return rect_area(rect_intersect(rect1, rect2_inner)) / (rect_area(rect1) - rect_area(rect_intersect(rect1, rect2_outer)) + rect_area(rect2_inner))


def label():
    import os 

    image_df_path = os.path.join("data", "dataframes", "labelled_image_bboxs", "labelled_image_bboxs.pickle")
    tshirt_df = utils.load_data(image_df_path)
    print(tshirt_df)

    with open("data/dataframes/labelled_image_bboxs/tight_bboxs.csv", "r") as f:
        tight_bboxs = [eval(row) for row in f.read().splitlines()]
    with open("data/dataframes/labelled_image_bboxs/relaxed_bboxs.csv", "r") as f:
        relaxed_bboxs = [eval(row) for row in f.read().splitlines()]

    print(tight_bboxs)
    print(relaxed_bboxs)
    for i, row in tshirt_df.iterrows():
        try:
            x,y,w,h = tshirt_df.loc[i, ["x", "y", "w", "h"]] = tight_bboxs[i]
            xr,yr,wr,hr = tshirt_df.loc[i, ["xr", "yr", "wr", "hr"]] = relaxed_bboxs[i]

            url = row["img_url"]
            image = utils.get_image_from_url(url).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle((x,y,x+w,y+h), outline="red", width=3)
            draw.rectangle((xr,yr,xr+wr,yr+hr), outline="green", width=3)
            image.show()

        except Exception as e:
            print(f"error on {i}th: {e}")

    utils.save_data(tshirt_df, image_df_path)


def conduct_evaluation(image_df_path):
    tshirt_df = utils.load_data(image_df_path)

    seg_models: list[TshirtDesignSegmentationModel] = [
        NoSegmentation(),
        ContourSegmentation(),
        ProceduralSegmentation(),
        EntropySegmentation(),
        SegformerB3ClothesSegmentation()
    ]
    iou_scores = [[] for _ in seg_models]
    miou_scores = [[] for _ in seg_models]

    for i, row in tshirt_df.iterrows():
        print(i)
        url = row["img_url"]
        image = utils.get_image_from_url(url).convert("RGBA")

        for j, model in enumerate(seg_models):
            actual_bbox = tuple(row[["x","y","w","h"]])
            if actual_bbox == (0,0,0,0):
                continue
            relaxed_bbox = tuple(row[["xr","yr","wr","hr"]])
            extracted_bbox = model.extract_design_bbox(image)
            
            iou_score = iou(extracted_bbox, actual_bbox)
            iou_scores[j].append(iou_score)
            
            miou_score = modified_iou(extracted_bbox, actual_bbox, relaxed_bbox)
            miou_scores[j].append(miou_score)

    baseline_scores = np.array(miou_scores[0])
    print("Baseline:")
    print(baseline_scores)
    print(f"Mean MIoU score: {np.mean(baseline_scores)}")

    contour_scores = np.array(miou_scores[1])
    print("Contour:")
    print(contour_scores)
    print(f"Mean MIoU score: {np.mean(contour_scores)}")

    procedural_scores = np.array(miou_scores[2])
    print("Procedural:")
    print(procedural_scores)
    print(f"Mean MIoU score: {np.mean(procedural_scores)}")

    entropy_scores = np.array(miou_scores[3])
    print("Entropy:")
    print(entropy_scores)
    print(f"Mean MIoU score: {np.mean(entropy_scores)}")

    segformerB3_scores = np.array(miou_scores[4])
    print("SegformerB3:")
    print(segformerB3_scores)
    print(f"Mean MIoU score: {np.mean(segformerB3_scores)}")


if __name__ == "__main__":
    import os

    image_df_path = os.path.join("data", "dataframes", "labelled_image_bboxs", "labelled_image_bboxs.pickle")
    tshirt_df = utils.load_data(image_df_path)
    #print(tshirt_df)
    #for i, row in tshirt_df.iterrows():
    #    print(i+1, row["img_url"])

    #label()
    conduct_evaluation(image_df_path)
