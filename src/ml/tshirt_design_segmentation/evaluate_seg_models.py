import itertools
import os
import time
import sqlite3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw
import torch
from torchvision.ops import box_iou

from src.ml.tshirt_design_segmentation.segmentation import (
    NoSegmentation,
    FixedSegmentation,
    ContourSegmentation,
    EntropySegmentation,
    SegformerB3ClothesSegmentation,
    TshirtDesignSegmentationModel
)
from src.common import utils, config


def rect_intersect(rect1: tuple, rect2: tuple) -> tuple:
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    xi = int(max(x1, x2))
    yi = int(max(y1, y2))
    wi = int(min(x1 + w1, x2 + w2) - xi)
    hi = int(min(y1 + h1, y2 + h2) - yi)

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
    # print("Inside miou", rect1, rect2_inner, rect2_outer)
    in_intersect_out = rect_intersect(rect2_inner, rect2_outer)
    if in_intersect_out != rect2_inner:
        print("Found label violating B <= C! Shrinking B to B intersect C.")
    return (
        rect_area(rect_intersect(rect1, in_intersect_out)) 
        / (rect_area(rect1) - rect_area(rect_intersect(rect1, rect2_outer)) 
            + rect_area(in_intersect_out))
    )


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


def evaluate_model(model: TshirtDesignSegmentationModel, df: pd.DataFrame, show=False):
    print(f"Evaluating {model}...")
    iou_scores = []
    miou_scores = []
    time_scores = []

    for i, row in df.iterrows():
        print(i)
        url = row["image_url"]
        image = utils.get_image_from_url(url).convert("RGB")

        x,y,w,h = actual_bbox = tuple(row[["left_in","top_in","width_in","height_in"]])
        xr,yr,wr,hr = relaxed_bbox = tuple(row[["left_out","top_out","width_out","height_out"]])

        start_time = time.perf_counter()
        xe1,ye1,xe2,ye2 = extracted_bbox = model.extract_design_bbox(image)  # (x1,y1,x2,y2) form
        end_time = time.perf_counter()
        
        iou_score = iou((xe1, ye1, xe2-xe1, ye2-ye1), actual_bbox)
        iou_scores.append(iou_score)
        
        miou_score = modified_iou((xe1, ye1, xe2-xe1, ye2-ye1), actual_bbox, relaxed_bbox)
        miou_scores.append(miou_score)

        duration = end_time - start_time
        time_scores.append(duration)

        if show:
            draw = ImageDraw.Draw(image.copy())
            draw.rectangle((x,y,x+w,y+h), outline="red", width=3)
            draw.rectangle((xr,yr,xr+wr,yr+hr), outline="green", width=3)
            draw.rectangle(extracted_bbox, outline="blue", width=3)
            print(model.__class__)
            draw._image.show()
            model.extract_design(image).show()
            input("next: press enter")

    return {"iou": iou_scores, "miou": miou_scores, "times": time_scores}


def conduct_evaluation(seg_models, df: pd.DataFrame, show=False):
    for model in seg_models:
        result = evaluate_model(model, df)
        print_result(model, result)
        if show:
            plot_histogram(model, result)


def print_result(model, result):
    iou_data = result["iou"]
    miou_data = result["miou"]
    time_data = result["times"]
    print("-"*80)
    print(model.__class__.__name__)
    print(f"IoU scores:")
    print(iou_data)
    print()
    print(f"MIoU scores:")
    print(miou_data)
    print()
    print("Times:")
    print(time_data)
    print()
    print(f"Mean IoU score: {np.mean(iou_data)}")
    print(f"Mean MIoU score: {np.mean(miou_data)}")
    print(f"Mean execution time: {np.mean(time_data)}")
    print("-"*80)


def plot_histogram(model, result):
    miou_data = result["miou"]
    plt.hist(miou_data, bins=10, density=True)
    plt.xlabel("MIoU")
    plt.xlim(left=0, right=1)
    plt.ylabel("Frequency Density")
    plt.title(model.__class__.__name__)
    plt.show()


def get_test_df():
    print("Fetching Test DataFrame...")
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    query = """
        SELECT clothes.source, clothes.item_id, clothes.image_url,
            pdr_in.left AS left_in, pdr_in.top AS top_in,
            pdr_in.width AS width_in, pdr_in.height AS height_in,
            pdr_out.left AS left_out, pdr_out.top AS top_out,
            pdr_out.width AS width_out, pdr_out.height AS height_out
        FROM clothes
            JOIN print_design_regions AS pdr_in
            ON clothes.source = pdr_in.source AND clothes.item_id = pdr_in.item_id
            JOIN print_design_regions AS pdr_out
            ON clothes.source = pdr_out.source AND clothes.item_id = pdr_out.item_id
        WHERE pdr_in.algorithm = 'InnerGroundTruth' AND pdr_out.algorithm = 'OuterGroundTruth'
    """
    result = cursor.execute(query)
    rows = result.fetchall()
    columns = [d[0] for d in result.description]
    df = pd.DataFrame(rows, columns=columns)

    df["image"] = df["image_url"].apply(lambda x: utils.get_image_from_url(x).convert("RGB"))
    print("Complete.")
    return df


def get_validation_df():
    print("Fetching Validation DataFrame...")
    image_df_path = os.path.join("data", "dataframes", "labelled_image_bboxs", "labelled_image_bboxs.pickle")
    df = utils.load_data(image_df_path)
    df = df.rename(columns={
        "img_url": "image_url",
        "x": "left_in",
        "y": "top_in",
        "w": "width_in",
        "h": "height_in",
        "xr": "left_out",
        "yr": "top_out",
        "wr": "width_out",
        "hr": "height_out"
    }).dropna()
    df[["left_out", "top_out", "width_out", "height_out"]] = df[["left_out", "top_out", "width_out", "height_out"]].astype(int)
    df = df[(df[["width_in", "height_in", "width_out", "height_out"]] != 0).any(axis=1)].reset_index(drop=True)

    df["image"] = df["image_url"].apply(lambda x: utils.get_image_from_url(x).convert("RGB"))
    print("Complete.")
    return df


def contour_hyperparam_search(validation_df):
    base_model = ContourSegmentation
    size_values = [16, 32, 64, 128, 256]
    centrality_values = np.linspace(0.4, 1, 6)
    min_dim_values = np.linspace(0.1, 0.4, 4)

    results = dict()

    for params in itertools.product(size_values, centrality_values, min_dim_values):
        size, centrality, min_dim = params
        model = base_model(size, centrality, min_dim, min_dim)
        
        validation_results = evaluate_model(model, validation_df)
        miou_result = np.mean(validation_results["miou"])
        results[params] = miou_result
        print(f"Params: {params}")
        print(f"MIoU: {miou_result}")
    
    best_params = max(results, key=results.get)
    max_miou = results[best_params]
    print(f"Best validation test results:")
    print(f"Params: {best_params}")
    print(f"MIoU: {max_miou}")
    return best_params


if __name__ == "__main__":
    """
    validation_df = get_validation_df()
    #params = contour_hyperparam_search(validation_df)
    
    test_df = get_test_df()
    conduct_evaluation(
        seg_models=[ContourSegmentation(params[0], params[1], params[2], params[2])],
        df=test_df,
        show=True
    )
    """
    seg_models: list[TshirtDesignSegmentationModel] = [
        NoSegmentation(),
        FixedSegmentation(),
        ContourSegmentation(),
        EntropySegmentation(),
        SegformerB3ClothesSegmentation(),
    ]
    conduct_evaluation(seg_models, get_test_df(), show=True)
