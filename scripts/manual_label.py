import os

from src.common import utils
from src.ml import preprocessor


save_path = os.path.join("data", "dataframes", "seller_hub_data", "labelled_ebay_data.pickle")

df = utils.load_data(save_path)
if "is_printed" not in df.columns:
    df["is_printed"] = 1

for index, row in df.iterrows():
    preprocessor.get_image(row["img_url"]).show()

    while True:
        value = input(f"Item {index} is_printed (0/1): ")

        match value:
            case "":
                value = df.loc[index, "is_printed"]
            case "0" | "1":
                value = int(value)
            case _:
                continue

        print(bool(value))
        df.loc[index, "is_printed"] = value
        break

utils.save_data(df, save_path=save_path)

labelled_df = utils.load_data(save_path)
print(labelled_df)
