import requests
from requests.auth import HTTPBasicAuth

import pandas as pd

from src.common import config


def get_access_token():
    url = "https://api.ebay.com/identity/v1/oauth2/token"
    headers = {
        "Content-Type": "application/x-www-form-urlenconded",
        "Authorization": f"Basic {config.CLIENT_ID}:{config.CLIENT_SECRET}",
    }
    body = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope"
    }

    response = requests.post(
        url=url,
        headers=headers,
        data=body,
        auth=HTTPBasicAuth(config.CLIENT_ID, config.CLIENT_SECRET)
    )
    response.raise_for_status()
    return response.json()["access_token"]


def search_items(access_token, query, limit=5):
    url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    params = {
        "q": query,
        "limit": limit
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def get_item_details_by_id(access_token, item_id):
    url = f"https://api.ebay.com/buy/browse/v1/item/{item_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    response = requests.get(url=url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_items_as_dataframe(access_token, query, limit=10) -> pd.DataFrame:
    try:
        result = search_items(access_token, query, limit)
    except Exception as e:
        print(f"Error in get_items_as_dataframe: {e}. Returning empty df")
        return pd.DataFrame(columns=["item_id", "title", "img_url"])

    df = pd.DataFrame(result["itemSummaries"])
    df = df[["itemId", "title", "thumbnailImages"]]
    df = df.rename(columns={"itemId": "item_id", "thumbnailImages": "img_url"})
    df["img_url"] = df["img_url"].apply(lambda x: x[0]["imageUrl"])

    return df


if __name__ == "__main__":
    access_token = get_access_token()
    df = get_items_as_dataframe(access_token, "inspirational graphic tee", limit=3)
    print(df)
    print(df.iloc[0]["img_url"])

    """
    result = search_items(access_token, "graphic tee", limit=10)
    print(result.keys())

    item_summaries = result["itemSummaries"]
    print(item_summaries[0].keys())

    for item in item_summaries:
        print(item["itemId"])
        print(item["legacyItemId"])
        print(item["listingMarketplaceId"])
        print(item["title"])
        print(item["thumbnailImages"][0]["imageUrl"])
        
        item_details = get_item_details_by_id(access_token, item["itemId"])
        print(item_details["price"])
        print(item_details["categoryPath"])
        print(item_details["image"])
        print(item_details["color"])
        print(item_details["pattern"])
        print(item_details["itemWebUrl"])

        print()
        """