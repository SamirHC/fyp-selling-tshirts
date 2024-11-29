import os
import requests
from dotenv import dotenv_values
import requests.auth
import pickle

config = dotenv_values(".env")

CLIENT_ID = config["CLIENT_ID"]
CLIENT_SECRET = config["CLIENT_SECRET"]


def get_access_token():
    url = "https://api.ebay.com/identity/v1/oauth2/token"
    headers = {
        "Content-Type": "application/x-www-form-urlenconded",
        "Authorization": f"Basic {CLIENT_ID}:{CLIENT_SECRET}",
    }
    body = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope"
    }

    response = requests.post(
        url=url,
        headers=headers,
        data=body,
        auth=requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    )
    response.raise_for_status()
    return response.json()["access_token"]


def search_items(access_token, query):
    url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    params = {
        "q": query,
        "limit": 5
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def get_item_details(access_token, item_id):
    url = f"https://api.ebay.com/buy/browse/v1/item/{item_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    response = requests.get(url=url, headers=headers)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    access_token = get_access_token()

    item = get_item_details(access_token, 176168653540)
    print(item)
    """
    items = search_items(access_token, "T-Shirt")
    with open("ebay_browse.pickle", "wb") as target:
        pickle.dump(items, target)
    """

    # items = search_items(access_token, "T-shirt")
    """
    with open(os.path.join("data","ebay_browse.pickle"), "rb") as target:
        items = pickle.load(target)

    for item in items.get("itemSummaries", []):
        item_id = item["itemId"]
        item_info = get_item_details(access_token, item_id)
        
        with open(os.path.join("data", f"item_info_{item_id}.pickle"), "wb") as target:
            pickle.dump(item_info, target)

        with open(f"item_info_{item_id}.pickle", "rb") as target:
            item_info = pickle.load(target)

        for k, v in item_info.items():
            print(f"{k}: {v} \n")
        
        print("="*80)
    """
