from dotenv import dotenv_values
import requests
from requests.auth import HTTPBasicAuth

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
        auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
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
