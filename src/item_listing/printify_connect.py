from dotenv import dotenv_values
import requests


config = dotenv_values(".env")

PRINTIFY_API_TOKEN = config["PRINTIFY_API_TOKEN"]
PRINTIFY_SHOP_ID = config["PRINTIFY_SHOP_ID"]
BASE_URL = "https://api.printify.com/v1"

headers = {
    "Authorization": f"Bearer {PRINTIFY_API_TOKEN}",
    "Content-Type": "application/json",
}


def get_blueprints():
    response = requests.get(
        f"{BASE_URL}/catalog/blueprints.json",
        headers=headers,
    )
    response.raise_for_status()

    content = response.json()
    for i in range(5):
        print(content[i])


def get_print_providers_by_blueprint_id(blueprint_id):
    response = requests.get(
        f"{BASE_URL}/catalog/blueprints/{blueprint_id}/print_providers.json",
        headers=headers
    )

    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to retrieve print providers:", response.status_code, response.text)


def get_variants_of_blueprint_from_print_provider(blueprint_id, print_provider_id):
    response = requests.get(
        f"{BASE_URL}/catalog/blueprints/{blueprint_id}/print_providers/{print_provider_id}/variants.json",
        headers=headers
    )

    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to retrieve variants:", response.status_code, response.text)



def get_shops():
    response = requests.get(f"{BASE_URL}/shops.json", headers=headers)
    
    if response.status_code == 200:
        shops = response.json()
        for shop in shops:
            print(f"Shop ID: {shop['id']}, Title: {shop['title']}")
    else:
        print("Failed to retrieve shops:", response.status_code, response.text)


def upload_image(file_name="placeholder.png", image_url="https://via.placeholder.com/500x500"):
    payload = {
        "file_name": file_name,
        "url": image_url,
    }
    response = requests.post(
        f"{BASE_URL}/uploads/images.json",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        print(f"Image {file_name} uploaded successfully! {image_url}")
        return response.json()
    else:
        print("Failed to upload image:", response.status_code, response.text)


def get_uploads():
    response = requests.get(
        f"{BASE_URL}/uploads.json",
        headers=headers
    )

    if response.status_code == 200:
        print("Uploads loaded successfully!")
        return response.json()
    else:
        print("Failed to get uploads:", response.status_code, response.text)


def create_product():
    blueprint_id = 5  # Unisex Cotton Crew Tee
    print_providers = get_print_providers_by_blueprint_id(blueprint_id)
    print_provider_id = print_providers[0]["id"]
    variants = get_variants_of_blueprint_from_print_provider(blueprint_id, print_provider_id)
    variant_id = variants["variants"][0]["id"]

    payload = {
        "title": "Placeholder Title",
        "description": f"Placeholder description.",
        "variants": [{
            "id": variant_id,
            "price": 1000,
            "is_enabled": True,
        }],
        "blueprint_id": blueprint_id,
        "print_provider_id": print_provider_id,
        "print_areas": [{
            "variant_ids": [variant_id],
            "placeholders": [{
                "position": "front",
                "images": [{
                    "id": "67749747324f716833dd653b",
                    "x": 0.5,
                    "y": 0.5,
                    "scale": 1,
                    "angle": 0,
                }]
            }]
        }],
    }

    response = requests.post(
        f"{BASE_URL}/shops/{PRINTIFY_SHOP_ID}/products.json",
        headers=headers,
        json=payload
    )
    if response.status_code == 200:
        print("Draft product created successfully!")
        print(response.json())
    else:
        print("Failed to create draft product:", response.status_code, response.text)


if __name__ == "__main__":
    create_product()
