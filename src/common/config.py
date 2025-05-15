from dotenv import dotenv_values


config = dotenv_values(".env")

# Ebay
CLIENT_ID = config["CLIENT_ID"]
CLIENT_SECRET = config["CLIENT_SECRET"]

# Printify
PRINTIFY_API_TOKEN = config["PRINTIFY_API_TOKEN"]
PRINTIFY_SHOP_ID = config["PRINTIFY_SHOP_ID"]

# Hugging Face
HUGGING_FACE_TOKEN = config["HUGGING_FACE_TOKEN"]

# OpenRouter
OPENROUTER_API_KEY = config["OPENROUTER_API_KEY"]

# OpenAI
OPENAI_API_KEY = config["OPENAI_API_KEY"]

# Chromedriver
CHROMEDRIVER_PATH = config["CHROMEDRIVER_PATH"]

# Device
GPU = int(config["GPU"])

# Runtime Settings
PAYMENT_ACTIVE = bool(int(config["PAYMENT_ACTIVE"]))
