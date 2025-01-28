import os

import pandas as pd


BASE_SLOGAN_PATH = os.path.join("data", "slogans")


def get_slogan_data():
    data = []

    CHATGPT_PATH = os.path.join(BASE_SLOGAN_PATH, "ChatGPT_generated_phrases.txt")
    CLAUDE_PATH = os.path.join(BASE_SLOGAN_PATH, "Claude_generated_phrases.txt")

    with open(CHATGPT_PATH, "r") as f:
        for line in f:
            data.append({"text": line.strip(), "source": "ChatGPT"})
    
    with open(CLAUDE_PATH, "r") as f:
        for line in f:
            data.append({"text": line.strip(), "source": "Claude"})

    return pd.DataFrame(data)


if __name__ == "__main__":
    print(get_slogan_data())
