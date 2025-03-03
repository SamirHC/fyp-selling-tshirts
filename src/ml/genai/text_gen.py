from abc import ABC

from openai import OpenAI
from dotenv import dotenv_values

from src.data_collection import slogans


config = dotenv_values(".env")
OPENROUTER_API_KEY = config["OPENROUTER_API_KEY"]


# TODO: Flesh out LLMs for text design generation and prompt generation.

class TextModel(ABC):
    def generate_text(self) -> str:
        return ""


class RandomSloganModel(TextModel):
    def __init__(self):
        self.slogan_df = slogans.get_slogan_data()

    def generate_text(self) -> str:
        return self.slogan_df.sample(n=1).iloc[0]["text"]


class DummyLLM(TextModel):
    def generate_text(self, prompt=""):
        return prompt


class DeepSeekLLM(TextModel):
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

    def generate_text(self, prompt=""):
        completion = self.client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ]
        )

        return completion.choices[0].message.content


if __name__ == "__main__":
    slogan = RandomSloganModel().generate_text()

    print(slogan)
