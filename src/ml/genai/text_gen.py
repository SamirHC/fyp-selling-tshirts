from abc import ABC

from openai import OpenAI

from src.data_collection import slogans
from src.common import config


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
            api_key=config.OPENROUTER_API_KEY,
        )

    def generate_text(self, prompt=""):
        completion = self.client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role":"system",
             "content": "WHEN GENERATING YOUR ANSWER, DO NOT INCLUDE EXPLANATIONS, JUSTIFICATIONS OR EXTRA OUTPUT. ONLY ANSWER EXACTLY WHAT IS ASKED."
            },
            {
            "role": "user",
            "content": prompt,
            }
        ]
        )

        return completion.choices[0].message.content


if __name__ == "__main__":
    slogan = RandomSloganModel().generate_text()

    print(slogan)
