from abc import ABC

from src.data_collection import slogans


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


if __name__ == "__main__":
    slogan = RandomSloganModel().generate_text()

    print(slogan)
