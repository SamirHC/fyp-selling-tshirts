from abc import ABC, abstractmethod


class TextSplitter(ABC):
    @abstractmethod
    def split(self, text: str) -> list[str]:
        pass


class WordSplit(TextSplitter):
    def split(self, text: str) -> list[str]:
        return text.split(" ")


class CharSplit(TextSplitter):
    def split(self, text: str) -> list[str]:
        return [char for char in text]


class LengthSplit(TextSplitter):
    def __init__(self, length: int):
        self.length = length

    def split(self, text: str) -> list[str]:
        return [text[i:i+self.length] for i in range(0, len(text), self.length)]


class WordIndexSplit(TextSplitter):
    def __init__(self, indices: list[int]):
        self.indices = indices
    
    def split(self, text: str) -> list[str]:
        words = text.split(" ")
        return [" ".join(words[i:j]) for i, j in zip(self.indices, self.indices[1:])]
