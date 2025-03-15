from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold
        self.word_count = {}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        idx = 2  # Start from 2 as 0 and 1 are reserved for PAD and UNK
        word_counts = Counter()

        for sentence in tqdm(sentence_list, desc="Building vocabulary"):
            for word in word_tokenize(sentence.lower()):
                word_counts[word] += 1

        for word, count in word_counts.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        print(f"Vocabulary size: {len(self.itos)}")

    def numericalize(self, text):
        tokenized_text = word_tokenize(text.lower())

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
    