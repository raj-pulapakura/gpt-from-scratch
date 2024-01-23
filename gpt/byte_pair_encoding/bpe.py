# Don't run this file unless you want to create a new tokenizer
# The tokenizer has already been trained with 10000 merges, which took around 10 minutes
# If you want to see how to load the trained tokenizer, go to test.py

import json

def load_corpus() -> str:
    return open("../shakespeare.txt", "r", encoding="utf-8").read()

class BPETokenizer:
    def __init__(self):
        self._vocab = []
        self._merges = []
        self._toktoi = {}
        self._itotok = {}
        self._has_trained = False

    # getters
    @property
    def vocab(self):
        return self._vocab
    @property
    def vocab_size(self):
        return len(self._vocab)
    @property
    def merges(self):
        return self._merges
    @property
    def toktoi(self):
        return self._toktoi
    @property
    def itotok(self):
        return self._itotok
    @property
    def has_trained(self):
        return self._has_trained
    
    def _split_corpus(self, corpus: str) -> list[str]:
        """split corpus into words"""
        # split corpus by spaces
        pieces = corpus.split(" ")
        # add space character to end of each word
        pieces = [p + " " for p in pieces[:-1]] + [pieces[-1]]
        # remove any blank spaces
        pieces = list(filter(lambda x: x not in [" ", ""], pieces))
        # split each sentence into individual words, retaining newlines
        words = []
        for p in pieces:
            words.extend(p.splitlines(keepends=True))
        # if there are any stray newlines, append them to the previous word
        i = 0
        while i < len(words):
            if words[i] == "\n":
                words[i-1] += "\n"
                words.pop(i)
            i += 1
        return words

    def _get_char_splits(self, words: list[str]) -> list[list[str]]:
        """converts list of words into list of lists of chars"""
        for i in range(len(words)):
            words[i] = [c for c in words[i]]
        return words

    def _get_initial_vocab(self, corpus: str) -> list[str]:
        """get all unique characters in the corpus"""
        vocab = set(corpus)
        return sorted(list(vocab))

    def _get_frequencies(self, char_splits: list[list[str]]) -> dict:
        """get frequencies of adjacent symbols in list of list of chars"""
        freq = {}
        for word in char_splits:
            for i in range(0, len(word)-1):
                pair = (word[i], word[i+1])
                if pair in freq:
                    freq[pair] += 1
                else:
                    freq[pair] = 1
        return freq
    
    def _bpe(self, corpus:str, num_merges: int, verbose=False) -> tuple[list[str], list[tuple[str, str]]]:
        """
        constructs vocabulary from corpus using Byte Pair Encoding algorithm
        returns vocabulary and merge rules 
        """
        words = self._split_corpus(corpus)
        char_splits = self._get_char_splits(words)
        vocab = self._get_initial_vocab(corpus)
        merges = []

        for i in range(num_merges):
            adj_freqs = self._get_frequencies(char_splits)
            # add most frequent adjacent pair
            pair, freq = sorted(adj_freqs.items(), key=lambda x: x[1], reverse=True)[0]
            token = "".join(pair)
            vocab.append(token)
            merges.append(pair)
            if verbose: print(f"({i+1}) Adding: {repr(token)} | Frequency: {freq}")
            # update char splits with new token
            for cs in char_splits:
                i = 0
                while i < len(cs)-1:
                    candidate = f"{cs[i]}{cs[i+1]}"
                    if candidate == token:
                        cs[i] = token
                        cs.pop(i+1)
                    i += 1

        return vocab, merges

    def _get_mappings(self, vocab: list[str]) -> tuple[dict, dict]:
        """compute 'token to integer' and 'integer to token' mappings"""
        # token to integer mapping
        toktoi = {tok:i for i, tok in enumerate(vocab)}
        # integer to token mapping
        itotok = {i:tok for tok, i in toktoi.items()}
        return toktoi, itotok

    def train(self, corpus: str, num_merges: int, verbose=False):
        """run BPE algorithm on corpus"""
        if self._has_trained:
            print("You have already trained. Training again will erase your previous state.")
            cont = input("Do you want to continue? [y/n]")
            if cont != "y": return
        # learn vocab and merges
        vocab, merges = self._bpe(corpus, num_merges, verbose)
        self._vocab = vocab
        self._merges = merges
        # save mappings
        toktoi, itotok = self._get_mappings(vocab)
        self._toktoi = toktoi
        self._itotok = itotok

        self._has_trained = True

    def tokenize(self, s: str, verbose=False) -> list[int]:
        """convert string into list of integers using learned vocabulary"""
        if not self._has_trained:
            raise Exception("You have not trained yet.")
        # split string into list of words
        words = self._split_corpus(s)
        # split list of words into list of lists of chars
        char_splits = self._get_char_splits(words)
        # go through each merge rule, and merge matching pairs
        for i, merge in enumerate(self._merges):
            if verbose: print(f"({i}) {merge}")
            token = "".join(merge)
            for cs in char_splits:
                i = 0
                while i < len(cs)-1:
                    pair = f"{cs[i]}{cs[i+1]}" 
                    if pair == token:
                        cs[i] = token
                        cs.pop(i+1)
                    i += 1
        # convert tokens into integers
        tokens = [self._toktoi[token] for cs in char_splits for token in cs]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """convert list of integers into string using learned vocabulary"""
        if not self._has_trained:
            raise Exception("You have not trained yet.")
        return "".join([self._itotok[i] for i in tokens])
    
    def save_to_json(self, filename: str):
        """saves vocab and merge rules to json file"""
        data = {
            "vocab": self._vocab,
            "merges": self._merges,
        }
        with open(filename + ".json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_from_json(self, path: str):
        """loads vocab and merge rules from json file, and computes mappings"""
        if self._has_trained:
            print("You have already trained. Training again will erase your previous state.")
            cont = input("Do you want to continue? [y/n] ")
            if cont != "y": return
        # load vocab and merge rules
        f = open(path, "r")
        data = json.load(f)
        vocab, merges = data["vocab"], data["merges"]
        self._vocab = vocab
        self._merges = merges
        # compute mappings
        toktoi, itotok = self._get_mappings(vocab)
        self._toktoi = toktoi
        self._itotok = itotok
        
        self._has_trained = True

if __name__ == "__main__":

    # load corpus
    corpus = load_corpus()
    print("✅ Loaded corpus.")
    print(f"Number of chars: {len(corpus)}")
    print(f"Approx. number of words: {len(corpus.split(' '))}")
    print(f"Approx. number of unique words: {len(set(corpus.split(' ')))}")

    # train tokenizer
    tokenizer = BPETokenizer()
    print("\n🥷 Training tokenizer:\n")
    tokenizer.train(corpus, num_merges=1000, verbose=True)

    # save tokenizer vocab and merge rules to json
    print("\n🚀 Saving tokenizer")
    tokenizer.save_to_json("tokenizer")

    print("\n✨ Done!")