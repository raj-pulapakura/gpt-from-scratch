from bpe import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.load_from_json("tokenizer.json")

string = "That is absolutely magnificent your highness"
print("Input:")
print(string)

tokens = tokenizer.tokenize(string)
print("\nIntegers:")
print(tokens)

print("\nTokens:")
print([tokenizer.itotok[i] for i in tokens])

print("\nDecoded:")
print(tokenizer.decode(tokenizer.tokenize(string)))