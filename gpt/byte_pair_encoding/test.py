from bpe import BPETokenizer, load_corpus

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

tokenize_entire_corpus = True
if tokenize_entire_corpus:
    corpus = load_corpus()
    print("\n⌚ Tokenizing entire corpus...")
    tokenizer.tokenize(corpus, verbose=True)
    print("✨ Done")