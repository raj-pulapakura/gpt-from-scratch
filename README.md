Andrej Karpathy has a wonderful series in which he builds out neural networks and language models from scratch, all the way up to implementing a Transformer! Check out [his playlist](
https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).

# Makemore (neural network fundamentals)

| Notebook       | Andjrey's Video |
| ----------- | ----------- |
| building makemore part 1.ipynb      | [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2)       |
| building makemore part 2 MLP.ipynb   | [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)        |
| building makemore part 3 activations,gradients,batchnorm.ipynb | [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2)       |
| building makemore part 5 wavenet.ipynb      | [Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=6)       |

# GPT

In Andrey's [*Let's build GPT: from scratch, in code, spelled out.*](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7), we implemented a GPT (Generative Pretrained Transformer) from scratch and train it on a corpus of Shakespeare Text. You can find the code for my implementation in the `gpt` folder.

| File       | Description |
| ----------- | ----------- |
| bigram 1.py | Implemented a bigram model, which uses the immediately previous char to predict the next char |
| gpt 2 with attention.py | Added Scaled Dot-Product Self-Attention to the model |
| gpt 3 ffwd.py | Continued the implementation of the Transformer with Feed Forward Neural nets |
| gpt 4 transformer blocks.py | Built transformer blocks containing multi-head attention and ffwd |
| gpt 5 residual connections.py | Added residual connections to mitigate vanishing gradients |
| gpt 6 layernorm.py | Added Layer Normalization |
| gpt 7 scaling up.py | Basically scaled up every hyperparameter of the network, achieves pretty decent generations (Train on a GPU) |
| gpt-dev.ipynb | Notebook which explains self-attention |
