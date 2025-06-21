# Summary
This repository contains the implementation of the assignment that can passed all the tests. Some optimizations points still exist. But due to time and resource limitation, I can not finish the optimizations, also i skip the ablation experiments and training on Open Web Text for same consideration.

## Repo-Tree
```
.
├── PROJECTLOG.md # details during the project
├── scripts
│   ├── exp_2_7.py # encoding tinystories and owt
│   ├── inference.py # generate text
│   ├── train_bpe.py
|   ├── train_on_ts.py # train loop for tinystories
|   └── ...
│   
├── cs336_basics
|   ├── bpe_tokenizer.py
|   ├── bpe_train.py 
|   ├── MyModules.py # implementation of transformer
|   └── transformer_train.py # implementation of training function

├── tinystories_v1_result.png # result of first training
├── tinystories_v3_result.png # result of third training

## Sample Output

```bash
/home/wen/learn/cs336/assignment1-basics/data/models/finals/final_model_v3.pt
temperature=1.2, top_p=0.9
=====output======
prompt: he quick brown fox jumps over the lazy dog
output: . The fox thought that this was how they both had each other to serve good pets.
Lily and the brown fox worked together to serve the tired little bunny. They helped the small rabbit grow bigger, so it could play again. Soon, the slow little animal was no longer playful. It loved all the treats and kept them from a stranger to learn from them. They all became best friends and had many more fun adventures together.

prompt: Once upon a time,
output:  there was a dog named Bob. Bob had a home on a walk. He liked to walk on the path, even if he was a dog. One day, Bob met a cat named Tom. Tom said, "Hi, Bob! What's your name?"
Bob said, "I am Bob. I live in a clean house on the path." Tom wanted to discuss more things, but Bob wanted to go on a trip. Tom was sad because he didn't have a clean home. Bob had an idea to help Tom.
Bob thought hard about how to help Tom. He said, "Let's try singing to help
prompt: Tom and Lily are best friends.
output:  They like to hug and slide and spin in the park. But today, Tom and Lily are not good dancers. They make loud noises and cry. Their moms hear them and look at them.
Tom and Lily are sad. They miss their moms and dads. They wish they did not look at all the time. They want them to come back. But they can't. The fun. They start to cry and whine. They give up and spin in circles. They are not loyal. They are their loyal friends. They are sad.

prompt: Once upon a time there was a little dog Taffy who was very fond of food. Her trainer Lily would give treats every time they went to the park
output: . One day, while they were at the park, Taffy found a cone. He wanted to offer it to her, so he said "Yes, we can take it home. Can we show it to Lily?"
Tongy said "Let's go back!" They ran to the park, waving the cone as they arrived. When they got to the park, they could not believe their eyes. "We have so much food!" said Tim.
All the friends began to eat the melting cone. Lily picked up a red apple and starting pulp. She laughed and said, "Yummy! Yummy!" 
They played
```

# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

