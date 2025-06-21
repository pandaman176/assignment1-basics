# 2.1

(a) What Unicode character does chr(0) return?
null character

(b) How does this character’s string representation (__repr__()) differ from its printed representation?
```bash
>>> print(chr(0).__repr__())
'\x00'
>>> print(chr(0))

```

(c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
```bash
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
'\x00' is not visible character so `print()` won't print it. 

# 2.2

(a)What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
utf-8 is more widely used, saves space for long text.

(b)Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>>decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
e.g. "你好“.encode() gives b'\xe4\xbd\xa0\xe5\xa5\xbd' where each chinese character takes
3 byte, and convert '\xe4' back to something is impossible. 

(c)Give a two byte sequence that does not decode to any Unicode character(s).
b'\xe4\xbd'

# 2.5

Problem (train_bpe): BPE Tokenizer Training
```bash
=================================================== test session starts ===================================================
platform linux -- Python 3.12.3, pytest-8.3.5, pluggy-1.5.0
rootdir: /home/wen/learn/cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.1
collected 3 items                                                                                                         

tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED

==================================================== 3 passed in 9.54s ====================================================
```
> Note: the test may failed on wsl/windows since file is in CRLF format.
Solution: convert file to LF format or use `git config --global core.autocrlf input` to convert automatically before git clone

## training result on TinyStory

longest token: b' accomplishment'
most time spent on merges

## training result on Online Web Text

Too long to take: estimate 3days to finish

# 2.6

```bash
(base) tuoge@hltsz01:~/workspace/cs336/assignment1-basics$ uv run pytest tests/test_tokenizer.py
================================================= test session starts ==================================================
platform linux -- Python 3.11.7, pytest-8.3.5, pluggy-1.5.0
rootdir: /home/tuoge/workspace/cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.1
collected 25 items

tests/test_tokenizer.py::test_roundtrip_empty PASSED
tests/test_tokenizer.py::test_empty_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_character PASSED
tests/test_tokenizer.py::test_single_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_unicode_character PASSED
tests/test_tokenizer.py::test_single_unicode_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_ascii_string PASSED
tests/test_tokenizer.py::test_ascii_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string PASSED
tests/test_tokenizer.py::test_unicode_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens PASSED
tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken PASSED
tests/test_tokenizer.py::test_overlapping_special_tokens PASSED
tests/test_tokenizer.py::test_address_roundtrip PASSED
tests/test_tokenizer.py::test_address_matches_tiktoken PASSED
tests/test_tokenizer.py::test_german_roundtrip PASSED
tests/test_tokenizer.py::test_german_matches_tiktoken PASSED
tests/test_tokenizer.py::test_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_special_token_trailing_newlines PASSED
tests/test_tokenizer.py::test_encode_special_token_double_newline_non_whitespace PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_iterable_memory_usage PASSED
tests/test_tokenizer.py::test_encode_memory_usage XFAIL (Tokenizer.encode is expected to take more memory th...)

====================================== 24 passed, 1 xfailed in 3349.54s (0:55:49) ======================================
```

# 2.7
```bash
(cs336-basics) wen@~/learn/cs336/assignment1-basics (main)$ uv run scripts/exp_2_7.py 
[2025-06-10 15:34:05] [INFO] __main__: tiny story tokenizer compress ratio on tinystories: 1.55
[2025-06-10 15:34:05] [INFO] __main__: tiny story tokenizer compress ratio on owt: 1.34
[2025-06-10 15:34:05] [INFO] __main__: throughput : 205437.20bytes/s
```
## TODO optimize bpe_tokenizer
利用反向索引加速merge pair的过程 可以进一步加速训练
compress ration goes down when encode open web text shows thathe tokenizer trained
on tiny stories is less efficiency in encoding OWT

Our tokenizer achieves a throughput of approximately 205,437 bytes/second. At this rate, it would take about 46 days to tokenize the full 825GB Pile dataset.

uint16 is used to save the ids, which is more efficient than uint32 since token_id wouldn't be very large(10k on ts and 32k on owt)

# 3.6 Resources Acounting for Transformer LM
Note: Although in the guide paper, transformer_lm end with softmax, but in code it is not since we directly use log-softmax to calculate loss, if we use softmax, we can not pass the test case.

> N(total num of tokens) = B(batch size) = S(sequence length)
### MHSA
**#param** = $3d_m \times d_m + d_m \times d_m$ (qkv + out) = $4d_m^2$
| 项目             | FLOPs                                   |
| -------------- | --------------------------------------- |
| qkv projection | $6N d_m^2$                              |
| RoPE       | $6N d_m$                                |
| QK^T           | $2B S^2 d_m$                            |
| softmax        | $4B h S^2$                              |
| AV             | $2B S^2 d_m$                            |
| out projection | $2N d_m^2$                              |
| **总 FLOPs**    | $8N d_m^2 + 4B S^2 d_m + \text{[RoPE]+[Softmax]}$ |

### FFN
**#param** = $3d_m \times d_{ff}$ (w1 + w2 + w3) 

| 项目             | FLOPs                                   |
| -------------- | --------------------------------------- |
| w1 projection | $2N d_{ff} d_m$                              |o
| w2 projection | $2N d_m d_{ff}$                              |
| sigmoid(w1_x) | $4N$ 4FLOP/element                          |
| w1_x * sigmoid * w2_x | $2 N d_{ff}$ 
| w2 projection | $2N d_{ff} d_m$                              |
| **总 FLOPs**    | $6N d_{ff} d_m + [2N d_{ff} + 4N]$ |

### RMSNorm
**#param** = $d_m$

| 步骤                     | FLOPs / token                         |
| ---------------------- | ----------------------------- |
| $x^2$               | $d_m$                       |
| average（addition + division）           | $d_m-1$ add + 1 div ≈ $d_m$ FLOPs |
| plus $\epsilon$ and sqrt       | 1 add + 1 sqrt ≈ 2 FLOPs    |
| 除法归一化 $x / \text{rms}$ | $d_m$ div                       |
| 乘以 $\gamma$            | $d_m$ mul                       |

**总 FLOPs** = $(2d_m+ 2 + 2d_m ) \times N= (4d_m + 2) \times B S$ (can be ignore)

### Transformer Block (MHSA + FFN + 2RMSNorm)
**#param** = $4d_m^2 + 3d_m \times d_{ff} + 2d_m$

**总 FLOPs** = $(8N d_m^2 + 4B S^2 d_m) + 6N d_{ff} d_m$
(only count the FLOPs of matrix multiplication)

### Token Embedding （tensor indexing）
$d_v$ = vocab size

**#param** = $d_m d_v$

no matrix multiplication

### final LayerNorm (RMSNorm)
**#param** = $d_m$

**FLOPs** = $4N d_m$ (can be ignore)

### output embedding （Linear）
$d_v$ = vocab size

**#param** = $d_m d_v$
**FLOPs** = $2N d_m d_v$


## TOTAL
$n$ = num of transformer blocks

**TOTAL PARAM** 
$$
\begin{aligned}
&= d_m d_v + n(4d_m^2 + 3d_m d_{ff} + 2d_m) + d_m + d_m d_v  \\
&= 4n d_m^2 + 3n d_m d_{ff} + (2n+1)d_m + 2d_m d_v \\
\end{aligned}
$$  

**TOTAL FLOPs**
$$
\begin{aligned}
&= n(8N d_m^2 + 4B S^2 d_m + 6N d_{ff} d_m) + 2Nd_m d_v
\end{aligned}
$$

(a) Consider GPT-2 XL, which has the following configuration:

**vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400**

Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?

#param = 2,127,057,600 required memory = 7.924 GB

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length tokens.

#FLOPs = 4,513,336,524,800 

(c) Based on your analysis above, which parts of the model require the most FLOPs?

transformer block

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24
layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the
model size increases, which parts of the Transformer LM take up proportionally more or less of
the total FLOPs?
Deliverable: For each model, provide a breakdown of model components and its associated
FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a
one-to-two sentence description of how varying the model size changes the proportional FLOPs
of each component.

| 模型           | Attention (%) | FFN (%) | Output Head (%) | 总 FLOPs（TFLOPs） |
| ------------ | ------------- | ------- | --------------- | --------------- |
| GPT-2 Small  | 99.37%        | 0.44%   | 0.20%           | 0.04            |
| GPT-2 Medium | 99.32%        | 0.58%   | 0.10%           | 0.10            |
| GPT-2 Large  | 99.21%        | 0.72%   | 0.07%           | 0.20            |
| GPT-2 XL     | 99.05%        | 0.90%   | 0.05%           | 0.33            |

随着模型规模增大（特别是层数和 d_model 增加）：
注意力机制（Attention）始终占主导地位（>99%），但其比例略微下降。
前馈网络（FFN）FLOPs 占比上升，从 0.44% 增长到 0.90%，表明其相对开销增大。
输出层开销占比递减，说明模型越大，输出层的计算占比越小。

(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one
forward pass change?
How do the relative contribution of FLOPs of the model components
change?

当将 GPT-2 XL 的上下文长度从 1,024 增加到 **16,384** 时，**单次前向传播的总 FLOPs 从约 0.33 TFLOPs 激增到 82.47 TFLOPs**，增加了约 **250 倍**。

与此同时，**几乎所有 FLOPs 都集中在注意力机制（占比约 100%）**，而 FFN 和输出层的计算量占比趋近于 0，这表明**长上下文时注意力计算成为瓶颈**。

这是因为注意力计算中的 $4BS^2 d_m$ 项对上下文长度非常敏感，增长为平方级别，大于其他线性项。


# 4.2 Fine-tuning learning rate
As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s
see that in practice in our toy example. Run the SGD example above with three other values for the
learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each
of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of
training)?

```bash
(cs336-basics) wen@~/learn/cs336/assignment1-basics (main)$ uv run scripts/learning_rate_tuning.py 
learning rate 10.0
    iteration 0, loss 25.616933822631836
    iteration 1, loss 16.39483642578125
    iteration 2, loss 12.08557415008545
    iteration 3, loss 9.455672264099121
    iteration 4, loss 7.659092903137207
    iteration 5, loss 6.350265979766846
    iteration 6, loss 5.355607509613037
    iteration 7, loss 4.576518535614014
    iteration 8, loss 3.9521842002868652
    iteration 9, loss 3.442791700363159
learning rate 100.0
    iteration 0, loss 3.021080255508423
    iteration 1, loss 3.0210800170898438
    iteration 2, loss 0.5183352828025818
    iteration 3, loss 0.012404930777847767
    iteration 4, loss 2.3065317613026662e-17
    iteration 5, loss 2.570772912262475e-19
    iteration 6, loss 8.656697141960322e-21
    iteration 7, loss 5.156853662060393e-22
    iteration 8, loss 4.423881784925862e-23
    iteration 9, loss 4.915424249298786e-24
learning rate 1000.0
    iteration 0, loss 6.640196240576474e-25
    iteration 1, loss 2.397110496735385e-22
    iteration 2, loss 4.140186953609368e-20
    iteration 3, loss 4.6055162439783374e-18
    iteration 4, loss 3.7304678308861113e-16
    iteration 5, loss 2.3543521326290298e-14
    iteration 6, loss 1.2086473315242596e-12
    iteration 7, loss 5.2001181138905395e-11
    iteration 8, loss 1.9166528364422675e-09
    iteration 9, loss 6.154584752948722e-08
```

it start to diverge when learning rate get bigger


# 4.3 Resources Acounting for AdamW

Let us compute how much memory and compute running AdamW requires. Assume we are using
float32 for every tensor.

(a) How much peak memory does running AdamW require? Decompose your answer based on the
memory usage of the parameters, activations, gradients, and optimizer state. Express your answer
in terms of the batch_size and the model hyperparameters (vocab_size, context_length,
num_layers, d_model, num_heads). Assume d_ff = 4 × d_model.
For simplicity, when calculating memory usage of activations, consider only the following compo-
nents: 
* Transformer block
    - RMSNorm(s)
    - Multi-head self-attention sublayer: QKV projections, Q⊤K matrix multiply, softmax,
    weighted sum of values, output projection.
    - Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
* final RMSNorm
* output embedding
* cross-entropy on logits

#parameter = $4n d_m^2 + 3n d_m d_{ff} + (2n+1)d_m + 2d_m d_v + $ \
#gradiant = #parameter \
#optimizer state = 2 * #parameter (m and v) \
#activation

    QKV: batch_size * seq_len * d_model
    attention score: batch_size * seq_len * seq_len
    atention output: batch_size * seq_len * d_model
    FFN w1 output: batch_size * seq_len * 4d_model
    FFN siglu output: batch_size * seq_len * 4d_model
    final_output: batch_size * seq_len * d_model
    logits: batch_size * seq_len * vocab_size
total activation: $B S d_v + (12n+1)BSd_m + nBS^2$

**total memory cost** = \
**4*total memory cost of parameters** + **total memory cost of activations** \
~ **30GB**

(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on
the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?

mem(B)=4.39B + 23.6 (GB)
$mem(B) \le 80GB \Rightarrow B \le 12$

(c) How many FLOPs does running one step of AdamW take?

| 步骤                        | 操作                                     | FLOPs/param        |
| ------------------------- | -------------------------------------- | ------------------ |
| 一阶矩更新（`m_t`）              | 2 mul + 1 add                          | 3                  |
| 二阶矩更新（`v_t`）              | 2 mul + 1 add + 1 square               | 4                  |
| 偏差校正（`m_hat`, `v_hat`）    | 2 div                                  | 2                  |
| 参数更新（`-lr * m / sqrt(v)`） | 1 sqrt + 1 add + 1 div + 1 mul + 1 sub | \~5                |
| 权重衰减                      | 2 mul + 1 sub                          | 3                  |
| **合计**                    |                                        | **17** FLOPs/param |

**Total flops = 17 * 2,12 * 1e9 ~= 3.6 * 1e10**


(d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second)
relative to the hardware’s theoretical peak FLOP throughput [Chowdhery et al., 2022].
An NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming
you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a
batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022],
assume that the backward pass has twice the FLOPs of the forward pass.

**4.6days**


# Training on tiny stories
## v0
problem: gradient explosion
reason: RMS parameter(scale) 因该用torch.ones初始化而不是torch.empty
否则会导致梯度爆炸
->需要合理进行系数初始j

## v1 and v2
```bash
(cs336-basics) wen@~/learn/cs336/assignment1-basics (main)$ uv run scripts/generate.py 
=====output====== # on checkpoint_v1_16000.pt
prompt: he quick brown fox jumps over the lazy dog
output: y sawnd salad. She had to clap h n and she was stiid a happy box.
Her m The came  T and sall, "What are you doing  Tnd salad?"
Lily replied, "It's veryarice, little girl.om bird flapped her wings,
prompt: Once upon a time,
output:  butndn Sam saw thatnd tree wasarot green. He was fierce and began to snapnd sunrise. It was dimk and dirty. 
Sam felt sad and hungry. He went to his h Thee and sall, "Don't worry, it's just for
prompt: Tom and Lily are best friends.
output:  He was sad.
One day, one worker saw a smaid, fluffy cat that wasarot for a place to sayaro, butaro. He thought it was his best frie the outslle and wanted to play withnd place to hlle.
The little bu
(cs336-basics) wen@~/learn/cs336/assignment1-basics (main)$ uv run scripts/generate.py 
=====output====== # on checkpoint_v1_32000.pt
prompt: he quick brown fox jumps over the lazy dog
output: ore,ndy saw a big fire. "M The, look what I fou the!" Lucy sall. Her m The smiled and sall, "Oh, that's a big firework! Let's prepare fornd big fire truck!"
But when Lucy sawnd mess, she
prompt: Once upon a time,
output:  he wanted to cl nb it. He asked his frie the, Sam, to help h n. Sam was very excited to reachnd big tree.
Sam tooknd big tree and cl nbed up.omre was a big pile of food and a lot of yummy food.
prompt: Tom and Lily are best friends.
output:  Maybe she has a choice to dr Tk warm. She has value to dr Tk.
One day, she walks tond pimk with her m The. She likesnd po the very much. She seesnd po the andnd sun andnd sky. She goes down and sllles down
=====output====== # on checkpoint_v1_48000.pt
prompt: he quick brown fox jumps over the lazy dog
output: . It was a big, gray caterpiidim!
Asin n ran out ofnd bag, s Theething unexpected happened.om bag was green and colorful too! It wasarot just a toy thatin n had been hlling, butnd ro The was emp
prompt: Once upon a time,
output:  so she setnd bananas down to h n.
Lucy gavend bananas to her frie the.omy took turns eatingnd bananas.omy raced high and fast. Lucy was very happy and had so much fun. After a while,
prompt: Tom and Lily are best friends.
output: 
"What are you doing,in The?" Mia asked.
"I am drawing and a big dog and it gets very unc Thefortable. It can breaknd pages,"in The sall.
Mia dllarot like being grumpy. She dllarot want to g
=====output====== # on final_v1.pt
prompt: he quick brown fox jumps over the lazy dog
output:  cont Tued to sp Tnd wa the and laugh. But when M The la theed onnd tree, she laughed. "Wow, Lily, that was a fun surprise. Let's go play!" M The sall.
Lily was sad at first, but she knew M The li
prompt: Once upon a time,
output:  Jazz had an llea. He sall, "Let's fi the s Thee water and makend music pimty for everyone!"
Jazz fou the a pretty vase and helped Beida fiid it with water.omn,ndy put a few everything toget
prompt: Tom and Lily are best friends.
output: om dog's owner cl nbed up a taid mounta T. Whenndy got tond top,ndy saw a yummy cake!in nmy andnd dog shimednd cake with aidndir frie thes. Everyone lovednd cake, andin nmy ate aidnd cake she m
=====output====== # on checkpoint_v1_2000.pt
prompt: he quick brown fox jumps over the lazy dog
output:  zooms underst park Sparkle Jones filthy quacks rudgles todd Slinky Frankie charming Springbol gardenensiveZero squashes knewquito raked rockets Okay sore chewedellie Patch Daisy dangerous courtatter war wr ashamed Chirp spid dessertsotte whenma mixingedy Meganll mo brick bouncingipesBang fa lights wheiness thougim Adamasketama Ball butter Sophiaaensiveabbyobile Janey Adam strum itchyizes best aloudhesareip. Sue�as tummy dull guitarsurbetterid and A the Snuggles Mimiswaybued her kit was too his too.
 crI am
prompt: Once upon a time,
output:  butin n and thingElla tear Sm app protectionSque her Teal's about.om bird dllarot aboutnd wroug would.om frog sou thes teals onndir have to share.omy laughed, s Theething right ever aboutnd cawiJalk.om ogknd lea
prompt: Tom and Lily are best friends.
output: om little feratla was jeep!" sall  Tinais, she sall.
Adpl shape waved an nals withnd big box.om an crawls Oxygen goodbyeso, butaramedin n was sad. Aidndy cel Fighting vanillain n and Bow. He is she was too strong of acong.omarext was chased
```
problem: val_loss and train_loss are very low but generation quality is shit
reason: 数据出现了问题, encode 大型文件时没有mmap，存储过大的nparray导致内容出现了异常

**v3**
```bash 
/home/wen/learn/cs336/assignment1-basics/data/models/checkpoints/checkpoint_v3_12000.pt
temperature=1.2, top_p=0.9
=====output======
prompt: he quick brown fox jumps over the lazy dog
output:  while trying to fetch up for herself. He smiled and grabbed one of his bricks while Sam was holding on tight.
The little girl grabbed a berry from Joe, and she handed the berry to her mum. But there was nothing left ahead. She felt scared, so she quickly called for a relieved little girl. The mean fox reminded her: Now it is too late to understand things - never to separate the scent the next time she ever got here.

prompt: Once upon a time,
output:  in a messy room, there was a girl named Anna. She had a big bottle of shampoo. She loved to drink juice from it every day. One day, a dog named Spot came running to play with Anna. Spot was very envious of Anna's wine. They both wanted their bottle.
Anna decided to tell their truth. She felt happy and fast. She said, "Sue, I have a toy for you. We can have some and share our juice." Sue nodded and smiled. She liked balloons too. Spot gave her the bottle of wine. It was not so happy.
Anna gave Spot all her cocoa. Spot
prompt: Tom and Lily are best friends.
output:  They do not hate each other.

prompt: Once upon a time there was a little dog Taffy who was very fond of food. Her trainer Lily would give treats every time they went to the park
output:  to try new things. They would pause wondering what they needed.
One day, T's friend over the shop, came over to trade. At first, Tealing with a screw. He wanted a toy though so he asked his third friend to trade if he could trade. Lily thought for a moment and then smiled.
E him gave Toni a pair of big, red apples for Santa's senses! Toni were so happy to share what he saw. He showed everyone how to get all of the things he found in the store. From then on, Toni helped Toy mummy and two brothers and Dog.
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