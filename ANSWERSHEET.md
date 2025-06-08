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

> TODO: can further speed up

## training result on TinyStory
```bash
(base) tuoge@hltsz01:~/workspace/cs336/assignment1-basics$ uv run scripts/train_bpe_tinystories.py
[2025-06-08 17:11:47] [INFO] __main__: start training bpe on tinystories
[2025-06-08 17:11:47] [INFO] cs336_basics.bpe: pre_tokenize start
[2025-06-08 17:11:48] [INFO] cs336_basics.bpe: pre_tokenize cost time_cost=1.81s
[2025-06-08 17:11:48] [INFO] cs336_basics.bpe: initial_freq_count start
[2025-06-08 17:11:48] [INFO] cs336_basics.bpe: initial_freq_count cost time_cost=0.02s
training bpe: 100%|█████████████████████████████████████████████████████████████████| 9743/9743 [01:15<00:00, 128.83r/s]
[2025-06-08 17:13:04] [INFO] __main__: finish training bpe on tinystories, time_cost=77.46s
[2025-06-08 17:13:04] [INFO] __main__: dump file ...
[2025-06-08 17:13:04] [INFO] __main__: vocab size 10000
[2025-06-08 17:13:04] [INFO] __main__: merges size 9743
```
