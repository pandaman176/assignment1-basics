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
compress ration goes down when encode open web text shows thathe tokenizer trained
on tiny stories is less efficiency in encoding OWT

Our tokenizer achieves a throughput of approximately 205,437 bytes/second. At this rate, it would take about 46 days to tokenize the full 825GB Pile dataset.

uint16 is used to save the ids, which is more efficient than uint32 since token_id wouldn't be very large(10k on ts and 32k on owt)

# 3.6
Note: Although in the guide paper, transformer_lm end with softmax, but in code it is not since we directly use log-softmax to calculate loss, if we use softmax, we can not pass the test case.