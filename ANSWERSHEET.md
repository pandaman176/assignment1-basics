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