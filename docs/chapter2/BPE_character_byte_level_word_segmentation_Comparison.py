# 字节级Tokenizer
from collections import Counter
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, indices):
        return bytes(indices).decode("utf-8")

# 字符级Tokenizer
class CharTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    def encode(self, text: str):
        tokens = []
        for ch in text:
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.inverse_vocab[idx] = ch
            tokens.append(self.vocab[ch])
        return tokens

    def decode(self, indices):
        return "".join(self.inverse_vocab[i] for i in indices)

# 计算压缩率（byte/token）
def get_compression_ratio(text: str, token_len: int):
    input_byte_len = len(text.encode("utf-8"))
    return input_byte_len / token_len if token_len > 0 else 1


# 简易 BPE Tokenizer
class BPETokenizer:
    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.merges = {}  # {(a,b): new_token_id}
        self.vocab_size = 256  # 从byte开始

    def get_stats(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i+1])] += 1
        return pairs

    def merge_tokens(self, tokens, pair, new_token):
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text: str):
        tokens = list(text.encode("utf-8"))

        for _ in range(self.num_merges):
            pairs = self.get_stats(tokens)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_token = self.vocab_size
            self.vocab_size += 1

            self.merges[best_pair] = new_token
            tokens = self.merge_tokens(tokens, best_pair, new_token)

    def encode(self, text: str):
        tokens = list(text.encode("utf-8"))

        # 按训练好的 merges 顺序应用
        for pair, new_token in self.merges.items():
            tokens = self.merge_tokens(tokens, pair, new_token)

        return tokens

    def decode(self, tokens):
        # 需要反向展开merges（递归）
        reverse_merges = {v: k for k, v in self.merges.items()}

        def expand(token):
            if token < 256:
                return [token]
            else:
                left, right = reverse_merges[token]
                return expand(left) + expand(right)

        byte_seq = []
        for t in tokens:
            byte_seq.extend(expand(t))

        return bytes(byte_seq).decode("utf-8")

if __name__ == "__main__":

    print("=== UTF-8测试 ===")
    assert bytes("a", encoding="utf-8") == b"a"
    assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"
    print("测试通过\n")

    text = "Hello, 🌍! 你好!"
    print("原始字符串：", text)
    print("原始字节长度：", len(text.encode("utf-8")))

    # 字节级
    byte_tokenizer = ByteTokenizer()
    byte_tokens = byte_tokenizer.encode(text)
    byte_ratio = get_compression_ratio(text, len(byte_tokens))

    # 字符级
    char_tokenizer = CharTokenizer()
    char_tokens = char_tokenizer.encode(text)
    char_ratio = get_compression_ratio(text, len(char_tokens))

    # BPE
    train_text = """
    Hello world! Hello world!
    Machine learning is fun.
    你好 世界 你好 世界
    """

    bpe_tokenizer = BPETokenizer(num_merges=20)
    bpe_tokenizer.train(train_text)

    bpe_tokens = bpe_tokenizer.encode(text)
    bpe_ratio = get_compression_ratio(text, len(bpe_tokens))

    # 对比输出
    print("\n=== 对比总结 ===")
    print(f"字节级 token数: {len(byte_tokens)}")
    print(f"字符级 token数: {len(char_tokens)}")
    print(f"BPE token数: {len(bpe_tokens)}")

    print("\n=== 压缩率(byte/token) ===")
    print(f"字节级: {byte_ratio:.2f}")
    print(f"字符级: {char_ratio:.2f}")
    print(f"BPE: {bpe_ratio:.2f}")

    print("\n=== 结论 ===")
    print("1️⃣ 字节级：无压缩，最稳定")
    print("2️⃣ 字符号级：用UTF-8字符压缩（中文/emoji更明显）")
    print("3️⃣ BPE：通过学习高频子串，实现真正“数据驱动压缩”")

    if bpe_ratio > char_ratio:
        print("\n由此可得，BPE压缩效果最好（最接近真实LLM tokenizer）")
