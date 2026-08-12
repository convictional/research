import tiktoken


tokenizer = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = tokenizer.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])
