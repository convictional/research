import tiktoken
from tqdm import tqdm


tokenizer = tiktoken.encoding_for_model("gpt-4")


def split_content_into_chunks_by_tokens(content: list[dict], content_key: str, max_tokens: int) -> list[dict]:
    """
    Split the content into chunks by tokens.
    """
    print(f"Splitting {len(content)} content items into chunks by {max_tokens} tokens...")
    chunked_content = []

    for c in tqdm(content, total=len(content), desc="Splitting chunks..."):
        split_content = split_text(c[content_key], max_tokens)

        for i, content in enumerate(split_content):
            chunked_content.append(
                {
                    **c,
                    "chunk_index": i + 1,
                    "text_chunk": content,
                }
            )

    print(f"Finished split into {len(chunked_content)} chunks")

    return chunked_content


def split_text(text: str, max_tokens: int) -> list[str]:
    split_texts = []
    current_split = []

    tokens = tokenizer.encode(text)
    for token in tokens:
        current_split.append(token)
        if len(current_split) >= max_tokens:
            split_texts.append(tokenizer.decode(current_split))
            current_split = []

    if current_split:
        split_texts.append(tokenizer.decode(current_split))

    return split_texts


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))
