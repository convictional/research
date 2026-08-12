import tiktoken
from tqdm import tqdm
from typing import List


tokenizer = tiktoken.encoding_for_model("gpt-4")


def split_chunks_by_tokens(content_chunks: list[dict], max_tokens: int = 10000) -> list[dict]:
    """
    Split the content chunks by tokens.
    """
    print(f"Splitting {len(content_chunks)} content chunks by {max_tokens} tokens...")
    chunked_content = []

    for chunk in tqdm(content_chunks, total=len(content_chunks), desc="Splitting chunks..."):
        split_content = split_text(chunk["content"], max_tokens)

        for i, content in enumerate(split_content):
            chunked_content.append(
                {
                    "content_id": chunk["content_id"],
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "content": content,
                    "created_at": chunk["created_at"],
                    "updated_at": chunk["updated_at"],
                    "chunk_index": i + 1,
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


def trunc_on_tokens(results: dict, max_tokens: int = 150000) -> dict:
    tokenized_tool_results = tokenizer.encode(str(results))
    total_tokens = len(tokenized_tool_results)
    if total_tokens > max_tokens:
        tokenized_tool_results = tokenized_tool_results[:max_tokens]

    results = tokenizer.decode(tokenized_tool_results)
    return results


def get_tokens_from_text(text: str) -> List[int]:
    return tokenizer.encode(text)


def get_tokens_from_text_batch(text: str) -> List[int]:
    return tokenizer.encode_batch(text)


def get_text_from_tokens(tokens: List[int]) -> str:
    return tokenizer.decode(tokens)
