import tiktoken

from ..models import SourceContentBase, SourceContent


tokenizer = tiktoken.encoding_for_model("gpt-4")


def split_content_into_chunks_by_tokens(content: SourceContentBase, max_tokens: int, source_content_type: str) -> list[SourceContent]:
    """
    Split the content from a source content base object into chunks by tokens.
    """
    chunked_content: list[SourceContent] = []
    split_content = split_text(content.content, max_tokens)

    for i, text_chunk in enumerate(split_content):
        chunked_content.append(
            SourceContent(
                **content.model_dump(),
                chunk_index=i + 1,
                text_chunk=text_chunk,
                type=source_content_type,
            )
        )

    return chunked_content


def split_text(text: str, max_tokens: int) -> list[str]:
    split_texts = []
    current_split = []

    tokens = tokenizer.encode(text, disallowed_special=())
    # print(len(tokens))
    for token in tokens:
        current_split.append(token)
        if len(current_split) >= max_tokens:
            split_texts.append(tokenizer.decode(current_split))
            current_split = []

    if current_split:
        split_texts.append(tokenizer.decode(current_split))

    return split_texts


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, disallowed_special=()))
