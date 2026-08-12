import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class ColBERTEmbedder:
    def __init__(
        self,
        model_name: str = "jinaai/jina-colbert-v2",
        device: str | None = None,
        max_length: int = 8192,
    ):
        self.model_name = model_name
        self.max_length = max_length

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def embed_texts(self, texts: list[str], batch_size: int = 8) -> list[torch.Tensor]:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch(self, texts: list[str]) -> list[torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state

        attention_mask = inputs["attention_mask"]

        embeddings = []
        for i in range(len(texts)):
            mask = attention_mask[i]
            valid_token_count = mask.sum().item()
            text_embeddings = token_embeddings[i, :valid_token_count, :]

            normalized_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            embeddings.append(normalized_embeddings.cpu())

        return embeddings

    def embed_single(self, text: str) -> torch.Tensor:
        return self.embed_texts([text])[0]
