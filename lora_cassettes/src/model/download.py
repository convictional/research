"""
Fetch open-source encoder models from Hugging Face.
"""

from transformers import AutoModel, AutoTokenizer


class EncoderModelDownloader:
    """
    A class to download encoder models from Hugging Face.
    """

    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        self.model_name = model_name

    async def download_encoder_model(self):
        """
        Download an encoder model and its tokenizer from Hugging Face.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        return model, tokenizer

    async def save_model(
        self, model: AutoModel, tokenizer: AutoTokenizer, save_directory: str = "./base_models/e5-base-v2"
    ):
        """
        Save the encoder model and tokenizer to a specified directory.
        """
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
