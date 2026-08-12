"""
Main entry point for LoRA Cassettes experiments.

See PLAN.md for implementation plan (Nov 2, 2025).
"""

from .model.download import EncoderModelDownloader


async def main():
    downloader = EncoderModelDownloader()
    model, tokenizer = await downloader.download_encoder_model()
    await downloader.save_model(model, tokenizer)
