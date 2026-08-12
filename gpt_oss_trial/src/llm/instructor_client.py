"""Instructor client wrapper for structured responses."""

import os
import time
from typing import Type, TypeVar

import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class InstructorClientFactory:
    """Factory for creating Instructor-patched clients."""

    @staticmethod
    def create_gpt_oss_client(access_token: str, project_id: str, region: str = "global") -> instructor.Instructor:
        """
        Create Instructor client for GPT-OSS via Vertex AI.

        Args:
            access_token: Google Cloud access token
            project_id: GCP project ID
            region: GCP region

        Returns:
            Instructor-patched OpenAI client using JSON mode (GPT-OSS doesn't support forced tool calling)
        """
        base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints/openapi"

        client = OpenAI(base_url=base_url, api_key=access_token)

        return instructor.from_openai(client, mode=instructor.Mode.JSON)

    @staticmethod
    def create_claude_client(api_key: str) -> instructor.Instructor:
        """
        Create Instructor client for Claude.

        Args:
            api_key: Anthropic API key

        Returns:
            Instructor-patched Anthropic client
        """
        client = Anthropic(api_key=api_key)
        return instructor.from_anthropic(client)


class StructuredResponseClient:
    """Wrapper for structured response generation."""

    def __init__(self, client: instructor.Instructor, model_id: str) -> None:
        self.client = client
        self.model_id = model_id
        self.is_gpt_oss = "gpt-oss" in model_id or "openai/" in model_id

    def generate(
        self, prompt: str, response_model: Type[T], system_prompt: str | None = None, max_retries: int = 3
    ) -> tuple[T | None, dict]:
        """
        Generate structured response.

        Args:
            prompt: User prompt
            response_model: Pydantic model class for response
            system_prompt: Optional system prompt
            max_retries: Maximum retry attempts for validation failures

        Returns:
            Tuple of (response_object, metrics_dict) where metrics_dict contains:
            - success: bool
            - validation_error: str | None
            - latency: float
            - retry_count: int
        """
        start_time = time.time()
        metrics = {"success": False, "validation_error": None, "latency": 0.0, "retry_count": 0}

        messages = self._format_messages(prompt, system_prompt)

        response = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id, response_model=response_model, messages=messages, max_tokens=2000
                )
                metrics["success"] = True
                break
            except Exception as e:
                metrics["retry_count"] = attempt + 1
                metrics["validation_error"] = str(e)
                if attempt == max_retries - 1:
                    break

        metrics["latency"] = time.time() - start_time
        return response, metrics

    def _format_messages(self, prompt: str, system_prompt: str | None) -> list[dict]:
        """
        Format messages based on model type.

        GPT-OSS requires first message to be role="user", so system prompt
        is prepended to user content. Claude supports standard system messages.
        """
        if self.is_gpt_oss:
            return self._format_messages_gpt_oss(prompt, system_prompt)
        else:
            return self._format_messages_claude(prompt, system_prompt)

    def _format_messages_gpt_oss(self, prompt: str, system_prompt: str | None) -> list[dict]:
        """Format messages for GPT-OSS (first message must be user)."""
        content = prompt
        if system_prompt:
            content = f"{system_prompt}\n\n{prompt}"

        return [{"role": "user", "content": content}]

    def _format_messages_claude(self, prompt: str, system_prompt: str | None) -> list[dict]:
        """Format messages for Claude (supports system role)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
