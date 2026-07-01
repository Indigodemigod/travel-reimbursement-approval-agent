"""Google Gemini API client wrapper."""

import logging
from functools import lru_cache

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Return a cached Gemini client using the configured API key."""
    logger.debug("Initializing Gemini client for model %s", GEMINI_MODEL)
    return genai.Client(api_key=settings.google_api_key)


def generate_json_response(prompt: str) -> str:
    """Call Gemini 2.5 Flash and return a JSON string response."""
    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response")

    logger.info("Gemini response received (%d characters)", len(response.text))
    return response.text
