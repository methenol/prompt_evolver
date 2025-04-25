"""
Intent Analyzer module for analyzing prompt intent.
"""

import json
import asyncio
from typing import Dict, Any, List

from openai import OpenAI
from config import default_config

class IntentAnalyzer:
    """Analyzes prompt intent for multi-faceted evaluation."""

    def __init__(self, client: OpenAI):
        self.client = client

    async def analyze_multi_faceted(self, prompt: str) -> Dict[str, Any]:
        """Perform multi-faceted intent analysis"""
        aspects = [
            self._analyze_style_and_tone(prompt),
            self._analyze_complexity(prompt),
            self._analyze_domain(prompt),
            self._analyze_constraints(prompt)
        ]
        results = await asyncio.gather(*aspects)
        return {
            "style_tone": results[0],
            "complexity": results[1],
            "domain": results[2],
            "constraints": results[3]
        }

    async def _analyze_style_and_tone(self, prompt: str) -> Dict[str, str]:
        """Analyze the style and tone of the prompt."""
        response = await self._get_completion(
            """Analyze the style and tone of this prompt. Format the response exactly like this JSON:
            {
                "style": "descriptive|narrative|technical|creative|formal|informal",
                "tone": "professional|casual|authoritative|friendly|serious|playful"
            }""",
            prompt,
            temperature=0.3
        )
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {"style": "neutral", "tone": "neutral"}

    async def _analyze_complexity(self, prompt: str) -> Dict[str, Any]:
        """Analyze the complexity of the prompt."""
        response = await self._get_completion(
            """Analyze the complexity of this prompt. Format the response exactly like this JSON:
            {
                "level": 3,
                "factors": ["multiple requirements", "technical terms"]
            }""",
            prompt,
            temperature=0.2
        )
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {"level": 1, "factors": []}

    async def _analyze_domain(self, prompt: str) -> str:
        """Identify the primary domain/field of the prompt."""
        response = await self._get_completion(
            "Identify the primary domain/field this prompt relates to. Respond with a single word or short phrase.",
            prompt,
            temperature=0.3
        )
        return response.strip()

    async def _analyze_constraints(self, prompt: str) -> List[str]:
        """Identify constraints or requirements in the prompt."""
        response = await self._get_completion(
            """Identify any explicit constraints or requirements mentioned in this prompt. Only include constraints that are directly stated or strongly implied in the prompt itself. Format the response as a JSON array of strings.

            Example format (but do not use these specific constraints unless they are actually in the prompt):
            ["constraint 1", "constraint 2"]""",
            prompt,
            temperature=0.2
        )
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return []

    async def _get_completion(self, instruction: str, prompt: str, temperature: float) -> str:
        """Get completion from the LLM."""
        try:
            response = self.client.chat.completions.create(
                model=default_config.model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You are a prompt analysis assistant. Always format responses exactly as requested, especially when JSON is required."},
                    {"role": "user", "content": f"{instruction}\n\nPrompt: {prompt}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in completion: {str(e)}")
            raise