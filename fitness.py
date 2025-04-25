"""
FitnessEvaluator module for evaluating the fitness of prompt enhancement strategies.
"""

import asyncio
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from openai import OpenAI
from enhancement_strategy import EnhancementStrategy
from population import Population
from individual import Individual
from config import default_config
from intent_analyzer import IntentAnalyzer
from cross_validator import ValidationResult

from unified_fitness_metrics import UnifiedFitnessMetrics
from enhanced_fitness_evaluator import EnhancedFitnessEvaluator
from fitness_evaluation_strategy import FitnessEvaluationStrategy
from evolution_history import EvolutionHistory
from cross_validator import CrossValidator

@dataclass
class EvaluationContext:
    """Context for fitness evaluation"""
    original_prompt: str
    intent_analysis: Dict[str, Any]
    generation: int

class FitnessEvaluator:
    """Evaluates the fitness of individuals based on unified metrics system."""

    def __init__(self, client: OpenAI, intent_analyzer: 'IntentAnalyzer'):
        self.client = client
        self.intent_analyzer = intent_analyzer
        self.cache: Dict[str, Dict[str, float]] = {}

        # Initialize new evaluation components
        self.metrics = UnifiedFitnessMetrics()
        self.evaluator = EnhancedFitnessEvaluator(client=client)
        self.strategy = FitnessEvaluationStrategy(client=client)
        self.history = EvolutionHistory()
        self.validator = CrossValidator(client=client)

    def _clean_llm_output(self, text: str, original_prompt: str) -> str:
        """Cleans common LLM artifacts like preambles and original prompt repetition."""
        if not text or not text.strip():
            return text

        cleaned_text = text.strip()
        original_prompt_stripped = original_prompt.strip()

        # 1. Remove markdown code fences (optional language specifier)
        cleaned_text = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned_text)
        cleaned_text = re.sub(r'\n?```$', '', cleaned_text)
        cleaned_text = cleaned_text.strip() # Strip again after removing fences

        # 2. Remove common prefixes (case-insensitive, optional colon/whitespace)
        # Covers "Enhanced Prompt:", "Adjusted Prompt:", "Revised Prompt:", "Mutated Prompt:", "Original:", etc.
        prefix_pattern = r'^(?:enhanced|adjusted|revised|mutated|original|improved|new|modified|output|result|final)\s*(?:prompt|version|text)?\s*[:\-]?\s*'
        cleaned_text = re.sub(prefix_pattern, '', cleaned_text, flags=re.IGNORECASE).strip()

        # 3. Check if the LLM has enhanced the prompt by building upon it
        # This is a common case where the LLM keeps the original prompt and adds to it
        if cleaned_text.startswith(original_prompt_stripped):
            # Get the part after the original prompt
            additional_content = cleaned_text[len(original_prompt_stripped):].strip()

            # If there's substantial additional content, keep the entire enhanced prompt
            # This preserves the original prompt plus the enhancements
            if len(additional_content) > 10:
                return cleaned_text

        # 4. Handle "Original: ... Enhanced: ..." format with various patterns
        for pattern in [
            r'(?:original|input)(?:\s*prompt)?(?:\s*[:\-])?\s*(.*?)(?:enhanced|output|improved|revised)(?:\s*prompt)?(?:\s*[:\-])?\s*(.*)',
            r'(?:original|input)[:\-]?\s*(.*?)(?:enhanced|improved|output)[:\-]?\s*(.*)',
            r'(?:original|input)[:\-]?\s*(.*?)\n+(?:enhanced|improved|output)[:\-]?\s*(.*)'
        ]:
            match = re.search(pattern, cleaned_text, re.IGNORECASE | re.DOTALL)
            if match and match.group(2) and len(match.group(2).strip()) > 20:
                # If the original part matches our original prompt, use the enhanced part
                if self._text_similarity(match.group(1).strip(), original_prompt_stripped) > 0.7:
                    return match.group(2).strip()

        # 5. Remove metadata and instructions that might have been added by the LLM
        # This handles cases where the LLM adds things like "Domain: Testing" or "Style: technical"
        metadata_pattern = r'(?:domain|style|context|keywords|constraints|format):\s*[^\n]+\n*'
        cleaned_text = re.sub(metadata_pattern, '', cleaned_text, flags=re.IGNORECASE)

        # Remove instructions like "Ensure the enhanced version maintains..."
        instruction_pattern = r'(?:ensure|make sure|verify)[^.]+(?:original|context|domain|intent)[^.]+\.'
        cleaned_text = re.sub(instruction_pattern, '', cleaned_text, flags=re.IGNORECASE)

        # Clean up any resulting double newlines and extra whitespace
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
        cleaned_text = cleaned_text.strip()

        # 6. Check if the text starts with a fragment (like a numbered list continuation)
        fragment_start_pattern = r'^(?:\d+\)|\d+\.|\-|\*|\,|\;)\s*'
        if re.match(fragment_start_pattern, cleaned_text):
            # If it starts with a fragment, it's likely not a complete prompt
            # Try to find a complete sentence in the original text
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            if len(sentences) > 1:
                # Use the first complete sentence that's substantial
                for sentence in sentences:
                    if len(sentence.strip()) > 30 and not re.match(fragment_start_pattern, sentence.strip()):
                        return sentence.strip()

            # If we couldn't find a good sentence, prepend the original prompt
            if len(cleaned_text) > 20:
                return f"{original_prompt_stripped} that includes {cleaned_text}"
            else:
                return original_prompt_stripped

        # 7. If the cleaned text is very short, it might be a fragment
        if len(cleaned_text) < 30:
            # If it's too short, it's likely not a complete prompt
            if len(text.strip()) > 50:
                # Try to extract a complete sentence from the original text
                sentences = re.split(r'(?<=[.!?])\s+', text.strip())
                if len(sentences) > 1:
                    # Use the first complete sentence that's substantial
                    for sentence in sentences:
                        if len(sentence.strip()) > 30:
                            return sentence.strip()

                # If we couldn't find a good sentence, return the original text
                return text.strip()
            else:
                # If the original text is also short, prepend the original prompt
                return f"{original_prompt_stripped} that {cleaned_text}"

        # 8. If we've made it this far, the LLM might have completely rewritten the prompt
        # Check if the cleaned text is substantially different from the original
        if self._text_similarity(cleaned_text, original_prompt_stripped) < 0.7 and len(cleaned_text) > 30:
            # If it's different enough and substantial, use the cleaned text
            return cleaned_text

        # 9. If none of the above conditions are met, return the original text
        # This is a fallback to ensure we don't lose content
        return text.strip()

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two texts.
        Returns a value between 0 (completely different) and 1 (identical).
        """
        # Convert to lowercase for comparison
        text1 = text1.lower()
        text2 = text2.lower()

        # If either text is empty, return 0
        if not text1 or not text2:
            return 0

        # Simple character-based similarity
        shorter = min(len(text1), len(text2))
        longer = max(len(text1), len(text2))

        # Count matching characters
        matches = sum(1 for i in range(shorter) if text1[i] == text2[i])

        # Return similarity ratio
        return matches / longer if longer > 0 else 0

    async def evaluate_population(
        self,
        population: Population,
        original_prompt: str,
        intent_analysis: Dict[str, Any]
    ) -> None:
        """Evaluate the fitness of all individuals in a population."""
        context = EvaluationContext(
            original_prompt=original_prompt,
            intent_analysis=intent_analysis,
            generation=population.generation
        )

        tasks = []
        for individual in population.individuals:
            # Check cache first
            cache_key = f"{individual.id}_{original_prompt}"
            if cache_key in self.cache:
                individual.fitness = self.cache[cache_key]
                continue

            # Create evaluation task
            task = self.evaluate_individual(individual, context)
            tasks.append(task)

        # Run evaluations in larger batches for better parallelization
        if tasks:
            batch_size = 10  # Process 10 evaluations at a time (increased from 5)
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]

                # Execute batch of evaluations concurrently
                await asyncio.gather(*batch)

                # Add minimal delay between batches to respect rate limits
                if i + batch_size < len(tasks):
                    await asyncio.sleep(0.2)  # 0.2 second delay between batches (reduced from 0.5)

        # Update evolution history
        self._update_history(population, context)

    async def evaluate_individual(
        self,
        individual: Individual,
        context: EvaluationContext
    ) -> None:
        """Evaluate the fitness of a single individual."""
        cache_key = f"{individual.id}_{context.original_prompt}"

        # Check cache first
        if cache_key in self.cache:
            individual.fitness = self.cache[cache_key]
            return

        try:
            # Apply the individual's strategy to get the enhanced prompt
            enhanced_prompt = await self._apply_strategy(
                context.original_prompt,
                individual.strategy,
                context.intent_analysis
            )

            # Store the result for later use
            individual.prompt_result = enhanced_prompt

            # Evaluate using the enhanced pipeline
            evaluation_context = {
                "original_prompt": context.original_prompt,
                "intent_analysis": context.intent_analysis,
                "generation": context.generation,
                "strategy": individual.strategy.__dict__,
                "client": self.client  # Add the OpenAI client to the context
            }

            # Get initial scores from the enhanced evaluator
            base_scores = await self.evaluator.evaluate(
                enhanced_prompt,
                evaluation_context
            )

            # Apply fallback strategy if needed
            if not self._is_valid_score(base_scores):
                base_scores = await self.strategy.evaluate_with_fallback(
                    enhanced_prompt,
                    evaluation_context
                )

            # Cross-validate scores
            validation_result = await self.validator.validate_fitness_scores(
                enhanced_prompt,
                evaluation_context,
                [context.original_prompt]  # history for innovation check
            )

            # Combine and normalize scores
            metrics = self._combine_scores(base_scores, validation_result)

            # Set the individual's fitness
            individual.fitness = metrics

            # Cache the result
            self.cache[cache_key] = metrics

        except Exception as e:
            print(f"Error evaluating individual {individual.id}: {str(e)}")
            # Use fallback evaluation
            try:
                metrics = await self.strategy.evaluate_with_fallback(
                    individual.prompt_result or context.original_prompt,
                    evaluation_context
                )
                individual.fitness = metrics
                self.cache[cache_key] = metrics
            except Exception as fallback_error:
                print(f"Fallback evaluation failed: {str(fallback_error)}")
                # Assign minimum fitness as last resort
                individual.fitness = {metric: 0.01 for metric in self.metrics.get_all_metrics()}
                self.cache[cache_key] = individual.fitness

    def _is_valid_score(self, scores: Dict[str, float]) -> bool:
        """Check if scores are valid and complete."""
        expected_metrics = set(self.metrics.get_all_metrics().keys())
        return all(
            metric in scores and isinstance(scores[metric], (int, float))
            for metric in expected_metrics
        )

    def _combine_scores(
        self,
        base_scores: Dict[str, float],
        validation_result: ValidationResult
    ) -> Dict[str, float]:
        """Combine and normalize evaluation scores."""
        weights = self.metrics.get_all_metrics()
        combined_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in base_scores:
                # Adjust score based on validation confidence
                confidence_factor = validation_result.confidence
                base_score = base_scores[metric]

                # If there's a significant discrepancy, adjust the score
                if metric in validation_result.discrepancies:
                    discrepancy = validation_result.discrepancies[metric]
                    adjusted_score = base_score * (1 - discrepancy)
                else:
                    adjusted_score = base_score

                # Apply confidence weighting
                final_score = max(0.3, adjusted_score * confidence_factor)
                combined_scores[metric] = final_score

                # Accumulate weighted sum for overall score
                weighted_sum += final_score * weight
                total_weight += weight
            else:
                # Fallback to minimum score if metric is missing
                combined_scores[metric] = 0.3
                weighted_sum += 0.3 * weight
                total_weight += weight

        # Calculate overall score
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.3

        # Ensure overall score is between 0.3 and 1.0
        combined_scores['overall'] = max(0.3, min(1.0, overall_score))

        return combined_scores

    def _update_history(
        self,
        population: Population,
        context: EvaluationContext
    ) -> None:
        """Update evolution history with current generation results."""
        generation_scores = {
            individual.id: (
                (sum(individual.fitness.values()) / len(individual.fitness))
                if isinstance(individual.fitness, dict)
                else individual.fitness
            ) if individual.fitness is not None else 0.0
            for individual in population.individuals
        }

        metadata = {
            "generation": context.generation,
            "population_size": len(population.individuals),
            "original_prompt": context.original_prompt,
            "intent_analysis": context.intent_analysis
        }

        self.history.add_generation_result(
            generation=context.generation,
            scores=generation_scores,
            metadata=metadata
        )

    async def _apply_strategy(
        self,
        prompt: str,
        strategy: EnhancementStrategy,
        intent_analysis: Dict[str, Any]
    ) -> str:
        """Apply the enhancement strategy to the prompt."""
        try:
            # Basic parameters
            completion_args = {
                "model": default_config.model_name,
                "temperature": strategy.temperature,
                "messages": [
                    {"role": "system", "content": strategy.system_prompt},
                    {"role": "user", "content": f"""
                    Enhance this prompt to make it more effective:

                    {prompt}

                    Provide only the enhanced prompt as output, no explanations.
                    """}
                ]
            }

            # Add optional parameters if specified
            if strategy.max_tokens:
                completion_args["max_tokens"] = strategy.max_tokens
            if strategy.top_p:
                completion_args["top_p"] = strategy.top_p
            if strategy.frequency_penalty:
                completion_args["frequency_penalty"] = strategy.frequency_penalty
            if strategy.presence_penalty:
                completion_args["presence_penalty"] = strategy.presence_penalty

            response = self.client.chat.completions.create(**completion_args)
            raw_enhanced = response.choices[0].message.content.strip()
            enhanced = self._clean_llm_output(raw_enhanced, prompt) # Clean the output

            # Apply additional processing based on strategy flags
            if strategy.semantic_check:
                validation_result = await self.validator.validate_fitness_scores(
                    enhanced,
                    {"original_prompt": prompt},
                    [prompt]
                )
                if validation_result.confidence < 0.7:
                    enhanced = await self._restore_semantic_meaning(enhanced, prompt, strategy)

            if strategy.context_preservation:
                enhanced = await self._preserve_context(enhanced, prompt, intent_analysis)

            return enhanced

        except Exception as e:
            print(f"Error applying strategy: {str(e)}")
            return prompt  # Return original prompt on error

    async def _restore_semantic_meaning(
        self,
        current: str,
        original: str,
        strategy: EnhancementStrategy
    ) -> str:
        """Restore semantic meaning while preserving improvements."""
        try:
            completion_args = {
                "model": default_config.model_name,
                "temperature": strategy.temperature,
                "messages": [
                    {"role": "system", "content": strategy.system_prompt},
                    {"role": "user", "content": f"""
                    The enhanced prompt may have diverged too much.
                    Original: {original}
                    Current: {current}

                    Revise the current version to maintain the core meaning of the original
                    while preserving the improvements. Return only the revised prompt.
                    """}
                ]
            }

            response = self.client.chat.completions.create(**completion_args)
            raw_restored = response.choices[0].message.content.strip()
            # Use 'original' prompt context for cleaning here
            return self._clean_llm_output(raw_restored, original)
        except Exception as e:
            print(f"Error in semantic restoration: {str(e)}")
            return current

    async def _preserve_context(
        self,
        current: str,
        original: str,
        intent_analysis: Dict[str, Any]
    ) -> str:
        """Ensure context, structure, and critical instructions are preserved during enhancement."""
        try:
            # Extract constraints from intent analysis
            constraints = intent_analysis.get('constraints', [])
            constraints_text = "\n".join([f"- {constraint}" for constraint in constraints])

            # Identify structural elements in the original prompt
            structural_elements = self._identify_structural_elements(original)

            response = self.client.chat.completions.create(
                model=default_config.model_name,
                temperature=0.3,  # Lower temperature for more conservative preservation
                messages=[
                    {"role": "system", "content": """You are a context and structure preservation specialist.
Your primary responsibility is to ensure that enhanced prompts maintain:
1. All critical instructions and verification steps from the original prompt
2. The structural integrity including sections, lists, and formatting
3. Domain-specific terminology and context
4. All safety-critical elements and verification requirements

NEVER remove sections, verification steps, or critical instructions from the original prompt.
If the enhanced version is missing any important elements from the original, restore them."""},
                    {"role": "user", "content": f"""
Carefully analyze and adjust the prompt to preserve ALL context, structure, and critical instructions:

ORIGINAL PROMPT:
{original}

CURRENT ENHANCED VERSION:
{current}

DOMAIN: {intent_analysis.get('domain', 'general')}
STYLE: {intent_analysis.get('style_tone', {}).get('style', 'neutral')}
TONE: {intent_analysis.get('style_tone', {}).get('tone', 'neutral')}
CONSTRAINTS:
{constraints_text}

STRUCTURAL ELEMENTS TO PRESERVE:
{structural_elements}

INSTRUCTIONS:
1. Ensure ALL sections, headings, and structural elements from the original are preserved
2. Maintain ALL verification steps, safety checks, and critical instructions
3. Preserve ALL numbered lists, bullet points, and formatting elements
4. Keep ALL domain-specific terminology and context
5. If ANY important elements are missing, restore them completely

Return ONLY the adjusted prompt with all critical elements preserved.
"""
                    }
                ]
            )
            raw_preserved = response.choices[0].message.content.strip()
            # Use 'original' prompt context for cleaning here
            return self._clean_llm_output(raw_preserved, original)
        except Exception as e:
            print(f"Error in context preservation: {str(e)}")
            return current

    def _identify_structural_elements(self, text: str) -> str:
        """Identify structural elements in the text that should be preserved."""
        structural_elements = []

        # Check for sections with headings (markdown style)
        heading_matches = re.findall(r'#{1,6}\s+(.+)$', text, re.MULTILINE)
        if heading_matches:
            structural_elements.append(f"Headings: {', '.join(heading_matches)}")

        # Check for numbered lists
        if re.search(r'^\d+\.\s+', text, re.MULTILINE):
            structural_elements.append("Numbered lists")

        # Check for bullet points
        if re.search(r'^[\*\-\+]\s+', text, re.MULTILINE):
            structural_elements.append("Bullet point lists")

        # Check for code blocks
        if re.search(r'```', text):
            structural_elements.append("Code blocks")

        # Check for tables
        if re.search(r'\|.*\|.*\|', text):
            structural_elements.append("Tables")

        # Check for verification steps or critical instructions
        verification_keywords = ['verify', 'ensure', 'check', 'confirm', 'validate', 'must', 'required', 'critical', 'important']
        for keyword in verification_keywords:
            if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
                structural_elements.append(f"Verification/critical instructions containing '{keyword}'")

        # Check for sections with all caps headers (non-markdown)
        caps_headers = re.findall(r'^([A-Z][A-Z\s]+[A-Z])$', text, re.MULTILINE)
        if caps_headers:
            structural_elements.append(f"ALL CAPS sections: {', '.join(caps_headers)}")

        # If no structural elements were found
        if not structural_elements:
            return "No specific structural elements identified, preserve the general structure and flow"

        return "\n".join([f"- {element}" for element in structural_elements])