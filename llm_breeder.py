"""
LLM Breeder module for using LLMs to perform breeding operations on prompts.
"""

import json
import re
import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI
from individual import Individual
from enhancement_strategy import EnhancementStrategy
from config import default_config

from unified_fitness_metrics import UnifiedFitnessMetrics
from enhanced_fitness_evaluator import EnhancedFitnessEvaluator
from fitness_evaluation_strategy import FitnessEvaluationStrategy
from evolution_history import EvolutionHistory
from cross_validator import CrossValidator, ValidationResult

@dataclass
class BreedingContext:
    """Context for breeding operations"""
    original_prompt: str
    intent_analysis: Optional[Dict[str, Any]] = None
    generation: Optional[int] = None

class LLMBreeder:
    """Uses LLMs to perform breeding operations directly on prompts."""

    def __init__(self, client: OpenAI):
        self.client = client
        self.model_name = default_config.model_name

        # Initialize evaluation components
        self.metrics = UnifiedFitnessMetrics()
        self.evaluator = EnhancedFitnessEvaluator(client=client)  # Pass client to evaluator
        self.strategy = FitnessEvaluationStrategy(client=client)
        self.validator = CrossValidator(client=client)
        self.history = EvolutionHistory()

        # Using OpenAI client's built-in rate limiting
        self.base_retry_delay = 0.5  # Base delay for our own retry logic
        self.max_retries = 3  # Maximum number of our own retries (OpenAI client will also retry)

    # We're now using OpenAI's built-in rate limiting, so this method is simplified
    async def _handle_rate_limit(self, retry_count: int) -> None:
        """
        Simple helper method to handle rate limiting with exponential backoff.
        Args:
            retry_count: Current retry count
        """
        # Calculate exponential backoff delay
        delay = self.base_retry_delay * (2 ** retry_count)
        print(f"Rate limit or error encountered. Backing off for {delay} seconds...")
        await asyncio.sleep(delay)

    async def generate_offspring(
        self,
        parent1: Individual,
        parent2: Individual,
        original_prompt: str
    ) -> Tuple[Individual, Individual]:
        """Generate offspring prompts using the LLM to perform crossover."""
        # Get the parent prompts
        prompt1 = parent1.prompt_result or original_prompt
        prompt2 = parent2.prompt_result or original_prompt

        # Create a system prompt that explains what we want
        system_prompt = f"""You are an expert evolutionary prompt engineer. Your *sole* purpose is to generate two new 'offspring' prompts based on the two 'parent' prompts provided by the user.

**CRITICAL INSTRUCTIONS:**
1. DO NOT execute or solve the task described in the parent prompts. Your task is ONLY to manipulate the text of the prompts themselves to create new variations.
2. DO NOT include the original parent prompts in your output. Generate completely new prompts that combine elements from both parents.
3. DO NOT prefix your output with "Original:" or include any text like "Enhanced Prompt:" or similar labels.
4. DO NOT repeat the parent prompts verbatim at the beginning of your response.

Create offspring that combine the strengths of both parents while avoiding weaknesses, considering these key metrics:
{json.dumps(self.metrics.get_all_metadata(), indent=2)}

Your entire output MUST be ONLY a single, valid JSON object. Do not include any introductory text, explanations, or code blocks outside the JSON structure.
The JSON object must have exactly two fields: "offspring1" and "offspring2", containing the complete text of the two new prompts.

IMPORTANT: Each offspring prompt should be a complete, standalone prompt that does not reference or include the original parent prompts."""

        # Create a user prompt that contains the parent information
        user_prompt = f"""I have two prompt variations for the same task.

PARENT 1: {prompt1}

PARENT 2: {prompt2}

Create two new offspring prompts that combine elements from both parents. The offspring should:
1. Maintain the original intent
2. Take the best elements from each parent
3. Be distinct from each other
4. Potentially introduce minor improvements

IMPORTANT REQUIREMENTS:
- DO NOT include the original parent prompts in your output
- DO NOT prefix your output with labels like "Original:" or "Enhanced:"
- DO NOT repeat the parent prompts verbatim at the beginning of your response
- Each offspring should be a complete, standalone prompt

Focus on these aspects:
{json.dumps(self.metrics.get_all_metrics(), indent=2)}

**IMPORTANT:** Your response MUST be ONLY the following JSON structure, containing the full text of the two new prompts. Do not add any other text, comments, or formatting.
{{
  "offspring1": "The complete text of the first offspring prompt (without including the original parent prompts)",
  "offspring2": "The complete text of the second offspring prompt (without including the original parent prompts)"
}}"""

        # Try with gradual quality degradation (OpenAI client handles rate limiting)
        retry_count = 0
        max_retries = self.max_retries

        while retry_count < max_retries:
            try:
                # Adjust temperature based on retry count for gradual quality degradation
                current_temperature = min(0.7 + (retry_count * 0.1), 1.0)

                # OpenAI client will handle rate limiting and retries automatically
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=current_temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                # Extract and parse the JSON response using a more robust approach
                result_text = response.choices[0].message.content

                # First try to find JSON using regex
                match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if not match:
                    raise json.JSONDecodeError("No JSON object found in breeding response", result_text, 0)

                json_str = match.group(0)

                # Apply a series of fixes to make the JSON valid
                # 1. Replace single quotes with double quotes for keys and string values
                fixed_json_str = re.sub(r"'([^']*)'(?=\s*:)", r'"\1"', json_str)
                fixed_json_str = re.sub(r':\s*\'([^\']*)\'', r': "\1"', fixed_json_str)

                # 2. Fix any trailing commas in arrays or objects
                fixed_json_str = re.sub(r',\s*}', '}', fixed_json_str)
                fixed_json_str = re.sub(r',\s*]', ']', fixed_json_str)

                # 3. Ensure all property names are in double quotes
                fixed_json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', fixed_json_str)

                try:
                    # Try to parse the fixed JSON
                    result = json.loads(fixed_json_str)
                except json.JSONDecodeError as e:
                    # If that fails, try a more aggressive approach - extract just the offspring values
                    print(f"First JSON parsing attempt failed: {str(e)}")

                    # Try to extract offspring1 and offspring2 directly using regex
                    offspring1_match = re.search(r'"offspring1"\s*:\s*"([^"]*)"', fixed_json_str, re.DOTALL)
                    offspring2_match = re.search(r'"offspring2"\s*:\s*"([^"]*)"', fixed_json_str, re.DOTALL)

                    if offspring1_match and offspring2_match:
                        # Create a simple JSON object with the extracted values
                        result = {
                            "offspring1": offspring1_match.group(1),
                            "offspring2": offspring2_match.group(1)
                        }
                    else:
                        # If we can't extract the values, re-raise the exception
                        raise

                # Create new individuals with the offspring prompts
                child1 = Individual(strategy=self._create_strategy_from_parents(parent1, parent2, "offspring1"))
                child2 = Individual(strategy=self._create_strategy_from_parents(parent2, parent1, "offspring2"))

                # Get the offspring prompts
                offspring1 = result.get("offspring1", "")
                offspring2 = result.get("offspring2", "")

                # Post-process the offspring prompts to ensure they don't contain the original prompts
                offspring1 = self._clean_offspring_prompt(offspring1, prompt1, prompt2)
                offspring2 = self._clean_offspring_prompt(offspring2, prompt1, prompt2)

                # Set the prompt results
                child1.prompt_result = offspring1
                child2.prompt_result = offspring2

                # Set parent IDs and generation
                child1.parent_ids = [parent1.id, parent2.id]
                child2.parent_ids = [parent1.id, parent2.id]
                child1.generation = max(parent1.generation, parent2.generation) + 1
                child2.generation = max(parent1.generation, parent2.generation) + 1

                return child1, child2

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Error in LLM breeding: {str(e)}")
                    print(f"Attempt {retry_count}/{max_retries}. Retrying...")
                    await self._handle_rate_limit(retry_count)
                else:
                    print(f"Error in LLM breeding after {max_retries} attempts: {str(e)}")
                    break

        # If all retries failed, try a simpler breeding approach before falling back
        try:
            # Attempt a final breeding with minimal prompt and higher temperature
            simple_system_prompt = "Create two new prompts by combining elements from the given parent prompts. Do not include the original prompts in your output. Return only JSON."
            simple_user_prompt = f"Parent 1: {prompt1}\n\nParent 2: {prompt2}\n\nReturn only JSON with offspring1 and offspring2. Do not include the original parent prompts in your output."

            # OpenAI client will handle rate limiting automatically
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.9,
                messages=[
                    {"role": "system", "content": simple_system_prompt},
                    {"role": "user", "content": simple_user_prompt}
                ]
            )

            # Extract and parse the JSON response using a more robust approach
            result_text = response.choices[0].message.content

            # First try to find JSON using regex
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not match:
                raise json.JSONDecodeError("No JSON object found in simple breeding response", result_text, 0)

            json_str = match.group(0)

            # Apply a series of fixes to make the JSON valid
            # 1. Replace single quotes with double quotes for keys and string values
            fixed_json_str = re.sub(r"'([^']*)'(?=\s*:)", r'"\1"', json_str)
            fixed_json_str = re.sub(r':\s*\'([^\']*)\'', r': "\1"', fixed_json_str)

            # 2. Fix any trailing commas in arrays or objects
            fixed_json_str = re.sub(r',\s*}', '}', fixed_json_str)
            fixed_json_str = re.sub(r',\s*]', ']', fixed_json_str)

            # 3. Ensure all property names are in double quotes
            fixed_json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', fixed_json_str)

            try:
                # Try to parse the fixed JSON
                result = json.loads(fixed_json_str)
            except json.JSONDecodeError as e:
                # If that fails, try a more aggressive approach - extract just the offspring values
                print(f"Simple breeding JSON parsing attempt failed: {str(e)}")

                # Try to extract offspring1 and offspring2 directly using regex
                offspring1_match = re.search(r'"offspring1"\s*:\s*"([^"]*)"', fixed_json_str, re.DOTALL)
                offspring2_match = re.search(r'"offspring2"\s*:\s*"([^"]*)"', fixed_json_str, re.DOTALL)

                if offspring1_match and offspring2_match:
                    # Create a simple JSON object with the extracted values
                    result = {
                        "offspring1": offspring1_match.group(1),
                        "offspring2": offspring2_match.group(1)
                    }
                else:
                    # If we can't extract the values, re-raise the exception
                    raise

            child1 = Individual(strategy=self._create_strategy_from_parents(parent1, parent2, "offspring1-simple"))
            child2 = Individual(strategy=self._create_strategy_from_parents(parent2, parent1, "offspring2-simple"))

            # Get the offspring prompts
            offspring1 = result.get("offspring1", "")
            offspring2 = result.get("offspring2", "")

            # Post-process the offspring prompts to ensure they don't contain the original prompts
            offspring1 = self._clean_offspring_prompt(offspring1, prompt1, prompt2)
            offspring2 = self._clean_offspring_prompt(offspring2, prompt1, prompt2)

            # Set the prompt results
            child1.prompt_result = offspring1
            child2.prompt_result = offspring2

            child1.parent_ids = [parent1.id, parent2.id]
            child2.parent_ids = [parent1.id, parent2.id]
            child1.generation = max(parent1.generation, parent2.generation) + 1
            child2.generation = max(parent1.generation, parent2.generation) + 1
            return child1, child2

        except Exception as e:
            print(f"Simple breeding attempt failed: {str(e)}")

        # If all attempts failed, fallback to regular crossover
        print("All LLM breeding attempts failed, falling back to regular crossover")
        return Individual.crossover(parent1, parent2)

    async def mutate_prompt(
        self,
        individual: Individual,
        original_prompt: str,
        mutation_strength: float = 0.3
    ) -> Individual:
        """Use the LLM to mutate a prompt with rate limiting and retries."""
        prompt_to_mutate = individual.prompt_result or original_prompt

        # Get metric metadata for focused mutations
        metrics_metadata = self.metrics.get_all_metadata()

        # Create system prompt for mutation
        system_prompt = f"""You are an evolutionary prompt engineer specializing in prompt mutation.
Your task is to introduce variations to a prompt that might improve its effectiveness.
Your mutations should maintain the original intent while making potentially beneficial changes.

**CRITICAL INSTRUCTIONS:**
1. DO NOT include the original prompt in your output.
2. DO NOT prefix your output with "Original:" or include any text like "Enhanced Prompt:" or similar labels.
3. DO NOT repeat the original prompt verbatim at the beginning of your response.
4. Return ONLY the mutated prompt as a complete, standalone prompt.

Consider these metrics when mutating:
{json.dumps(metrics_metadata, indent=2)}"""

        # Create user prompt with mutation instructions
        strength_text = "minor" if mutation_strength < 0.3 else "moderate" if mutation_strength < 0.7 else "significant"
        user_prompt = f"""Mutate the following prompt with {strength_text} changes:

PROMPT: {prompt_to_mutate}

Apply {strength_text} mutations that could improve the prompt while maintaining its core intent.
Focus on these weighted aspects:
{json.dumps(self.metrics.get_all_metrics(), indent=2)}

Some mutation ideas:
- Adjust clarity and specificity
- Enhance technical validity
- Improve context retention
- Introduce innovative elements
- Strengthen intent alignment
- Optimize effectiveness

IMPORTANT REQUIREMENTS:
- DO NOT include the original prompt in your output
- DO NOT prefix your output with labels like "Original:" or "Enhanced:"
- DO NOT repeat the original prompt verbatim at the beginning of your response
- The mutated prompt should be a complete, standalone prompt

Return ONLY the mutated prompt with no explanations or other text."""

        # Try with gradual quality degradation (OpenAI client handles rate limiting)
        retry_count = 0
        max_retries = self.max_retries

        while retry_count < max_retries:
            try:
                # Adjust temperature based on retry count and mutation strength
                current_temperature = min(0.9, 0.3 + mutation_strength + (retry_count * 0.1))

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=current_temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                # Create a new individual with the mutated prompt
                mutated = Individual(strategy=self._create_mutated_strategy(individual.strategy))

                # Get the mutated prompt and clean it
                mutated_prompt = response.choices[0].message.content.strip()
                cleaned_prompt = self._clean_offspring_prompt(mutated_prompt, prompt_to_mutate, "")

                # If cleaning resulted in an empty string, use the original response
                if not cleaned_prompt:
                    print("Mutation cleaning resulted in empty string, using original response")
                    cleaned_prompt = mutated_prompt

                mutated.prompt_result = cleaned_prompt
                mutated.parent_ids = [individual.id]
                mutated.generation = individual.generation + 1

                return mutated

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Error in LLM mutation: {str(e)}")
                    print(f"Attempt {retry_count}/{max_retries}. Retrying...")
                    await self._handle_rate_limit(retry_count)
                else:
                    print(f"Error in LLM mutation after {max_retries} attempts: {str(e)}")
                    break

        # If all retries failed, try a simpler mutation approach before falling back
        try:
            # Attempt a final mutation with minimal prompt and higher temperature
            simple_system_prompt = "Modify the given prompt while maintaining its core intent. Do not include the original prompt in your output."
            simple_user_prompt = f"Original: {prompt_to_mutate}\n\nCreate a modified version. Return only the modified prompt without including the original."

            # OpenAI client will handle rate limiting automatically
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.9,
                messages=[
                    {"role": "system", "content": simple_system_prompt},
                    {"role": "user", "content": simple_user_prompt}
                ]
            )

            mutated = Individual(strategy=self._create_mutated_strategy(individual.strategy))

            # Get the mutated prompt and clean it
            mutated_prompt = response.choices[0].message.content.strip()
            cleaned_prompt = self._clean_offspring_prompt(mutated_prompt, prompt_to_mutate, "")

            # If cleaning resulted in an empty string, use the original response
            if not cleaned_prompt:
                print("Simple mutation cleaning resulted in empty string, using original response")
                cleaned_prompt = mutated_prompt

            mutated.prompt_result = cleaned_prompt
            mutated.parent_ids = [individual.id]
            mutated.generation = individual.generation + 1
            return mutated

        except Exception as e:
            print(f"Simple mutation attempt failed: {str(e)}")

        # If all attempts failed, fallback to regular mutation
        print("All LLM mutation attempts failed, falling back to regular mutation")
        return individual.mutate(mutation_strength)

    async def evaluate_fitness(
        self,
        prompt: str,
        original_prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Use the enhanced evaluation system to evaluate prompt fitness with rate limiting and retries."""
        # Try with gradual quality degradation (OpenAI client handles rate limiting)
        retry_count = 0
        max_retries = self.max_retries

        while retry_count < max_retries:
            try:

                # Create evaluation context with retry-aware settings
                eval_context = {
                    "original_prompt": original_prompt,
                    "intent_analysis": context or {},
                    "generation": 0,  # Default to 0 if not provided
                    "retry_count": retry_count  # Pass retry count for adaptive evaluation
                }

                # Get initial scores with confidence adjustment based on retry count
                base_scores = await self.evaluator.evaluate(prompt, eval_context)
                confidence_factor = max(0.3, 1.0 - (retry_count * 0.15))  # Decrease confidence with retries

                # Apply fallback strategy if needed, with adjusted thresholds
                if not self._is_valid_score(base_scores):
                    fallback_scores = await self.strategy.evaluate_with_fallback(
                        prompt,
                        eval_context
                    )
                    # Blend scores based on retry count
                    base_scores = {
                        k: (v * (1 - retry_count/max_retries) +
                            fallback_scores.get(k, 0.3) * (retry_count/max_retries))
                        for k, v in base_scores.items()
                    }

                # Cross-validate scores
                validation_result = await self.validator.validate_fitness_scores(
                    prompt,
                    eval_context,
                    [original_prompt]  # history for innovation check
                )

                # Adjust validation confidence based on retry count
                validation_result.confidence = max(0.3, validation_result.confidence * (1.0 - (retry_count * 0.15)))

                # Combine and normalize scores with adjusted confidence
                final_scores = self._combine_scores(base_scores, validation_result)
                final_scores = {k: max(0.3, v * confidence_factor) for k, v in final_scores.items()}

                return final_scores

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Error in LLM fitness evaluation: {str(e)}")
                    print(f"Attempt {retry_count}/{max_retries}. Retrying...")
                    await self._handle_rate_limit(retry_count)
                else:
                    print(f"Error in LLM fitness evaluation after {max_retries} attempts: {str(e)}")
                    break

        # If all retries failed, try a simplified evaluation before using conservative scores
        try:
            # Attempt a final evaluation with minimal context
            # OpenAI client will handle rate limiting automatically
            simple_context = {"original_prompt": original_prompt}
            simple_scores = await self.evaluator.evaluate(prompt, simple_context)
            if self._is_valid_score(simple_scores):
                # Apply heavy confidence penalty but keep some signal
                return {k: max(0.3, v * 0.5) for k, v in simple_scores.items()}
        except Exception as e:
            print(f"Simple evaluation attempt failed: {str(e)}")

        # If everything failed, return conservative scores
        print("All LLM fitness evaluation attempts failed, using conservative scores")
        conservative_scores = {metric: 0.3 for metric in self.metrics.get_all_metrics()}
        conservative_scores['overall'] = 0.3
        return conservative_scores

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

                # Apply confidence weighting and ensure minimum score
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

    def _create_strategy_from_parents(
        self,
        parent1: Individual,
        parent2: Individual,
        name_suffix: str
    ) -> EnhancementStrategy:
        """Create a new strategy based on two parent strategies."""
        # Take aspects from both parents
        p1 = parent1.strategy
        p2 = parent2.strategy

        new_strategy = EnhancementStrategy(
            name=f"LLM-Bred-{name_suffix}",
            temperature=(p1.temperature + p2.temperature) / 2,  # Average temperature
            chain_of_thought=p1.chain_of_thought,  # Take from first parent
            semantic_check=max(p1.semantic_check, p2.semantic_check),  # Take the more conservative approach
            context_preservation=max(p1.context_preservation, p2.context_preservation),  # Take the more conservative approach
            system_prompt=self._combine_system_prompts(p1.system_prompt, p2.system_prompt),
            max_tokens=p1.max_tokens if p1.max_tokens is not None else p2.max_tokens,
            top_p=p1.top_p if p1.top_p is not None else p2.top_p,
            frequency_penalty=p1.frequency_penalty if p1.frequency_penalty is not None else p2.frequency_penalty,
            presence_penalty=p1.presence_penalty if p1.presence_penalty is not None else p2.presence_penalty
        )

        return new_strategy

    def _create_mutated_strategy(self, strategy: EnhancementStrategy) -> EnhancementStrategy:
        """Create a mutated version of a strategy."""
        import random

        new_strategy = EnhancementStrategy(
            name=f"LLM-Mutated-{strategy.name}",
            temperature=max(0.1, min(1.0, strategy.temperature + random.uniform(-0.2, 0.2))),
            chain_of_thought=not strategy.chain_of_thought if random.random() < 0.2 else strategy.chain_of_thought,
            semantic_check=strategy.semantic_check,  # Keep this the same for safety
            context_preservation=strategy.context_preservation,  # Keep this the same for safety
            system_prompt=strategy.system_prompt,  # Keep the system prompt (we're mutating the actual prompt)
            max_tokens=strategy.max_tokens,
            top_p=strategy.top_p,
            frequency_penalty=strategy.frequency_penalty,
            presence_penalty=strategy.presence_penalty
        )

        return new_strategy

    def _clean_offspring_prompt(self, offspring: str, parent1: str, parent2: str) -> str:
        """
        Clean an offspring prompt to ensure it's a valid enhancement of the parent prompts.

        Args:
            offspring: The offspring prompt to clean
            parent1: The first parent prompt
            parent2: The second parent prompt

        Returns:
            The cleaned offspring prompt
        """
        if not offspring or not offspring.strip():
            return offspring

        # Strip the offspring prompt
        cleaned_offspring = offspring.strip()

        # Remove markdown code fences
        cleaned_offspring = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned_offspring)
        cleaned_offspring = re.sub(r'\n?```$', '', cleaned_offspring)
        cleaned_offspring = cleaned_offspring.strip()

        # Remove common prefixes
        prefix_pattern = r'^(?:enhanced|adjusted|revised|mutated|original|improved|new|modified|output|result|final)\s*(?:prompt|version|text)?\s*[:\-]?\s*'
        cleaned_offspring = re.sub(prefix_pattern, '', cleaned_offspring, flags=re.IGNORECASE).strip()

        # Check if the offspring contains either parent prompt
        parent1_stripped = parent1.strip() if parent1 else ""
        parent2_stripped = parent2.strip() if parent2 else ""

        # Check if the LLM has enhanced a parent prompt by building upon it
        # This is a common case where the LLM keeps the parent prompt and adds to it
        if parent1_stripped and cleaned_offspring.startswith(parent1_stripped):
            # Get the part after the parent prompt
            additional_content = cleaned_offspring[len(parent1_stripped):].strip()

            # If there's substantial additional content, keep the entire enhanced prompt
            if len(additional_content) > 10:
                return cleaned_offspring

        if parent2_stripped and cleaned_offspring.startswith(parent2_stripped):
            # Get the part after the parent prompt
            additional_content = cleaned_offspring[len(parent2_stripped):].strip()

            # If there's substantial additional content, keep the entire enhanced prompt
            if len(additional_content) > 10:
                return cleaned_offspring

        # Check for the "Original: ... Enhanced: ..." format
        for pattern in [
            r'(?:original|input|parent\s*\d*)[:\-]?\s*(.*?)(?:enhanced|improved|output|offspring)[:\-]?\s*(.*)',
            r'(?:original|input|parent\s*\d*)[:\-]?\s*(.*?)\n+(?:enhanced|improved|output|offspring)[:\-]?\s*(.*)'
        ]:
            match = re.search(pattern, cleaned_offspring, re.IGNORECASE | re.DOTALL)
            if match and match.group(2) and len(match.group(2).strip()) > 20:
                # If the original part matches either parent, use the enhanced part
                if ((parent1_stripped and self._text_similarity(match.group(1).strip(), parent1_stripped) > 0.7) or
                    (parent2_stripped and self._text_similarity(match.group(1).strip(), parent2_stripped) > 0.7)):
                    return match.group(2).strip()

        # Remove metadata and instructions that might have been added by the LLM
        # This handles cases where the LLM adds things like "Domain: Testing" or "Style: technical"
        metadata_pattern = r'(?:domain|style|context|keywords|constraints|format):\s*[^\n]+\n*'
        cleaned_offspring = re.sub(metadata_pattern, '', cleaned_offspring, flags=re.IGNORECASE)

        # Remove instructions like "Ensure the enhanced version maintains..."
        instruction_pattern = r'(?:ensure|make sure|verify)[^.]+(?:original|context|domain|intent)[^.]+\.'
        cleaned_offspring = re.sub(instruction_pattern, '', cleaned_offspring, flags=re.IGNORECASE)

        # Clean up any resulting double newlines and extra whitespace
        cleaned_offspring = re.sub(r'\n\s*\n', '\n', cleaned_offspring)
        cleaned_offspring = cleaned_offspring.strip()

        # Check if the text starts with a fragment (like a numbered list continuation)
        fragment_start_pattern = r'^(?:\d+\)|\d+\.|\-|\*|\,|\;)\s*'
        if re.match(fragment_start_pattern, cleaned_offspring):
            # If it starts with a fragment, it's likely not a complete prompt
            # Try to find a complete sentence in the original text
            sentences = re.split(r'(?<=[.!?])\s+', offspring.strip())
            if len(sentences) > 1:
                # Use the first complete sentence that's substantial
                for sentence in sentences:
                    if len(sentence.strip()) > 30 and not re.match(fragment_start_pattern, sentence.strip()):
                        return sentence.strip()

            # If we couldn't find a good sentence, use the most appropriate parent prompt as a base
            if parent1_stripped and parent2_stripped:
                # Choose the parent that's most similar to the fragment
                if self._text_similarity(cleaned_offspring, parent1_stripped) > self._text_similarity(cleaned_offspring, parent2_stripped):
                    base_prompt = parent1_stripped
                else:
                    base_prompt = parent2_stripped
            else:
                base_prompt = parent1_stripped or parent2_stripped or "Write a"

            if len(cleaned_offspring) > 20:
                return f"{base_prompt} that includes {cleaned_offspring}"
            else:
                return base_prompt

        # If the offspring is very short, it might be a fragment
        if len(cleaned_offspring) < 30:
            # If it's too short, it's likely not a complete prompt
            if len(offspring.strip()) > 50:
                # Try to extract a complete sentence from the original text
                sentences = re.split(r'(?<=[.!?])\s+', offspring.strip())
                if len(sentences) > 1:
                    # Use the first complete sentence that's substantial
                    for sentence in sentences:
                        if len(sentence.strip()) > 30:
                            return sentence.strip()

                # If we couldn't find a good sentence, return the original text
                return offspring.strip()
            else:
                # If the original text is also short, use a parent prompt as a base
                if parent1_stripped and parent2_stripped:
                    # Choose the parent that's most similar to the fragment
                    if self._text_similarity(cleaned_offspring, parent1_stripped) > self._text_similarity(cleaned_offspring, parent2_stripped):
                        base_prompt = parent1_stripped
                    else:
                        base_prompt = parent2_stripped
                else:
                    base_prompt = parent1_stripped or parent2_stripped or "Write a"

                return f"{base_prompt} that {cleaned_offspring}"

        # If the offspring is substantially different from both parents and long enough,
        # it might be a completely rewritten prompt
        if len(cleaned_offspring) > 50:
            if ((not parent1_stripped or self._text_similarity(cleaned_offspring, parent1_stripped) < 0.7) and
                (not parent2_stripped or self._text_similarity(cleaned_offspring, parent2_stripped) < 0.7)):
                return cleaned_offspring

        # If none of the above conditions are met, return the original offspring
        return offspring.strip()

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

    def _combine_system_prompts(self, prompt1: str, prompt2: str) -> str:
        """Combine two system prompts intelligently."""
        if prompt1 == prompt2:
            return prompt1

        # Very simple combination for now - in a real system would use more sophisticated NLP
        return f"You are a prompt enhancement assistant. {prompt1.split('.')[0]}. {prompt2.split('.')[0]}."