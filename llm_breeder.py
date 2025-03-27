"""
LLM Breeder module for using LLMs to perform breeding operations on prompts.
"""

import json
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time

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
        
        # Rate limiting configuration
        self.rate_limit_window = 60  # 1 minute window
        self.max_requests = 50  # Maximum requests per window
        self.max_tokens_per_window = 40000  # Token limit per minute
        self.request_history: List[Tuple[float, int]] = []  # [(timestamp, tokens)]
        self.base_retry_delay = 3  # Base delay for exponential backoff (starts at 3 seconds)
        self.max_retries = 5  # Maximum number of retry attempts
        
    async def _check_rate_limit(self, estimated_tokens: int = 1000) -> bool:
        """
        Check if we're within rate limits, considering both request count and token usage.
        Args:
            estimated_tokens: Estimated tokens for this request (default 1000 for safety)
        Returns:
            bool: True if within limits, False if limits exceeded
        """
        current_time = time.time()
        
        # Remove entries older than our window
        self.request_history = [(ts, tokens) for ts, tokens in self.request_history
                              if current_time - ts < self.rate_limit_window]
        
        # Calculate current usage
        request_count = len(self.request_history)
        token_count = sum(tokens for _, tokens in self.request_history)
        
        # Check both request and token limits
        if (request_count >= self.max_requests or
            token_count + estimated_tokens > self.max_tokens_per_window):
            return False
        
        # Record this request
        self.request_history.append((current_time, estimated_tokens))
        return True
    
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
        system_prompt = f"""You are an evolutionary prompt engineer. Your task is to create two new prompt offspring
based on two parent prompts. Create offspring that combine the strengths of both parents while avoiding weaknesses.

Consider these key metrics when breeding:
{json.dumps(self.metrics.get_all_metadata(), indent=2)}

Output must be formatted as valid JSON with two fields: "offspring1" and "offspring2" containing the two new prompts."""
        
        # Create a user prompt that contains the parent information
        user_prompt = f"""I have two prompt variations for the same task.

PARENT 1: {prompt1}

PARENT 2: {prompt2}

Create two new offspring prompts that combine elements from both parents. The offspring should:
1. Maintain the original intent
2. Take the best elements from each parent
3. Be distinct from each other
4. Potentially introduce minor improvements

Focus on these aspects:
{json.dumps(self.metrics.get_all_metrics(), indent=2)}

Return ONLY valid JSON with this format:
{{
  "offspring1": "The complete text of the first offspring prompt",
  "offspring2": "The complete text of the second offspring prompt"
}}"""

        # Try with exponential backoff and gradual quality degradation
        retry_count = 0
        max_retries = self.max_retries
        base_delay = self.base_retry_delay
        estimated_tokens = 2000  # Conservative estimate for breeding operation
        
        while retry_count < max_retries:
            try:
                # Check rate limit with token estimation
                if not await self._check_rate_limit(estimated_tokens):
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** retry_count)
                    print(f"Rate limit exceeded. Backing off for {delay} seconds...")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    continue
                
                # Adjust temperature based on retry count for gradual quality degradation
                current_temperature = min(0.7 + (retry_count * 0.1), 1.0)
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=current_temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                # Parse the JSON response
                result_text = response.choices[0].message.content
                result = json.loads(result_text)
                
                # Create new individuals with the offspring prompts
                child1 = Individual(strategy=self._create_strategy_from_parents(parent1, parent2, "offspring1"))
                child2 = Individual(strategy=self._create_strategy_from_parents(parent2, parent1, "offspring2"))
                
                # Set the prompt results
                child1.prompt_result = result.get("offspring1")
                child2.prompt_result = result.get("offspring2")
                
                # Set parent IDs and generation
                child1.parent_ids = [parent1.id, parent2.id]
                child2.parent_ids = [parent1.id, parent2.id]
                child1.generation = max(parent1.generation, parent2.generation) + 1
                child2.generation = max(parent1.generation, parent2.generation) + 1
                
                return child1, child2
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    delay = base_delay * (2 ** retry_count)
                    print(f"Error in LLM breeding: {str(e)}")
                    print(f"Attempt {retry_count}/{max_retries}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Error in LLM breeding after {max_retries} attempts: {str(e)}")
                    break
        
        # If all retries failed, try a simpler breeding approach before falling back
        try:
            # Attempt a final breeding with minimal prompt and higher temperature
            simple_system_prompt = "Create two new prompts by combining elements from the given parent prompts."
            simple_user_prompt = f"Parent 1: {prompt1}\n\nParent 2: {prompt2}\n\nReturn only JSON with offspring1 and offspring2."
            
            if await self._check_rate_limit(1000):  # Lower token estimate for simple prompt
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0.9,
                    messages=[
                        {"role": "system", "content": simple_system_prompt},
                        {"role": "user", "content": simple_user_prompt}
                    ]
                )
                
                result = json.loads(response.choices[0].message.content)
                child1 = Individual(strategy=self._create_strategy_from_parents(parent1, parent2, "offspring1-simple"))
                child2 = Individual(strategy=self._create_strategy_from_parents(parent2, parent1, "offspring2-simple"))
                child1.prompt_result = result.get("offspring1")
                child2.prompt_result = result.get("offspring2")
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

Return ONLY the mutated prompt with no explanations or other text."""

        # Try with exponential backoff and gradual quality degradation
        retry_count = 0
        max_retries = self.max_retries
        base_delay = self.base_retry_delay
        estimated_tokens = 1500  # Conservative estimate for mutation operation
        
        while retry_count < max_retries:
            try:
                # Check rate limit with token estimation
                if not await self._check_rate_limit(estimated_tokens):
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** retry_count)
                    print(f"Rate limit exceeded. Backing off for {delay} seconds...")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    continue
                
                # Adjust temperature and mutation strength based on retry count
                current_temperature = min(0.9, 0.3 + mutation_strength + (retry_count * 0.1))
                current_strength = min(1.0, mutation_strength + (retry_count * 0.15))
                
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
                mutated.prompt_result = response.choices[0].message.content.strip()
                mutated.parent_ids = [individual.id]
                mutated.generation = individual.generation + 1
                
                return mutated
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    delay = base_delay * (2 ** retry_count)
                    print(f"Error in LLM mutation: {str(e)}")
                    print(f"Attempt {retry_count}/{max_retries}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Error in LLM mutation after {max_retries} attempts: {str(e)}")
                    break
        
        # If all retries failed, try a simpler mutation approach before falling back
        try:
            # Attempt a final mutation with minimal prompt and higher temperature
            simple_system_prompt = "Modify the given prompt while maintaining its core intent."
            simple_user_prompt = f"Original: {prompt_to_mutate}\n\nCreate a modified version."
            
            if await self._check_rate_limit(800):  # Lower token estimate for simple prompt
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0.9,
                    messages=[
                        {"role": "system", "content": simple_system_prompt},
                        {"role": "user", "content": simple_user_prompt}
                    ]
                )
                
                mutated = Individual(strategy=self._create_mutated_strategy(individual.strategy))
                mutated.prompt_result = response.choices[0].message.content.strip()
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
        # Try with exponential backoff and gradual quality degradation
        retry_count = 0
        max_retries = self.max_retries
        base_delay = self.base_retry_delay
        estimated_tokens = 1200  # Conservative estimate for evaluation operation
        
        while retry_count < max_retries:
            try:
                # Check rate limit with token estimation
                if not await self._check_rate_limit(estimated_tokens):
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** retry_count)
                    print(f"Rate limit exceeded. Backing off for {delay} seconds...")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    continue

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
                    delay = base_delay * (2 ** retry_count)
                    print(f"Error in LLM fitness evaluation: {str(e)}")
                    print(f"Attempt {retry_count}/{max_retries}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Error in LLM fitness evaluation after {max_retries} attempts: {str(e)}")
                    break
        
        # If all retries failed, try a simplified evaluation before using conservative scores
        try:
            # Attempt a final evaluation with minimal context
            if await self._check_rate_limit(800):  # Lower token estimate for simple evaluation
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
    
    def _combine_system_prompts(self, prompt1: str, prompt2: str) -> str:
        """Combine two system prompts intelligently."""
        if prompt1 == prompt2:
            return prompt1
        
        # Very simple combination for now - in a real system would use more sophisticated NLP
        return f"You are a prompt enhancement assistant. {prompt1.split('.')[0]}. {prompt2.split('.')[0]}."