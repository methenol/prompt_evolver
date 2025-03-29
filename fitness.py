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
        cleaned_text = text.strip()
        original_prompt_stripped = original_prompt.strip()

        # 1. Remove markdown code fences (optional language specifier)
        cleaned_text = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned_text)
        cleaned_text = re.sub(r'\n?```$', '', cleaned_text)
        cleaned_text = cleaned_text.strip() # Strip again after removing fences

        # 2. Remove common prefixes (case-insensitive, optional colon/whitespace)
        # Covers "Enhanced Prompt:", "Adjusted Prompt:", "Revised Prompt:", "Mutated Prompt:", "Original:", etc.
        cleaned_text = re.sub(r'^(?:enhanced|adjusted|revised|mutated|original)\s*prompt\s*[:\-]?\s*', '', cleaned_text, flags=re.IGNORECASE).strip()

        # 3. Attempt to remove the original prompt if it's prepended
        # Be careful not to remove it if the LLM *only* returned the original
        if cleaned_text.startswith(original_prompt_stripped):
            potential_mutation = cleaned_text[len(original_prompt_stripped):].strip()
            # Only remove the original if there's something substantial after it
            if len(potential_mutation) > 10: # Heuristic: require at least 10 chars difference
                 cleaned_text = potential_mutation

        # 4. Remove simple "Original:" prefix if step 2/3 missed it
        cleaned_text = re.sub(r'^Original\s*[:\-]?\s*', '', cleaned_text, flags=re.IGNORECASE).strip()

        return cleaned_text
        
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
        
        # Run evaluations concurrently
        if tasks:
            await asyncio.gather(*tasks)
            
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
        """Ensure context is preserved during enhancement."""
        try:
            response = self.client.chat.completions.create(
                model=default_config.model_name,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": "You are a context preservation assistant. Ensure the enhanced prompt maintains its original context and domain-specific elements."},
                    {"role": "user", "content": f"""
                    Verify and adjust the prompt to preserve context:
                    Original: {original}
                    Current: {current}
                    Domain: {intent_analysis.get('domain', 'general')}
                    Style: {intent_analysis.get('style_tone', {}).get('style', 'neutral')}
                    
                    Ensure the enhanced version maintains the original context and domain-specific elements.
                    Return only the adjusted prompt.
                    """}
                ]
            )
            raw_preserved = response.choices[0].message.content.strip()
            # Use 'original' prompt context for cleaning here
            return self._clean_llm_output(raw_preserved, original)
        except Exception as e:
            print(f"Error in context preservation: {str(e)}")
            return current