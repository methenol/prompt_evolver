from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import asyncio
import time
from enum import Enum
from openai import OpenAI
from unified_fitness_metrics import UnifiedFitnessMetrics
from enhanced_fitness_evaluator import EnhancedFitnessEvaluator  # Import EnhancedFitnessEvaluator

class EvaluationMethod(Enum):
    PRIMARY_LLM = "primary_llm"
    SECONDARY_LLM = "secondary_llm"
    RULE_BASED = "rule_based"
    HISTORICAL = "historical"

@dataclass
class EvaluationResult:
    scores: Dict[str, float]
    method: EvaluationMethod
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class FitnessEvaluationStrategy:
    """
    Implements a robust error handling strategy for fitness evaluation, using ONLY LLM.
    No fallback to rule-based methods.
    """

    def __init__(self, client: OpenAI):
        self.metrics = UnifiedFitnessMetrics()
        self.retry_delays = [1, 2, 5]  # Seconds between retries
        self.rate_limit_window = 60  # 1 minute window
        self.max_requests = 50  # Maximum requests per window
        self.request_timestamps: List[float] = []
        self.client = client

    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        # Remove timestamps older than our window
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if current_time - ts < self.rate_limit_window]
        
        if len(self.request_timestamps) >= self.max_requests:
            return False
        
        self.request_timestamps.append(current_time)
        return True

    async def evaluate_with_fallback(self, prompt: str, context: Dict) -> Dict[str, float]:
        """
        Main evaluation method, using ONLY EnhancedFitnessEvaluator for LLM evaluation.
        Retries LLM evaluation in case of errors, but no fallback to rule-based methods.
        """
        evaluator = EnhancedFitnessEvaluator(client=self.client)  # Client is passed in Evolution class

        for retry_delay in self.retry_delays:
            try:
                # Directly use EnhancedFitnessEvaluator for LLM evaluation
                scores = await evaluator.evaluate(prompt, context)
                return scores

            except Exception as e:
                # Log the error and continue to next retry
                print(f"LLM Evaluation attempt failed: {str(e)}")
                await asyncio.sleep(retry_delay)

        # If all retries failed, raise an error since LLM-based evaluations failed.
        raise RuntimeError("All LLM evaluation attempts failed")

    async def evaluate_population_batch(self, prompts: List[str], contexts: List[Dict]) -> List[Dict[str, float]]:
         """
         Evaluate a batch of prompts using the fallback strategy.
         """
         batch_scores = []
         for prompt, context in zip(prompts, contexts):
             try:
                 scores = await self.evaluate_with_fallback(prompt, context)
                 batch_scores.append(scores)
             except Exception as e:
                 print(f"Batch evaluation failed for prompt: {prompt}. Error: {e}")
                 # Handle error appropriately, e.g., return default scores or re-raise
                 default_scores = {metric: 0.1 for metric in self.metrics.get_all_metrics()} # Minimum score
                 default_scores['overall'] = 0.1
                 batch_scores.append(default_scores)
         return batch_scores