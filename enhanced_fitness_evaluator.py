import asyncio
import json
import re
from typing import Dict, List, Optional, Union, TypeVar, Callable, Any
from unified_fitness_metrics import UnifiedFitnessMetrics
from config import default_config
from openai import OpenAI

T = TypeVar('T')  # Type variable for generic return type


class InvalidScoreError(Exception):
    """Custom exception for invalid score parsing."""
    pass

class EnhancedFitnessEvaluator:
    """
    A class that implements an enhanced evaluation pipeline for prompt fitness calculation.
    Provides primary LLM evaluation with rule-based backup metrics.
    """

    def __init__(self, client: OpenAI):
        """Initialize the evaluator with required OpenAI client."""
        if not client:
            raise ValueError("OpenAI client is required")

        self.client = client
        self.model_name = default_config.model_name
        self.metrics = UnifiedFitnessMetrics()
        self.evaluation_results: Dict[str, List[float]] = {}

    def _parse_score(self, content: str) -> float:
        """Parse the score from LLM JSON response, extracting JSON even if extra text exists."""
        try:
            # Use regex to find the first JSON object in the content
            match = re.search(r'\{.*?\}', content, re.DOTALL)
            if not match:
                raise json.JSONDecodeError("No JSON object found in response", content, 0)

            json_str = match.group(0)

            # Attempt to parse the extracted JSON string
            data = json.loads(json_str)

            # Extract the score
            score = data['score']

            # Convert score to float and validate range
            # Removed isinstance check to allow string scores like "0.83"
            score = float(score) # Attempt conversion

            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Score out of range (0.0-1.0): {score}")


            # Return the valid score, clamped to a minimum of 0.3 as per original logic
            return max(0.3, score)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {str(e)}. Response: '{content}'")
            return 0.3 # Default score on JSON error
        except KeyError:
            print(f"Error: 'score' key not found in JSON response: '{json_str if 'json_str' in locals() else content}'")
            return 0.3 # Default score if key is missing
        except (ValueError, TypeError) as e:
            print(f"Error validating score: {str(e)}. Response: '{json_str if 'json_str' in locals() else content}'")
            return 0.3 # Default score on validation error

    async def _llm_evaluation(self, prompt: str, context: Dict, max_retries: int = 3) -> Dict[str, float]:
        """
        Primary evaluation using LLM-based scoring with retry mechanism.
        Returns scores for each metric defined in UnifiedFitnessMetrics.

        :param prompt: The prompt to evaluate
        :param context: Context dictionary containing intent analysis
        :param max_retries: Maximum number of retries for each LLM call
        :return: Dictionary of metric scores
        """
        def _safe_get_score(evaluation_func):
            """
            Wrapper to retry LLM scoring with error handling

            :param evaluation_func: Function that performs LLM call and scoring
            :return: Parsed score or default score on failure
            """
            for attempt in range(max_retries):
                try:
                    response = evaluation_func()
                    score = self._parse_score(response.choices[0].message.content)

                    # Validate score is between 0.0 and 1.0
                    if 0.0 <= score <= 1.0:
                        return score

                    # If score is invalid, continue to retry
                    raise ValueError(f"Invalid score: {score}")

                except (ValueError, Exception) as e:
                    print(f"Scoring attempt {attempt + 1} failed: {str(e)}")

                    # If this was the last retry, return a conservative default
                    if attempt == max_retries - 1:
                        return 0.3

            # Fallback return (should not normally be reached)
            return 0.3

        try:
            # Extract intent analysis from context
            intent_analysis = context.get('intent_analysis', {})

            # Scoring functions for each metric
            def clarity_eval():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a prompt clarity evaluator. Score the clarity of the given prompt on a scale from 0.00 to 1.00. Consider:\n- Clear and unambiguous language\n- Well-structured sentences\n- Logical flow of ideas\n- Absence of confusing statements\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )

            def specificity_eval():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a prompt specificity evaluator. Score the specificity of the given prompt on a scale from 0.00 to 1.00. Consider:\n- Precise requirements\n- Detailed expectations\n- Concrete examples\n- Measurable outcomes\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )

            def technical_eval():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a technical validity evaluator. Score the technical soundness of the given prompt on a scale from 0.00 to 1.00. Consider:\n- Correct terminology\n- Feasible requirements\n- Logical constraints\n- Technical best practices\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )

            def context_retention_eval():
                context_info = f"Keywords: {', '.join(intent_analysis.get('keywords', []))}\nContext: {intent_analysis.get('description', '')}"
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a context retention evaluator. Score how well the prompt maintains the provided context on a scale from 0.00 to 1.00. Consider:\n- Inclusion of key elements\n- Appropriate use of context\n- Maintenance of details\n- Relevance to context\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text."},
                        {"role": "user", "content": f"Context:\n{context_info}\n\nPrompt:\n{prompt}"}
                    ],
                    temperature=0.0
                )

            def effectiveness_eval():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an effectiveness evaluator. Score the overall effectiveness of the given prompt on a scale from 0.00 to 1.00. Consider:\n- Likelihood of success\n- Balance of precision and flexibility\n- Practical applicability\n- Overall quality\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )

            def innovation_eval():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an innovation evaluator. Score the creativity and novelty of the given prompt on a scale from 0.00 to 1.00. Consider:\n- Novel approaches\n- Creative problem-solving\n- Unique combinations\n- Innovative language\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )

            def intent_alignment_eval():
                intent_info = f"Goals: {intent_analysis.get('goals', '')}\nIntent: {intent_analysis.get('intent', '')}"
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an intent alignment evaluator. Score how well the prompt aligns with the goals on a scale from 0.00 to 1.00. Consider:\n- Alignment with goals\n- Purpose fulfillment\n- Requirement adherence\n- Outcome achievement\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text."},
                        {"role": "user", "content": f"Intent Information:\n{intent_info}\n\nPrompt:\n{prompt}"}
                    ],
                    temperature=0.0
                )

            # Define async wrapper for _safe_get_score
            async def async_safe_get_score(evaluation_func):
                return _safe_get_score(evaluation_func)

            # Run all evaluations concurrently
            clarity_task = asyncio.create_task(async_safe_get_score(clarity_eval))
            specificity_task = asyncio.create_task(async_safe_get_score(specificity_eval))
            technical_task = asyncio.create_task(async_safe_get_score(technical_eval))
            context_task = asyncio.create_task(async_safe_get_score(context_retention_eval))
            effectiveness_task = asyncio.create_task(async_safe_get_score(effectiveness_eval))
            innovation_task = asyncio.create_task(async_safe_get_score(innovation_eval))
            intent_task = asyncio.create_task(async_safe_get_score(intent_alignment_eval))

            # Wait for all tasks to complete
            await asyncio.gather(
                clarity_task, specificity_task, technical_task, context_task,
                effectiveness_task, innovation_task, intent_task
            )

            # Return results
            return {
                'clarity': clarity_task.result(),
                'specificity': specificity_task.result(),
                'technical_validity': technical_task.result(),
                'context_retention': context_task.result(),
                'effectiveness': effectiveness_task.result(),
                'innovation': innovation_task.result(),
                'intent_alignment': intent_task.result()
            }

        except Exception as e:
            print(f"LLM evaluation error: {str(e)}")
            # Return conservative scores on overall error
            return {metric: 0.3 for metric in self.metrics.get_all_metrics()}

    def _rule_based_evaluation(self, prompt: str, context: Dict) -> Dict[str, float]:
        """
        Backup evaluation using rule-based metrics.
        """
        scores = {}

        # Clarity score based on sentence structure and length
        words = prompt.split()
        sentences = max(1, prompt.count('.') + prompt.count('!') + prompt.count('?'))
        avg_words_per_sentence = len(words) / sentences
        clarity_score = min(1.0, 2.0 / (1.0 + 0.1 * abs(avg_words_per_sentence - 15)))

        # Specificity score based on presence of specific details
        detail_keywords = ['specifically', 'exactly', 'precisely', 'must', 'required']
        specificity_score = min(1.0, sum(word.lower() in prompt.lower()
                                       for word in detail_keywords) / 3.0)

        # Technical validity score based on structure
        has_context = 'context' in prompt.lower() or 'background' in prompt.lower()
        has_requirements = 'require' in prompt.lower() or 'need' in prompt.lower()
        has_constraints = 'limit' in prompt.lower() or 'constraint' in prompt.lower()
        technical_score = (has_context + has_requirements + has_constraints) / 3.0

        # Context retention score
        context_keywords = context.get('keywords', [])
        retained_context = sum(keyword.lower() in prompt.lower()
                             for keyword in context_keywords)
        context_score = min(1.0, retained_context / max(1, len(context_keywords)))

        scores = {
            'clarity': clarity_score,
            'specificity': specificity_score,
            'technical_validity': technical_score,
            'context_retention': context_score,
            # Default scores for metrics without rule-based implementation
            'intent_alignment': 0.5,
            'effectiveness': 0.5,
            'innovation': 0.5
        }

        return scores

    def _aggregate_scores(self,
                          llm_scores: Dict[str, float],
                          rule_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate scores from different evaluation methods using weighted averaging.
        Uses a weighted combination favoring the higher score to prevent undervaluation.
        """
        aggregated_scores = {}
        weights = self.metrics.get_all_metrics()

        for metric in weights:
            # Get scores with minimum fallback of 0.1
            llm_score = llm_scores.get(metric, 0.1)
            rule_score = rule_scores.get(metric, 0.1)

            # Use weighted average favoring the higher score
            max_score = max(llm_score, rule_score)
            min_score = min(llm_score, rule_score)
            aggregated_scores[metric] = (max_score * 0.7) + (min_score * 0.3)

            # Ensure minimum score of 0.1
            aggregated_scores[metric] = max(0.1, aggregated_scores[metric])

        return aggregated_scores

    async def evaluate(self, prompt: str, context: Dict) -> Dict[str, float]:
        """
        Main evaluation method that orchestrates the evaluation pipeline based on config.
        Returns a dictionary of scores for each metric including an overall weighted score.
        """
        llm_scores = {}
        rule_scores = {}
        final_scores = {} # For combined

        # Conditionally get scores based on evaluation type
        if default_config.evaluation_type in ["llm", "combined"]:
            llm_scores = await self._llm_evaluation(prompt, context)
        if default_config.evaluation_type in ["rule", "combined"]:
            rule_scores = self._rule_based_evaluation(prompt, context)

        # Aggregate scores based on evaluation type
        if default_config.evaluation_type == "combined":
            final_scores = self._aggregate_scores(llm_scores, rule_scores)
        elif default_config.evaluation_type == "llm":
            final_scores = llm_scores
        else:  # "rule"
            final_scores = rule_scores

        # Calculate weighted overall score using metric weights from UnifiedFitnessMetrics
        weights = self.metrics.get_all_metrics()
        weighted_sum = 0.0
        total_weight = 0.0

        # Calculate weighted sum based on final_scores
        for metric, weight in weights.items():
            if metric in final_scores:
                weighted_sum += final_scores[metric] * weight
                total_weight += weight

        # Calculate overall score, ensuring it's never zero
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        else:
            # Fallback to simple average if weights are missing
            overall_score = sum(final_scores.values()) / len(final_scores) if final_scores else 0.3

        # Ensure minimum overall score and add to the final scores
        final_scores['overall'] = max(0.3, min(1.0, overall_score))

        # Store results for potential historical analysis
        prompt_hash = hash(prompt)
        if prompt_hash not in self.evaluation_results:
            self.evaluation_results[prompt_hash] = []
        self.evaluation_results[prompt_hash].append(final_scores)

        return final_scores

    def get_evaluation_history(self, prompt: str) -> List[Dict[str, float]]:
        """
        Retrieve evaluation history for a specific prompt.
        """
        prompt_hash = hash(prompt)
        return self.evaluation_results.get(prompt_hash, [])