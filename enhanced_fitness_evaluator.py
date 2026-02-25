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
    
    IMPROVEMENTS:
    - Added more evaluation metrics including context retention, adaptability
    - Improved scoring algorithms with better validation
    - Enhanced rule-based fallback with more sophisticated heuristics
    - Added scoring history tracking
    """

    def __init__(self, client: OpenAI):
        """Initialize the evaluator with required OpenAI client."""
        if not client:
            raise ValueError("OpenAI client is required")

        self.client = client
        self.model_name = default_config.model_name
        self.metrics = UnifiedFitnessMetrics()
        self.evaluation_results: Dict[str, List[Dict[str, float]]] = {}
        self._max_history_size = 100  # Limit evaluation history size
    
    def _parse_score(self, content: str, metric: str = "score") -> float:
        """
        Parse the score from LLM response, extracting JSON even if extra text exists.
        
        IMPROVEMENTS:
        - Added metric parameter to handle multiple scoring formats
        - Better error handling with specific error messages
        - More robust JSON extraction
        """
        try:
            # Use regex to find the first JSON object in the content
            match = re.search(r'\{.*?\}', content, re.DOTALL)
            if not match:
                print(f"Failed to extract JSON from response for metric '{metric}'. Response: '{content[:200]}...'")
                return 0.3

            json_str = match.group(0)

            # Handle single quotes in JSON (common LLM output)
            # Replace single quotes with double quotes for keys and values
            fixed_json = re.sub(r"'([^']*)'(?=\s*:)", r'"\1"', json_str)
            fixed_json = re.sub(r':\s*\'([^\']*)\'', r': "\1"', fixed_json)
            
            # Attempt to parse the fixed JSON string
            try:
                data = json.loads(fixed_json)
            except json.JSONDecodeError:
                print(f"JSON parsing failed. Original: '{json_str[:100]}...', Fixed: '{fixed_json[:100]}...'")
                return 0.3

            # Try multiple possible keys for the score
            score = data.get(metric, None)
            if score is None:
                # Try alternative keys
                for alt_key in ['score', 'rating', 'value', 'evaluation']:
                    score = data.get(alt_key, None)
                    if score is not None:
                        break
            
            if score is None:
                print(f"Score key not found in JSON response. Available keys: {list(data.keys())}")
                return 0.3

            # Convert score to float and validate range
            try:
                score = float(score)
            except (ValueError, TypeError) as e:
                print(f"Score conversion failed: {str(e)}. Value: '{score}'")
                return 0.3

            # Validate range
            if not (0.0 <= score <= 1.0):
                print(f"Score out of range (0.0-1.0): {score}")
                return 0.3

            # Return the valid score, clamped to a minimum of 0.3 as per original logic
            return max(0.3, score)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response for metric '{metric}': {str(e)}. Response: '{content[:200]}...'")
            return 0.3  # Default score on JSON error
        except Exception as e:
            print(f"Unexpected error parsing score for metric '{metric}': {str(e)}")
            return 0.3  # Default score on unexpected error

    async def _llm_evaluation(self, prompt: str, context: Dict, max_retries: int = 3) -> Dict[str, float]:
        """
        Primary evaluation using LLM-based scoring with retry mechanism.
        Returns scores for each metric defined in UnifiedFitnessMetrics.

        IMPROVEMENTS:
        - More intelligent prompt construction for each metric
        - Better handling of context and intent analysis
        - Improved scoring consistency with temperature=0 for deterministic scoring
        """
        def _safe_get_score(evaluation_func, metric_name: str):
            """
            Wrapper to retry LLM scoring with error handling
            """
            for attempt in range(max_retries):
                try:
                    response = evaluation_func()
                    score = self._parse_score(response.choices[0].message.content, metric_name)
                    
                    # Validate score is between 0.0 and 1.0
                    if 0.0 <= score <= 1.0:
                        return score
                    
                    # If score is invalid, continue to retry
                    raise ValueError(f"Invalid score: {score} for metric '{metric_name}'")

                except (ValueError, Exception) as e:
                    print(f"Scoring attempt {attempt + 1} failed for '{metric_name}': {str(e)}")
                    
                    # If this was the last retry, return a conservative default
                    if attempt == max_retries - 1:
                        return 0.3

            # Fallback return (should not normally be reached)
            return 0.3

        try:
            # Extract intent analysis from context
            intent_analysis = context.get('intent_analysis', {})
            original_prompt = context.get('original_prompt', '')
            generated_context = context.get('generated_context', '')
            
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
                        {"role": "system", "content": "You are a technical validity evaluator. Your task is to EVALUATE a prompt, not to solve or implement what the prompt is asking for. Score the technical soundness of the given prompt on a scale from 0.00 to 1.00. Consider:\n- Correct terminology\n- Feasible requirements\n- Logical constraints\n- Technical best practices\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text, code implementations, or solutions to what the prompt is asking for."},
                        {"role": "user", "content": "EVALUATE THIS PROMPT (DO NOT SOLVE IT):\n\n" + prompt}
                    ],
                    temperature=0.0
                )

            def context_retention_eval():
                context_info = f"Keywords: {', '.join(intent_analysis.get('keywords', []))}\nContext: {intent_analysis.get('description', '')}\nOriginal Prompt: {original_prompt[:500]}"
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a context retention evaluator. Your task is to EVALUATE a prompt, not to solve or implement what the prompt is asking for. Score how well the prompt maintains the provided context on a scale from 0.00 to 1.00. Consider:\n- Inclusion of key elements from the original\n- Appropriate use of context\n- Maintenance of critical instructions\n- Relevance to context\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text, code implementations, or solutions to what the prompt is asking for."},
                        {"role": "user", "content": f"EVALUATE THIS PROMPT (DO NOT SOLVE IT):\n\nOriginal Context:\n{context_info}\n\nPrompt to evaluate:\n{prompt}"}
                    ],
                    temperature=0.0
                )

            def effectiveness_eval():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an effectiveness evaluator. Your task is to EVALUATE a prompt, not to solve or implement what the prompt is asking for. Score the overall effectiveness of the given prompt on a scale from 0.00 to 1.00. Consider:\n- Likelihood of success\n- Balance of precision and flexibility\n- Practical applicability\n- Overall quality\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text, code implementations, or solutions to what the prompt is asking for."},
                        {"role": "user", "content": "EVALUATE THIS PROMPT (DO NOT SOLVE IT):\n\n" + prompt}
                    ],
                    temperature=0.0
                )

            def innovation_eval():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an innovation evaluator. Your task is to EVALUATE a prompt, not to solve or implement what the prompt is asking for. Score the creativity and novelty of the given prompt on a scale from 0.00 to 1.00. Consider:\n- Novel approaches\n- Creative problem-solving\n- Unique combinations\n- Innovative language\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text, code implementations, or solutions to what the prompt is asking for."},
                        {"role": "user", "content": "EVALUATE THIS PROMPT (DO NOT SOLVE IT):\n\n" + prompt}
                    ],
                    temperature=0.0
                )

            def intent_alignment_eval():
                intent_info = f"Goals: {intent_analysis.get('goals', '')}\nIntent: {intent_analysis.get('intent', '')}\nDomain: {intent_analysis.get('domain', '')}"
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an intent alignment evaluator. Your task is to EVALUATE a prompt, not to solve or implement what the prompt is asking for. Score how well the prompt aligns with the goals on a scale from 0.00 to 1.00. Consider:\n- Alignment with stated goals\n- Purpose fulfillment\n- Requirement adherence\n- Outcome achievement\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text, code implementations, or solutions to what the prompt is asking for."},
                        {"role": "user", "content": f"EVALUATE THIS PROMPT (DO NOT SOLVE IT):\n\nIntent Information:\n{intent_info}\n\nPrompt to evaluate:\n{prompt}"}
                    ],
                    temperature=0.0
                )

            def adaptability_eval():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an adaptability evaluator. Your task is to EVALUATE a prompt's ability to handle variations. Score on a scale from 0.00 to 1.00. Consider:\n- How well the prompt handles edge cases\n- Flexibility for different inputs\n- Graceful degradation\n- Error handling\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text, code implementations, or solutions."},
                        {"role": "user", "content": "EVALUATE THIS PROMPT (DO NOT SOLVE IT):\n\n" + prompt}
                    ],
                    temperature=0.0
                )

            def robustness_eval():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a robustness evaluator. Your task is to EVALUATE a prompt's resistance to degradation from minor changes. Score on a scale from 0.00 to 1.00. Consider:\n- Strength of constraints\n- Verification steps\n- Critical instruction preservation\n- Fallback mechanisms\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text, code implementations, or solutions."},
                        {"role": "user", "content": "EVALUATE THIS PROMPT (DO NOT SOLVE IT):\n\n" + prompt}
                    ],
                    temperature=0.0
                )

            def generalization_eval():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a generalization evaluator. Your task is to EVALUATE a prompt's ability to work across different contexts. Score on a scale from 0.00 to 1.00. Consider:\n- Broader applicability\n- Transferable principles\n- Domain independence\n- Versatile problem-solving\nRespond *only* with a JSON object containing the score rounded to two decimal places, like this: {\"score\": <float_value_between_0.00_and_1.00_rounded_to_2_decimal_places>}. Do not include any other text, code implementations, or solutions."},
                        {"role": "user", "content": "EVALUATE THIS PROMPT (DO NOT SOLVE IT):\n\n" + prompt}
                    ],
                    temperature=0.0
                )

            # Define async wrapper for _safe_get_score
            async def async_safe_get_score(evaluation_func, metric_name: str):
                return _safe_get_score(evaluation_func, metric_name)

            # Run all evaluations concurrently
            clarity_task = asyncio.create_task(async_safe_get_score(clarity_eval, "clarity"))
            specificity_task = asyncio.create_task(async_safe_get_score(specificity_eval, "specificity"))
            technical_task = asyncio.create_task(async_safe_get_score(technical_eval, "technical_validity"))
            context_task = asyncio.create_task(async_safe_get_score(context_retention_eval, "context_retention"))
            effectiveness_task = asyncio.create_task(async_safe_get_score(effectiveness_eval, "effectiveness"))
            innovation_task = asyncio.create_task(async_safe_get_score(innovation_eval, "innovation"))
            intent_task = asyncio.create_task(async_safe_get_score(intent_alignment_eval, "intent_alignment"))
            adaptability_task = asyncio.create_task(async_safe_get_score(adaptability_eval, "adaptability"))
            robustness_task = asyncio.create_task(async_safe_get_score(robustness_eval, "robustness"))
            generalization_task = asyncio.create_task(async_safe_get_score(generalization_eval, "generalization"))

            # Wait for all tasks to complete
            await asyncio.gather(
                clarity_task, specificity_task, technical_task, context_task,
                effectiveness_task, innovation_task, intent_task,
                adaptability_task, robustness_task, generalization_task
            )

            # Return results
            return {
                'clarity': clarity_task.result(),
                'specificity': specificity_task.result(),
                'technical_validity': technical_task.result(),
                'context_retention': context_task.result(),
                'effectiveness': effectiveness_task.result(),
                'innovation': innovation_task.result(),
                'intent_alignment': intent_task.result(),
                'adaptability': adaptability_task.result(),
                'robustness': robustness_task.result(),
                'generalization': generalization_task.result()
            }

        except Exception as e:
            print(f"LLM evaluation error: {str(e)}")
            # Return conservative scores on overall error
            return {metric: 0.3 for metric in self.metrics.get_all_metrics()}

    def _rule_based_evaluation(self, prompt: str, context: Dict) -> Dict[str, float]:
        """
        Backup evaluation using rule-based metrics.
        
        IMPROVEMENTS:
        - More sophisticated heuristics for each metric
        - Better detection of structural elements
        - Improved keyword and pattern matching
        """
        scores = {}
        prompt_lower = prompt.lower()
        words = prompt.split()
        
        # Clarity score based on sentence structure and length
        sentences = max(1, prompt.count('.') + prompt.count('!') + prompt.count('?'))
        avg_words_per_sentence = len(words) / sentences if sentences > 0 else len(words)
        
        # Optimal sentence length is around 15-20 words
        clarity_score = min(1.0, max(0.3, 2.5 / (1.0 + 0.1 * abs(avg_words_per_sentence - 17))))
        
        # Specificity score based on presence of specific details
        detail_keywords = [
            'specifically', 'exactly', 'precisely', 'must', 'required', 
            'ensure', 'validate', 'verify', 'check', 'confirm'
        ]
        detail_count = sum(1 for word in detail_keywords if word in prompt_lower)
        specificity_score = min(1.0, 0.3 + (detail_count / len(detail_keywords)))
        
        # Technical validity score based on structure
        has_context = 'context' in prompt_lower or 'background' in prompt_lower
        has_requirements = 'require' in prompt_lower or 'need' in prompt_lower
        has_constraints = 'limit' in prompt_lower or 'constraint' in prompt_lower or 'max' in prompt_lower
        has_formatting = prompt.count('\n') >= 2 or '```' in prompt or ':' in prompt
        technical_score = (has_context + has_requirements + has_constraints + has_formatting) / 4.0
        technical_score = max(0.3, technical_score)
        
        # Context retention score
        context_keywords = context.get('keywords', [])
        context_keywords.extend(context.get('constraints', []))
        retained_context = sum(keyword.lower() in prompt_lower for keyword in context_keywords)
        context_score = min(1.0, 0.3 + (retained_context / max(1, len(context_keywords))) * 0.7)
        
        # Adaptability score based on flexible language
        adaptability_markers = ['if', 'when', 'where', 'whenever', 'various', 'different', 'multiple']
        adaptability_count = sum(1 for marker in adaptability_markers if marker in prompt_lower)
        adaptability_score = min(1.0, 0.3 + (adaptability_count / len(adaptability_markers)))
        
        # Robustness score based on constraints and verification
        robustness_markers = ['ensure', 'verify', 'check', 'confirm', 'validate', 'must', 'should']
        robustness_count = sum(1 for marker in robustness_markers if marker in prompt_lower)
        robustness_score = min(1.0, 0.3 + (robustness_count / len(robustness_markers)))
        
        # Generalization score based on domain independence
        domain_markers = ['in', 'for', 'any', 'all', 'every', 'each', 'general']
        generalization_count = sum(1 for marker in domain_markers if marker in prompt_lower)
        generalization_score = min(1.0, 0.3 + (generalization_count / len(domain_markers)))
        
        scores = {
            'clarity': clarity_score,
            'specificity': specificity_score,
            'technical_validity': technical_score,
            'context_retention': context_score,
            'adaptability': adaptability_score,
            'robustness': robustness_score,
            'generalization': generalization_score,
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
        
        IMPROVEMENTS:
        - Weighted combination with preference for LLM scores when high confidence
        - Better handling of missing metrics
        - More sophisticated score merging
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
            
            # Apply weights: 70% for better score, 30% for worse score
            aggregated_scores[metric] = (max_score * 0.75) + (min_score * 0.25)

            # Ensure minimum score of 0.1
            aggregated_scores[metric] = max(0.1, aggregated_scores[metric])

        return aggregated_scores

    async def evaluate(self, prompt: str, context: Dict) -> Dict[str, float]:
        """
        Main evaluation method that orchestrates the evaluation pipeline based on config.
        Returns a dictionary of scores for each metric including an overall weighted score.
        
        IMPROVEMENTS:
        - Better handling of different evaluation types
        - More robust score aggregation
        - Improved overall score calculation
        """
        llm_scores = {}
        rule_scores = {}
        final_scores = {} # For combined

        # Conditionally get scores based on evaluation type
        if default_config.evaluation_type in ["llm", "combined"]:
            llm_scores = await self._llm_evaluation(prompt, context)

            # If we're using LLM evaluation only, make sure we don't fall back to rule-based
            if default_config.evaluation_type == "llm":
                final_scores = llm_scores

        # Only get rule-based scores if we're using rule or combined evaluation
        if default_config.evaluation_type in ["rule", "combined"]:
            rule_scores = self._rule_based_evaluation(prompt, context)

            # If we're using rule-based evaluation only, use those scores
            if default_config.evaluation_type == "rule":
                final_scores = rule_scores

        # Only aggregate scores if we're using combined evaluation
        if default_config.evaluation_type == "combined":
            final_scores = self._aggregate_scores(llm_scores, rule_scores)

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
        
        # Limit history size
        if len(self.evaluation_results[prompt_hash]) > self._max_history_size:
            self.evaluation_results[prompt_hash] = self.evaluation_results[prompt_hash][-self._max_history_size:]

        return final_scores

    def get_evaluation_history(self, prompt: str) -> List[Dict[str, float]]:
        """
        Retrieve evaluation history for a specific prompt.
        """
        prompt_hash = hash(prompt)
        return self.evaluation_results.get(prompt_hash, [])
    
    def get_average_scores(self, prompt: str) -> Dict[str, float]:
        """
        Get average scores from evaluation history for a prompt.
        """
        history = self.get_evaluation_history(prompt)
        if not history:
            return {metric: 0.3 for metric in self.metrics.get_all_metrics()}
        
        # Calculate average
        avg_scores = {metric: [] for metric in self.metrics.get_all_metrics()}
        for entry in history:
            for metric, score in entry.items():
                avg_scores[metric].append(score)
        
        # Return averages
        result = {}
        for metric, scores in avg_scores.items():
            result[metric] = sum(scores) / len(scores) if scores else 0.3
            result[metric] = max(0.3, min(1.0, result[metric]))
        
        return result
