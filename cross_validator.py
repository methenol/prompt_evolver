from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from enhanced_fitness_evaluator import EnhancedFitnessEvaluator
from advanced_metrics_evaluator import AdvancedMetricsEvaluator, MetricAnalysis
from fitness_evaluation_strategy import FitnessEvaluationStrategy
from unified_fitness_metrics import UnifiedFitnessMetrics

@dataclass
class ValidationResult:
    """Results from cross-validation"""
    score: float
    confidence: float
    discrepancies: Dict[str, float]
    recommendations: List[str]

class CrossValidator:
    """
    Implements cross-validation system for fitness scores.
    Validates results across different evaluation methods and ensures consistency.
    """

    def __init__(self, client=None):
        """Initialize the validator with optional OpenAI client."""
        # Always create metrics for fallback scores
        self.metrics = UnifiedFitnessMetrics()
        
        # Create evaluators
        self.basic_evaluator = EnhancedFitnessEvaluator(client) if client else None
        self.advanced_evaluator = AdvancedMetricsEvaluator()
        self.fallback_evaluator = FitnessEvaluationStrategy(client) if client else None
        
        # Confidence thresholds
        self.HIGH_CONFIDENCE = 0.8
        self.MEDIUM_CONFIDENCE = 0.6
        self.DISCREPANCY_THRESHOLD = 0.2

    async def validate_fitness_scores(self, 
                                   prompt: str,
                                   context: Dict,
                                   history: Optional[List[str]] = None) -> ValidationResult:
        """
        Perform cross-validation of fitness scores across different evaluators.
        Returns validated scores with confidence metrics.
        """
        # Get scores from different evaluators
        if self.basic_evaluator:
            basic_scores = await self.basic_evaluator.evaluate(prompt, context)
        else:
            # If no client available, use conservative scores
            basic_scores = {metric: 0.3 for metric in ['clarity', 'specificity', 'technical_validity', 'context_retention', 'effectiveness', 'innovation', 'intent_alignment']}
        
        advanced_scores = {
            'clarity': self.advanced_evaluator.evaluate_clarity(prompt),
            'specificity': self.advanced_evaluator.evaluate_specificity(prompt, context),
            'innovation': self.advanced_evaluator.evaluate_innovation(prompt, history or []),
            'technical_validity': self.advanced_evaluator.evaluate_technical_validity(prompt)
        }
        
        fallback_scores = await self.fallback_evaluator.evaluate_with_fallback(prompt, context)

        # Perform cross-validation
        validated_scores, discrepancies = self._cross_validate_scores(
            basic_scores,
            advanced_scores,
            fallback_scores
        )

        # Calculate confidence and generate recommendations
        confidence = self._calculate_validation_confidence(discrepancies)
        recommendations = self._generate_validation_recommendations(
            discrepancies,
            confidence
        )

        return ValidationResult(
            score=np.mean(list(validated_scores.values())),
            confidence=confidence,
            discrepancies=discrepancies,
            recommendations=recommendations
        )

    def _cross_validate_scores(self,
                              basic_scores: Dict[str, float],
                              advanced_scores: Dict[str, MetricAnalysis],
                              fallback_scores: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Cross-validate scores from different evaluators.
        Returns validated scores and identified discrepancies.
        """
        validated_scores = {}
        discrepancies = {}
        weights = self.basic_evaluator.metrics.get_all_metrics()
        weighted_sum = 0.0
        total_weight = 0.0

        for metric in weights.keys():
            if metric == 'overall':
                continue  # Skip overall, we'll calculate it after validating individual metrics
                
            scores = []
            eval_weights = []

            # Add basic score with minimum threshold
            basic_score = max(0.3, basic_scores.get(metric, 0.3))
            scores.append(basic_score)
            eval_weights.append(1.0)

            # Add advanced score if available
            if metric in advanced_scores:
                advanced_result = advanced_scores[metric]
                advanced_score = max(0.3, advanced_result.score)
                scores.append(advanced_score)
                eval_weights.append(1.5)  # Give higher weight to advanced metrics

            # Add fallback score
            fallback_score = max(0.3, fallback_scores.get(metric, 0.3))
            scores.append(fallback_score)
            eval_weights.append(0.8)  # Lower weight for fallback

            # Calculate weighted average
            metric_score = max(0.3, np.average(scores, weights=eval_weights))
            validated_scores[metric] = metric_score

            # Calculate discrepancy
            max_diff = max(abs(s - metric_score) for s in scores)
            if max_diff > self.DISCREPANCY_THRESHOLD:
                discrepancies[metric] = max_diff

            # Accumulate weighted sum for overall score
            metric_weight = weights[metric]
            weighted_sum += metric_score * metric_weight
            total_weight += metric_weight

        # Calculate overall score
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.3

        # Ensure overall score is between 0.3 and 1.0
        validated_scores['overall'] = max(0.3, min(1.0, overall_score))

        return validated_scores, discrepancies

    def _calculate_validation_confidence(self, discrepancies: Dict[str, float]) -> float:
        """
        Calculate overall confidence in validation results.
        """
        if not discrepancies:
            return 1.0

        avg_discrepancy = np.mean(list(discrepancies.values()))
        return max(0.0, 1.0 - (avg_discrepancy * 2))

    def _generate_validation_recommendations(self,
                                          discrepancies: Dict[str, float],
                                          confidence: float) -> List[str]:
        """
        Generate recommendations based on validation results.
        """
        recommendations = []

        if confidence < self.MEDIUM_CONFIDENCE:
            recommendations.append(
                "High score variability detected. Consider manual review."
            )

        for metric, discrepancy in discrepancies.items():
            if discrepancy > self.DISCREPANCY_THRESHOLD:
                recommendations.append(
                    f"Significant discrepancy in {metric} scores. "
                    "Consider additional evaluation."
                )

        if confidence < self.HIGH_CONFIDENCE:
            recommendations.append(
                "Consider collecting more evaluation samples to improve confidence."
            )

        return recommendations

    async def validate_evaluation_method(self,
                                      method_name: str,
                                      test_cases: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """
        Validate a specific evaluation method using test cases.
        Returns validation metrics and recommendations.
        """
        results = {
            'consistency_score': 0.0,
            'reliability_score': 0.0,
            'issues_found': [],
            'recommendations': []
        }

        scores_by_metric = {}
        for prompt, context in test_cases:
            validation = await self.validate_fitness_scores(prompt, context)
            
            # Track scores by metric
            for metric, score in validation.discrepancies.items():
                if metric not in scores_by_metric:
                    scores_by_metric[metric] = []
                scores_by_metric[metric].append(score)

        # Analyze consistency
        for metric, scores in scores_by_metric.items():
            std_dev = np.std(scores)
            if std_dev > 0.2:
                results['issues_found'].append(
                    f"High variance in {metric} scores (std_dev: {std_dev:.2f})"
                )

        # Calculate overall consistency
        avg_std_dev = np.mean([np.std(scores) for scores in scores_by_metric.values()])
        results['consistency_score'] = max(0.0, 1.0 - (avg_std_dev * 2))

        # Calculate reliability based on confidence scores
        confidence_scores = [
            await self.validate_fitness_scores(prompt, context).confidence
            for prompt, context in test_cases
        ]
        results['reliability_score'] = np.mean(confidence_scores)

        # Generate recommendations
        if results['consistency_score'] < self.HIGH_CONFIDENCE:
            results['recommendations'].append(
                "Consider calibrating evaluation parameters to improve consistency"
            )
        if results['reliability_score'] < self.HIGH_CONFIDENCE:
            results['recommendations'].append(
                "Implement additional validation checks to improve reliability"
            )

        return results

    def get_validation_stats(self) -> Dict[str, float]:
        """
        Get statistical information about validation performance.
        """
        if not self.basic_evaluator:
            return {
                'validation_count': 0,
                'avg_confidence': 0.0,
                'high_confidence_ratio': 0.0
            }
            
        return {
            'validation_count': len(self.basic_evaluator.evaluation_results),
            'avg_confidence': np.mean([
                result.confidence
                for result in self.basic_evaluator.evaluation_results.values()
            ]) if self.basic_evaluator.evaluation_results else 0.0,
            'high_confidence_ratio': np.mean([
                1.0 if result.confidence >= self.HIGH_CONFIDENCE else 0.0
                for result in self.basic_evaluator.evaluation_results.values()
            ]) if self.basic_evaluator.evaluation_results else 0.0
        }