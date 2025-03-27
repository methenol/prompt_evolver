from typing import Dict, Set, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
from pathlib import Path

@dataclass
class GenerationMetrics:
    """Stores metrics for a single generation"""
    generation_number: int
    timestamp: datetime
    scores: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class Pattern:
    """Represents a success or failure pattern"""
    pattern_type: str  # 'success' or 'failure'
    features: Dict[str, Any]
    frequency: int
    impact_score: float

class EvolutionHistory:
    """
    Tracks and analyzes historical performance of prompt evolution.
    Provides pattern recognition and trend analysis capabilities.
    """

    def __init__(self):
        self.generation_scores: Dict[int, GenerationMetrics] = {}
        self.success_patterns: List[Pattern] = []
        self.failure_patterns: List[Pattern] = []
        self.trend_data: Dict[str, List[float]] = {}
        
    def add_generation_result(self, 
                            generation: int,
                            scores: Dict[str, float],
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record results for a generation"""
        self.generation_scores[generation] = GenerationMetrics(
            generation_number=generation,
            timestamp=datetime.now(),
            scores=scores,
            metadata=metadata or {}
        )
        self._update_trends(generation, scores)
        self._analyze_patterns(generation)

    def _update_trends(self, generation: int, scores: Dict[str, float]) -> None:
        """Update trend data with new scores"""
        for metric, score in scores.items():
            if metric not in self.trend_data:
                self.trend_data[metric] = []
            self.trend_data[metric].append(score)

    def _analyze_patterns(self, generation: int) -> None:
        """Analyze current generation for patterns"""
        current_metrics = self.generation_scores[generation]
        
        # Define success/failure thresholds
        SUCCESS_THRESHOLD = 0.8
        FAILURE_THRESHOLD = 0.4
        
        # Extract features for pattern analysis
        # Helper function to safely compute mean from values that might be floats or dicts of floats
        def safe_mean(values):
            safe_values = []
            for v in values:
                if isinstance(v, dict):
                    # Compute average of inner dict values if dict is not empty
                    safe_values.append(sum(v.values()) / len(v) if len(v) > 0 else 0.0)
                else:
                    safe_values.append(v)
            return np.mean(safe_values)
        
        def safe_var(values):
            safe_values = []
            for v in values:
                if isinstance(v, dict):
                    safe_values.append(sum(v.values()) / len(v) if len(v) > 0 else 0.0)
                else:
                    safe_values.append(v)
            return np.var(safe_values)
        
        features = {
            'avg_score': safe_mean(list(current_metrics.scores.values())),
            'score_variance': safe_var(list(current_metrics.scores.values())),
            'top_metric': max({k: (sum(v.values())/len(v)) if isinstance(v, dict) else v for k, v in current_metrics.scores.items()}.items(), key=lambda x: x[1])[0],
            'bottom_metric': min({k: (sum(v.values())/len(v)) if isinstance(v, dict) else v for k, v in current_metrics.scores.items()}.items(), key=lambda x: x[1])[0]
        }
        
        # Analyze for success patterns
        if features['avg_score'] >= SUCCESS_THRESHOLD:
            pattern = Pattern(
                pattern_type='success',
                features=features,
                frequency=1,
                impact_score=features['avg_score']
            )
            self._update_patterns(self.success_patterns, pattern)
        
        # Analyze for failure patterns
        elif features['avg_score'] <= FAILURE_THRESHOLD:
            pattern = Pattern(
                pattern_type='failure',
                features=features,
                frequency=1,
                impact_score=1 - features['avg_score']
            )
            self._update_patterns(self.failure_patterns, pattern)

    def _update_patterns(self, pattern_list: List[Pattern], new_pattern: Pattern) -> None:
        """Update pattern list with new pattern"""
        for existing in pattern_list:
            if self._patterns_match(existing, new_pattern):
                existing.frequency += 1
                existing.impact_score = (existing.impact_score + new_pattern.impact_score) / 2
                return
        pattern_list.append(new_pattern)

    def _patterns_match(self, p1: Pattern, p2: Pattern) -> bool:
        """Compare two patterns for similarity"""
        if p1.pattern_type != p2.pattern_type:
            return False
            
        SIMILARITY_THRESHOLD = 0.9
        
        # Compare numerical features with tolerance
        for key in ['avg_score', 'score_variance']:
            if abs(p1.features[key] - p2.features[key]) > 0.1:
                return False
        
        # Compare categorical features
        return (p1.features['top_metric'] == p2.features['top_metric'] and
                p1.features['bottom_metric'] == p2.features['bottom_metric'])

    def get_generation_analysis(self, generation: int) -> Dict[str, Any]:
        """Get detailed analysis for a specific generation"""
        if generation not in self.generation_scores:
            raise ValueError(f"No data for generation {generation}")
            
        metrics = self.generation_scores[generation]
        
        # Calculate improvement over previous generation
        improvements = {}
        if generation > 0 and (generation - 1) in self.generation_scores:
            prev_metrics = self.generation_scores[generation - 1]
            for metric, score in metrics.scores.items():
                prev_score = prev_metrics.scores.get(metric, 0)
                improvements[metric] = score - prev_score
        
        return {
            'metrics': metrics,
            'improvements': improvements,
            'success_patterns': [p for p in self.success_patterns 
                               if self._pattern_applies(p, metrics)],
            'failure_patterns': [p for p in self.failure_patterns 
                               if self._pattern_applies(p, metrics)]
        }

    def _pattern_applies(self, pattern: Pattern, metrics: GenerationMetrics) -> bool:
        """Check if a pattern applies to given metrics"""
        features = {
            'avg_score': np.mean(list(metrics.scores.values())),
            'score_variance': np.var(list(metrics.scores.values())),
            'top_metric': max(metrics.scores.items(), key=lambda x: x[1])[0],
            'bottom_metric': min(metrics.scores.items(), key=lambda x: x[1])[0]
        }
        return self._patterns_match(
            Pattern(pattern.pattern_type, features, 1, 0),
            pattern
        )

    def get_trend_analysis(self) -> Dict[str, Any]:
        """Get trend analysis across all generations"""
        trends = {}
        for metric, scores in self.trend_data.items():
            if len(scores) < 2:
                continue
                
            trends[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'trend': np.polyfit(range(len(scores)), scores, 1)[0],
                'improvement': scores[-1] - scores[0]
            }
            
        return {
            'trends': trends,
            'top_success_patterns': sorted(
                self.success_patterns,
                key=lambda p: p.frequency * p.impact_score,
                reverse=True
            )[:3],
            'top_failure_patterns': sorted(
                self.failure_patterns,
                key=lambda p: p.frequency * p.impact_score,
                reverse=True
            )[:3]
        }

    def save_history(self, filepath: str) -> None:
        """Save history to a file"""
        data = {
            'generation_scores': {
                gen: {
                    'generation_number': metrics.generation_number,
                    'timestamp': metrics.timestamp.isoformat(),
                    'scores': metrics.scores,
                    'metadata': metrics.metadata
                }
                for gen, metrics in self.generation_scores.items()
            },
            'trend_data': self.trend_data
        }
        
        Path(filepath).write_text(json.dumps(data, indent=2))

    def load_history(self, filepath: str) -> None:
        """Load history from a file"""
        data = json.loads(Path(filepath).read_text())
        
        self.generation_scores = {
            int(gen): GenerationMetrics(
                generation_number=metrics['generation_number'],
                timestamp=datetime.fromisoformat(metrics['timestamp']),
                scores=metrics['scores'],
                metadata=metrics['metadata']
            )
            for gen, metrics in data['generation_scores'].items()
        }
        
        self.trend_data = data['trend_data']
        
        # Rebuild patterns
        self.success_patterns.clear()
        self.failure_patterns.clear()
        for gen in self.generation_scores:
            self._analyze_patterns(gen)