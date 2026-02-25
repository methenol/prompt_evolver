class UnifiedFitnessMetrics:
    """
    A class that implements standardized weights and validation for fitness metrics
    in the prompt evolution system.
    
    IMPROVEMENTS:
    - Added new metrics: 'adaptability', 'robustness', 'generalization'
    - Improved weight distribution to balance new and existing metrics
    - Better metric metadata with more detailed descriptions
    """

    CORE_METRICS = {
        'clarity': 0.13,               # Clear and understandable
        'specificity': 0.13,           # Detailed and precise
        'intent_alignment': 0.16,      # Matches original purpose
        'effectiveness': 0.13,         # Likely to achieve goal
        'innovation': 0.04,            # Introduces improvements
        'technical_validity': 0.09,    # Structurally sound
        'context_retention': 0.17,     # Maintains context
        'adaptability': 0.07,          # NEW: Adapts well to variations
        'robustness': 0.04,            # NEW: Resilient to minor changes
        'generalization': 0.04,        # NEW: Works across different contexts
    }

    def __init__(self):
        self.metrics = self.CORE_METRICS.copy()
        self._validate_weights()
        self.metric_metadata = self._initialize_metadata()

    def _validate_weights(self) -> None:
        """Validates that metric weights sum to 1.0"""
        total_weight = sum(self.metrics.values())
        if not abs(total_weight - 1.0) < 1e-6:  # Using small epsilon for float comparison
            raise ValueError(f"Metric weights must sum to 1.0, got {total_weight}")

    def _initialize_metadata(self) -> dict:
        """Initialize metadata for each metric including description and example scores"""
        return {
            'clarity': {
                'description': 'How clear and understandable the prompt is',
                'example_high': 'Precise, well-structured prompt with clear instructions',
                'example_low': 'Vague or ambiguous instructions with poor structure'
            },
            'specificity': {
                'description': 'Level of detail and precision in the prompt',
                'example_high': 'Detailed requirements with specific constraints and expectations',
                'example_low': 'Generic instructions lacking necessary details'
            },
            'intent_alignment': {
                'description': 'How well the prompt aligns with intended purpose',
                'example_high': 'Prompt directly addresses core objectives and requirements',
                'example_low': 'Prompt deviates from or misses key objectives'
            },
            'effectiveness': {
                'description': 'Likelihood of achieving desired outcome',
                'example_high': 'Prompt structured to effectively guide desired behavior',
                'example_low': 'Prompt unlikely to produce desired results'
            },
            'innovation': {
                'description': 'Introduction of improvements or novel approaches',
                'example_high': 'Creative solutions that enhance effectiveness',
                'example_low': 'Standard approach without improvement'
            },
            'technical_validity': {
                'description': 'Structural soundness and technical accuracy',
                'example_high': 'Well-formed prompt following best practices',
                'example_low': 'Poor structure or technical inaccuracies'
            },
            'context_retention': {
                'description': 'Ability to maintain relevant context, structure, and critical instructions',
                'example_high': 'Preserves all important sections, instructions, and structural elements while maintaining contextual information',
                'example_low': 'Loses critical sections, instructions, or structural elements from the original prompt'
            },
            'adaptability': {
                'description': 'How well the prompt can adapt to variations in input or context',
                'example_high': 'Prompt handles edge cases gracefully and adapts to different scenarios',
                'example_low': 'Prompt is rigid and fails when conditions change'
            },
            'robustness': {
                'description': 'Resistance to degradation from minor perturbations or variations',
                'example_high': 'Prompt maintains effectiveness despite small changes in wording or structure',
                'example_low': 'Small changes significantly reduce prompt quality'
            },
            'generalization': {
                'description': 'Ability to work across different contexts and domains',
                'example_high': 'Prompt principles apply broadly across multiple scenarios',
                'example_low': 'Prompt is too narrowly focused and only works in specific cases'
            }
        }

    def get_weight(self, metric_name: str) -> float:
        """Get the weight for a specific metric"""
        if metric_name not in self.metrics:
            raise ValueError(f"Unknown metric: {metric_name}")
        return self.metrics[metric_name]

    def get_metadata(self, metric_name: str) -> dict:
        """Get metadata for a specific metric"""
        if metric_name not in self.metric_metadata:
            raise ValueError(f"Unknown metric: {metric_name}")
        return self.metric_metadata[metric_name]

    def get_all_metrics(self) -> dict:
        """Get all metrics and their weights"""
        return self.metrics.copy()

    def get_all_metadata(self) -> dict:
        """Get metadata for all metrics"""
        return self.metric_metadata.copy()

    def get_category_weights(self) -> dict:
        """Get weights grouped by category for advanced analysis"""
        return {
            'quality': {
                'clarity': self.metrics['clarity'],
                'specificity': self.metrics['specificity'],
                'technical_validity': self.metrics['technical_validity']
            },
            'purpose': {
                'intent_alignment': self.metrics['intent_alignment'],
                'effectiveness': self.metrics['effectiveness']
            },
            'preservation': {
                'context_retention': self.metrics['context_retention']
            },
            'resilience': {
                'adaptability': self.metrics['adaptability'],
                'robustness': self.metrics['robustness'],
                'generalization': self.metrics['generalization']
            },
            'improvement': {
                'innovation': self.metrics['innovation']
            }
        }

    def get_improvement_suggestions(self, metrics: dict) -> list:
        """
        Generate improvement suggestions based on current metrics.
        
        Args:
            metrics: Dictionary of current metric scores
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Get lowest scoring metrics
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1])
        
        if sorted_metrics:
            lowest_metric = sorted_metrics[0][0]
            lowest_score = sorted_metrics[0][1]
            
            if lowest_score < 0.5:
                suggestions.append(
                    f"Focus on improving '{lowest_metric}' - current score is {lowest_score:.2f}"
                )
            
            # Check for specific areas of weakness
            if 'adaptability' in metrics and metrics['adaptability'] < 0.5:
                suggestions.append(
                    "Consider adding more flexible instructions that handle edge cases"
                )
            
            if 'robustness' in metrics and metrics['robustness'] < 0.5:
                suggestions.append(
                    "Strengthen key constraints and add verification steps for robustness"
                )
            
            if 'context_retention' in metrics and metrics['context_retention'] < 0.5:
                suggestions.append(
                    "Ensure critical context and constraints are preserved from the original prompt"
                )
        
        return suggestions
