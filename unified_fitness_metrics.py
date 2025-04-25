class UnifiedFitnessMetrics:
    """
    A class that implements standardized weights and validation for fitness metrics
    in the prompt evolution system.
    """

    CORE_METRICS = {
        'clarity': 0.15,         # Clear and understandable
        'specificity': 0.15,     # Detailed and precise
        'intent_alignment': 0.20, # Matches original purpose
        'effectiveness': 0.15,    # Likely to achieve goal
        'innovation': 0.05,      # Introduces improvements
        'technical_validity': 0.10, # Structurally sound
        'context_retention': 0.20  # Maintains context - increased from 0.05 to 0.20
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