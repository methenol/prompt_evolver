from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from unified_fitness_metrics import UnifiedFitnessMetrics

@dataclass
class MetricAnalysis:
    """Detailed analysis results for a metric"""
    score: float
    confidence: float
    factors: Dict[str, float]
    suggestions: List[str]

class AdvancedMetricsEvaluator:
    """
    Implements advanced metrics evaluation with sophisticated analysis techniques
    and detailed scoring factors.
    """

    def __init__(self):
        self.metrics = UnifiedFitnessMetrics()
        self.analysis_history: List[Dict[str, MetricAnalysis]] = []

    def evaluate_clarity(self, prompt: str) -> MetricAnalysis:
        """
        Advanced clarity evaluation considering multiple factors:
        - Sentence complexity
        - Vocabulary level
        - Structure coherence
        - Formatting quality
        """
        # Analyze sentence complexity
        sentences = prompt.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        complexity_score = self._normalize_score(15, avg_sentence_length, inverse=True)

        # Analyze vocabulary
        words = prompt.lower().split()
        unique_words = len(set(words))
        vocabulary_ratio = unique_words / len(words)
        vocabulary_score = self._normalize_score(0.6, vocabulary_ratio)

        # Analyze structure
        has_sections = any(marker in prompt.lower() for marker in ['context:', 'requirements:', 'example:'])
        structure_score = 0.8 if has_sections else 0.4

        # Analyze formatting
        formatting_score = self._evaluate_formatting(prompt)

        factors = {
            'sentence_complexity': complexity_score,
            'vocabulary_usage': vocabulary_score,
            'structure_coherence': structure_score,
            'formatting_quality': formatting_score
        }

        # Calculate weighted score
        score = np.mean(list(factors.values()))
        confidence = self._calculate_confidence(factors)

        suggestions = self._generate_clarity_suggestions(factors)

        return MetricAnalysis(score, confidence, factors, suggestions)

    def evaluate_specificity(self, prompt: str, context: Dict) -> MetricAnalysis:
        """
        Advanced specificity evaluation considering:
        - Detail density
        - Context utilization
        - Constraint precision
        - Example concreteness
        """
        # Analyze detail density
        detail_markers = ['specifically', 'exactly', 'must', 'required', 'ensure']
        detail_density = sum(prompt.lower().count(marker) for marker in detail_markers)
        detail_score = self._normalize_score(5, detail_density)

        # Analyze context utilization
        context_keywords = context.get('keywords', [])
        context_usage = sum(keyword.lower() in prompt.lower() for keyword in context_keywords)
        context_score = self._normalize_score(len(context_keywords) * 0.7, context_usage)

        # Analyze constraint precision
        constraints = self._extract_constraints(prompt)
        constraint_score = min(1.0, len(constraints) / 5)

        # Analyze example concreteness
        example_score = self._evaluate_examples(prompt)

        factors = {
            'detail_density': detail_score,
            'context_utilization': context_score,
            'constraint_precision': constraint_score,
            'example_concreteness': example_score
        }

        score = np.mean(list(factors.values()))
        confidence = self._calculate_confidence(factors)

        suggestions = self._generate_specificity_suggestions(factors)

        return MetricAnalysis(score, confidence, factors, suggestions)

    def evaluate_innovation(self, prompt: str, history: List[str]) -> MetricAnalysis:
        """
        Advanced innovation evaluation considering:
        - Uniqueness from history
        - Creative elements
        - Improvement patterns
        - Novel combinations
        """
        # Analyze uniqueness
        similarity_scores = [self._calculate_similarity(prompt, hist) for hist in history]
        uniqueness_score = 1.0 - (min(similarity_scores) if similarity_scores else 0.0)

        # Analyze creative elements
        creative_patterns = [
            'alternatively', 'novel', 'innovative', 'creative', 'unique',
            'improve', 'enhance', 'optimize', 'different approach'
        ]
        creativity_score = min(1.0, sum(prompt.lower().count(p) for p in creative_patterns) / 3)

        # Analyze improvement suggestions
        improvement_score = self._evaluate_improvements(prompt)

        # Analyze novel combinations
        combination_score = self._evaluate_novel_combinations(prompt)

        factors = {
            'uniqueness': uniqueness_score,
            'creative_elements': creativity_score,
            'improvement_patterns': improvement_score,
            'novel_combinations': combination_score
        }

        score = np.mean(list(factors.values()))
        confidence = self._calculate_confidence(factors)

        suggestions = self._generate_innovation_suggestions(factors)

        return MetricAnalysis(score, confidence, factors, suggestions)

    def evaluate_technical_validity(self, prompt: str) -> MetricAnalysis:
        """
        Advanced technical validity evaluation considering:
        - Structural integrity
        - Technical accuracy
        - Implementation feasibility
        - Best practices alignment
        """
        # Analyze structural integrity
        structure_score = self._evaluate_structure(prompt)

        # Analyze technical accuracy
        accuracy_score = self._evaluate_technical_accuracy(prompt)

        # Analyze implementation feasibility
        feasibility_score = self._evaluate_feasibility(prompt)

        # Analyze best practices
        practices_score = self._evaluate_best_practices(prompt)

        factors = {
            'structural_integrity': structure_score,
            'technical_accuracy': accuracy_score,
            'implementation_feasibility': feasibility_score,
            'best_practices': practices_score
        }

        score = np.mean(list(factors.values()))
        confidence = self._calculate_confidence(factors)

        suggestions = self._generate_technical_suggestions(factors)

        return MetricAnalysis(score, confidence, factors, suggestions)

    def _normalize_score(self, target: float, value: float, inverse: bool = False) -> float:
        """Normalize a value to a 0.3-1.0 score range, handling zero target gracefully."""
        if target == 0:
            return 1.0 if value == 0 else 0.3
        if inverse:
            raw_score = 2.0 / (1.0 + 0.1 * abs(value - target))
            return max(0.3, min(1.0, raw_score))
        raw_score = value / target
        return max(0.3, min(1.0, raw_score))

    def _calculate_confidence(self, factors: Dict[str, float]) -> float:
        """Calculate confidence score based on factor variance"""
        values = list(factors.values())
        return 1.0 - min(1.0, np.std(values) * 2)

    def _evaluate_formatting(self, prompt: str) -> float:
        """Evaluate prompt formatting quality"""
        has_paragraphs = prompt.count('\n\n') > 0
        has_lists = prompt.count('\n-') > 0 or prompt.count('\n*') > 0
        has_sections = prompt.count(':') > 2
        
        format_score = 0.0
        format_score += 0.3 if has_paragraphs else 0.0
        format_score += 0.3 if has_lists else 0.0
        format_score += 0.4 if has_sections else 0.0
        
        return format_score

    def _extract_constraints(self, prompt: str) -> List[str]:
        """Extract constraint statements from prompt"""
        constraints = []
        for sentence in prompt.split('.'):
            if any(word in sentence.lower() for word in ['must', 'should', 'require', 'need']):
                constraints.append(sentence.strip())
        return constraints

    def _evaluate_examples(self, prompt: str) -> float:
        """Evaluate the concreteness of examples in the prompt"""
        has_example_section = 'example:' in prompt.lower()
        has_code_blocks = '```' in prompt
        has_specific_cases = 'for instance' in prompt.lower() or 'e.g.' in prompt.lower()
        
        example_score = 0.0
        example_score += 0.4 if has_example_section else 0.0
        example_score += 0.3 if has_code_blocks else 0.0
        example_score += 0.3 if has_specific_cases else 0.0
        
        return example_score

    def _calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts"""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        union = words1.union(words2)
        if len(union) == 0:
            return 1.0
        intersection = words1.intersection(words2)
        return len(intersection) / len(union)

    def _evaluate_improvements(self, prompt: str) -> float:
        """Evaluate improvement suggestions in the prompt"""
        improvement_patterns = [
            'improve', 'enhance', 'optimize', 'better than',
            'more efficient', 'increased', 'upgraded'
        ]
        return min(1.0, sum(prompt.lower().count(p) for p in improvement_patterns) / 4)

    def _evaluate_novel_combinations(self, prompt: str) -> float:
        """Evaluate novel combination of concepts"""
        combination_markers = [
            'combine', 'integrate', 'merge', 'hybrid',
            'mixed', 'blend', 'fusion', 'composite'
        ]
        return min(1.0, sum(prompt.lower().count(m) for m in combination_markers) / 3)

    def _evaluate_structure(self, prompt: str) -> float:
        """Evaluate structural integrity of the prompt"""
        has_intro = any(s.strip().endswith(':') for s in prompt.split('\n')[:3])
        has_body = len(prompt.split('\n')) > 5
        has_conclusion = any(
            marker in prompt.lower() 
            for marker in ['finally', 'in conclusion', 'to summarize']
        )
        
        structure_score = 0.0
        structure_score += 0.3 if has_intro else 0.0
        structure_score += 0.4 if has_body else 0.0
        structure_score += 0.3 if has_conclusion else 0.0
        
        return structure_score

    def _evaluate_technical_accuracy(self, prompt: str) -> float:
        """Evaluate technical accuracy of the prompt"""
        technical_markers = [
            'function', 'method', 'class', 'interface',
            'api', 'database', 'algorithm', 'protocol'
        ]
        return min(1.0, sum(prompt.lower().count(m) for m in technical_markers) / 4)

    def _evaluate_feasibility(self, prompt: str) -> float:
        """Evaluate implementation feasibility"""
        complexity_markers = [
            'complex', 'difficult', 'challenging', 'advanced',
            'sophisticated', 'complicated', 'intricate'
        ]
        complexity_count = sum(prompt.lower().count(m) for m in complexity_markers)
        return max(0.0, 1.0 - (complexity_count * 0.2))

    def _evaluate_best_practices(self, prompt: str) -> float:
        """Evaluate alignment with best practices"""
        practice_markers = [
            'best practice', 'standard', 'convention',
            'pattern', 'principle', 'guideline'
        ]
        return min(1.0, sum(prompt.lower().count(m) for m in practice_markers) / 3)

    def _generate_clarity_suggestions(self, factors: Dict[str, float]) -> List[str]:
        """Generate suggestions for improving clarity"""
        suggestions = []
        if factors['sentence_complexity'] < 0.7:
            suggestions.append("Consider simplifying complex sentences")
        if factors['vocabulary_usage'] < 0.7:
            suggestions.append("Use more diverse vocabulary while maintaining clarity")
        if factors['structure_coherence'] < 0.7:
            suggestions.append("Add clear section markers to improve structure")
        if factors['formatting_quality'] < 0.7:
            suggestions.append("Improve formatting with paragraphs and lists")
        return suggestions

    def _generate_specificity_suggestions(self, factors: Dict[str, float]) -> List[str]:
        """Generate suggestions for improving specificity"""
        suggestions = []
        if factors['detail_density'] < 0.7:
            suggestions.append("Add more specific requirements and constraints")
        if factors['context_utilization'] < 0.7:
            suggestions.append("Incorporate more context-specific information")
        if factors['constraint_precision'] < 0.7:
            suggestions.append("Define more precise constraints and requirements")
        if factors['example_concreteness'] < 0.7:
            suggestions.append("Include more concrete examples")
        return suggestions

    def _generate_innovation_suggestions(self, factors: Dict[str, float]) -> List[str]:
        """Generate suggestions for improving innovation"""
        suggestions = []
        if factors['uniqueness'] < 0.7:
            suggestions.append("Introduce more unique elements to differentiate")
        if factors['creative_elements'] < 0.7:
            suggestions.append("Add more creative approaches and alternatives")
        if factors['improvement_patterns'] < 0.7:
            suggestions.append("Include more specific improvement suggestions")
        if factors['novel_combinations'] < 0.7:
            suggestions.append("Consider combining different approaches")
        return suggestions

    def _generate_technical_suggestions(self, factors: Dict[str, float]) -> List[str]:
        """Generate suggestions for improving technical validity"""
        suggestions = []
        if factors['structural_integrity'] < 0.7:
            suggestions.append("Improve structural organization")
        if factors['technical_accuracy'] < 0.7:
            suggestions.append("Include more precise technical specifications")
        if factors['implementation_feasibility'] < 0.7:
            suggestions.append("Consider simplifying implementation requirements")
        if factors['best_practices'] < 0.7:
            suggestions.append("Align more closely with industry best practices")
        return suggestions