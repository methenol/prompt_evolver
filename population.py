"""
Population module for managing groups of strategy individuals.

IMPROVEMENTS:
- Added weighted tournament selection for better fitness-based selection
- Enhanced diversity maintenance with genetic distance calculation
- Improved elite preservation with configurable parameters
- Better population management with age-based culling
"""

import random
import math
from typing import List, Dict, Any, Optional, Tuple

from individual import Individual
from enhancement_strategy import EnhancementStrategy

class Population:
    """Manages a group of individual strategies."""
    
    def __init__(self, size: int = 20, init_strategies: Optional[List[EnhancementStrategy]] = None):
        self.individuals: List[Individual] = []
        self.size = size
        self.generation = 0
        self.age_threshold = 15  # Individuals older than this may be culled
        
        # Initialize population with provided strategies or randomly
        if init_strategies:
            for strategy in init_strategies[:size]:
                self.individuals.append(Individual(strategy=strategy))
            
            # If we need more individuals, generate random ones
            remaining = size - len(init_strategies)
            if remaining > 0:
                for _ in range(remaining):
                    self.individuals.append(Individual())
        else:
            # Generate a fully random population
            for _ in range(size):
                self.individuals.append(Individual())
    
    def select_tournament(self, tournament_size: int = 3, 
                         weighted: bool = False) -> Individual:
        """
        Select an individual using tournament selection.
        
        Args:
            tournament_size: Number of individuals in tournament
            weighted: Whether to use weighted selection (higher fitness = higher probability)
            
        Returns:
            The selected individual
        """
        if not self.individuals:
            raise ValueError("Population is empty, cannot select individuals")
        
        # Select random contestants
        contestants = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
        
        if not weighted:
            # Standard tournament selection - highest fitness wins
            return max(contestants, key=lambda ind: ind.fitness.get("overall", 0))
        
        # Weighted tournament selection - fitness-probability weighted
        # Calculate fitness scores
        fitness_scores = [ind.fitness.get("overall", 0) for ind in contestants]
        
        # Apply soft selection pressure (exponential weighting)
        max_fitness = max(fitness_scores)
        if max_fitness > 0:
            weighted_scores = [math.exp(score / (max_fitness * 0.1 + 0.001)) 
                             for score in fitness_scores]
        else:
            weighted_scores = [1.0 for _ in fitness_scores]
        
        total_weight = sum(weighted_scores)
        probabilities = [w / total_weight for w in weighted_scores]
        
        # Select based on probabilities
        selected = random.choices(contestants, weights=probabilities, k=1)[0]
        return selected
    
    def select_roulette(self) -> Individual:
        """Select an individual using roulette wheel selection."""
        if not self.individuals:
            raise ValueError("Population is empty, cannot select individuals")
        
        # Get fitness values, handling possible negative values by shifting
        fitness_values = [ind.fitness.get("overall", 0) for ind in self.individuals]
        min_fitness = min(fitness_values)
        
        # Shift if there are negative values
        if min_fitness < 0:
            adjusted_fitness = [f - min_fitness + 0.1 for f in fitness_values]
        else:
            adjusted_fitness = [max(f, 0.01) for f in fitness_values]  # Ensure positive values
        
        # Calculate selection probabilities
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]
        
        # Select an individual
        return random.choices(self.individuals, weights=probabilities, k=1)[0]
    
    def select_elitism(self, n: int = 2) -> List[Individual]:
        """
        Select top individuals for direct preservation (elitism).
        
        Args:
            n: Number of top individuals to select
            
        Returns:
            List of elite individuals
        """
        if len(self.individuals) < n:
            return self.individuals.copy()
        
        # Sort by fitness and take top n
        sorted_individuals = sorted(
            self.individuals,
            key=lambda ind: ind.fitness.get("overall", 0),
            reverse=True
        )
        return sorted_individuals[:n]
    
    def add_individual(self, individual: Individual) -> None:
        """Add an individual to the population."""
        self.individuals.append(individual)
    
    def get_elite(self, n: int) -> List[Individual]:
        """
        Get the top n individuals.
        
        Args:
            n: Number of elite individuals to return
            
        Returns:
            List of elite individuals sorted by fitness
        """
        if not self.individuals:
            return []
        
        # Clamp n to available population
        n = min(n, len(self.individuals))
        
        sorted_individuals = sorted(
            self.individuals, 
            key=lambda ind: ind.fitness.get("overall", 0),
            reverse=True
        )
        return sorted_individuals[:n]
    
    def replace_with_offspring(self, offspring: List[Individual], 
                              elite_size: int = 2, 
                              diversity_preserved: bool = True) -> None:
        """
        Replace the population with new offspring, preserving elites and maintaining diversity.
        
        Args:
            offspring: List of new individuals to add to population
            elite_size: Number of top individuals to preserve from previous generation
            diversity_preserved: Whether to actively maintain diversity
        """
        if not offspring:
            return
        
        # Increment generation counter
        self.generation += 1
        
        # Sort individuals by fitness
        sorted_individuals = sorted(
            self.individuals, 
            key=lambda ind: ind.fitness.get("overall", 0),
            reverse=True
        )
        
        # Keep elite individuals - clamp to min of elite_size, len(sorted_individuals), and self.size
        elite_size = min(elite_size, len(sorted_individuals), self.size)
        elite = sorted_individuals[:elite_size]
        
        # Optionally remove low-fitness, old individuals - apply to sorted_individuals before building new_population
        if diversity_preserved:
            # Filter out individuals that are very old with low fitness before creating new_population
            filtered_individuals = [
                ind for ind in sorted_individuals
                if ind.age < self.age_threshold or 
                   ind.fitness.get("overall", 0) > 0.3
            ]
            # Re-sort after filtering to get correct elites
            sorted_individuals = filtered_individuals
            elite = sorted_individuals[:elite_size]
        else:
            self.individuals = sorted_individuals
        
        # Create new population with elites and offspring
        new_population = elite + offspring
        
        # If we have too many individuals, truncate
        if len(new_population) > self.size:
            # Keep elites, then add diverse offspring
            new_population = elite + self._select_diverse_offspring(offspring, self.size - elite_size)
        
        # If we don't have enough, add random individuals
        while len(new_population) < self.size:
            new_population.append(Individual())
        
        self.individuals = new_population
        
        # Increment age of all individuals
        for ind in self.individuals:
            ind.age += 1
    
    def _select_diverse_offspring(self, offspring: List[Individual], count: int) -> List[Individual]:
        """
        Select offspring that maximize diversity.
        
        Args:
            offspring: List of candidate offspring
            count: Number of offspring to select
            
        Returns:
            List of diverse offspring
        """
        if not offspring or count <= 0:
            return []
        
        selected = []
        remaining = offspring.copy()
        
        # Always select the highest fitness individual first
        if remaining:
            highest = max(remaining, key=lambda ind: ind.fitness.get("overall", 0))
            selected.append(highest)
            remaining = [ind for ind in remaining if ind != highest]
        
        # Select diverse individuals based on fitness and diversity
        while len(selected) < count and remaining:
            # Calculate diversity score for each candidate
            candidates_with_scores = []
            for ind in remaining:
                diversity_score = self._calculate_diversity(ind, selected)
                fitness_score = ind.fitness.get("overall", 0)
                # Combined score: 60% fitness, 40% diversity
                combined_score = fitness_score * 0.6 + diversity_score * 0.4
                candidates_with_scores.append((ind, combined_score))
            
            if candidates_with_scores:
                # Select the one with best combined score
                candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
                selected_candidate, score = candidates_with_scores[0]
                selected.append(selected_candidate)
                remaining = [ind for ind in remaining if ind != selected_candidate]
            else:
                break
        
        return selected[:count]
    
    def _calculate_diversity(self, individual: Individual, 
                            selected: List[Individual]) -> float:
        """
        Calculate how diverse an individual is from the selected set.
        
        Args:
            individual: Individual to evaluate
            selected: List of already selected individuals
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if not selected:
            return 1.0
        
        # Calculate minimum distance to any selected individual
        min_distances = []
        for s in selected:
            distance = 1.0 - self._calculate_similarity(individual, s)
            min_distances.append(distance)
        
        # Return average diversity
        return sum(min_distances) / len(min_distances) if min_distances else 0.0
    
    def apply_niching(self, similarity_threshold: float = 0.85, 
                     penalty_strength: float = 0.15) -> None:
        """
        Apply niching to maintain diversity in the population.
        
        Args:
            similarity_threshold: Similarity above which individuals are considered duplicates
            penalty_strength: How much to penalize similar individuals
        """
        if len(self.individuals) <= 1:
            return
        
        # Calculate similarity between all individuals
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                similarity = self._calculate_similarity(self.individuals[i], self.individuals[j])
                
                # If individuals are very similar, penalize the one with lower fitness
                if similarity > similarity_threshold:
                    ind1, ind2 = self.individuals[i], self.individuals[j]
                    fitness1 = ind1.fitness.get("overall", 0)
                    fitness2 = ind2.fitness.get("overall", 0)
                    
                    # Calculate penalty
                    penalty = penalty_strength * (similarity - similarity_threshold)
                    
                    if fitness1 >= fitness2:
                        # Penalize ind2
                        ind2.fitness["overall"] = max(0, ind2.fitness.get("overall", 0) - penalty)
                    else:
                        # Penalize ind1
                        ind1.fitness["overall"] = max(0, ind1.fitness.get("overall", 0) - penalty)
    
    def _calculate_similarity(self, ind1: Individual, ind2: Individual) -> float:
        """
        Calculate similarity between two individuals.
        
        This improved version uses a more sophisticated approach including
        both parameter similarity and system prompt similarity.
        """
        total_components = 0
        total_similarity = 0.0
        
        # Compare boolean parameters
        bool_params = ["chain_of_thought", "semantic_check", "context_preservation"]
        for param in bool_params:
            total_components += 1
            if getattr(ind1.strategy, param) == getattr(ind2.strategy, param):
                total_similarity += 1.0
        
        # Compare numeric parameters with tolerance
        numeric_params = ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
        for param in numeric_params:
            total_components += 1
            val1 = getattr(ind1.strategy, param)
            val2 = getattr(ind2.strategy, param)
            
            if val1 is None and val2 is None:
                total_similarity += 1.0
            elif val1 is not None and val2 is not None:
                if param == "max_tokens":
                    # For token count, use relative difference
                    max_val = max(val1, val2)
                    if max_val > 0:
                        diff = abs(val1 - val2) / max_val
                        total_similarity += max(0, 1.0 - diff)
                    else:
                        total_similarity += 1.0 if val1 == val2 else 0.0
                else:
                    # For other params, use absolute difference
                    max_val = max(abs(val1), abs(val2), 1.0)
                    diff = abs(val1 - val2) / max_val
                    total_similarity += max(0, 1.0 - diff)
            else:
                # One is None, one is not
                total_similarity += 0.5
        
        # Compare system prompts using word overlap
        s1_words = set(ind1.strategy.system_prompt.lower().split())
        s2_words = set(ind2.strategy.system_prompt.lower().split())
        
        if s1_words and s2_words:
            total_components += 1
            intersection = len(s1_words & s2_words)
            union = len(s1_words | s2_words)
            total_similarity += intersection / union if union > 0 else 0.0
        elif ind1.strategy.system_prompt == ind2.strategy.system_prompt:
            total_similarity += 1.0
            total_components += 1
        
        return total_similarity / total_components if total_components > 0 else 0.0
    
    def get_diversity_metrics(self) -> Dict[str, float]:
        """
        Calculate diversity metrics for the current population.
        
        Returns:
            Dictionary with diversity metrics
        """
        if len(self.individuals) < 2:
            return {
                'average_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 1.0,
                'average_fitness': 0.0
            }
        
        similarities = []
        fitness_values = []
        
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                similarity = self._calculate_similarity(self.individuals[i], self.individuals[j])
                similarities.append(similarity)
            
            fitness_values.append(self.individuals[i].fitness.get("overall", 0))
        
        if not similarities:
            avg_sim = 0.0
        else:
            avg_sim = sum(similarities) / len(similarities)
            max_sim = max(similarities)
            min_sim = min(similarities)
        
        return {
            'average_similarity': avg_sim,
            'max_similarity': max_sim,
            'min_similarity': min_sim,
            'average_fitness': sum(fitness_values) / len(fitness_values) if fitness_values else 0.0,
            'fitness_variance': sum((f - sum(fitness_values)/len(fitness_values))**2 for f in fitness_values) / len(fitness_values) if fitness_values else 0.0
        }
    
    def to_json(self) -> List[Dict[str, Any]]:
        """Convert population to a JSON-serializable list."""
        return [ind.to_json() for ind in self.individuals]
    
    @classmethod
    def from_json(cls, data: List[Dict[str, Any]], size: int = 20) -> 'Population':
        """Create a population from a JSON list."""
        population = cls(size=size)
        population.individuals = [Individual.from_json(ind_data) for ind_data in data]
        return population
