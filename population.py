"""
Population module for managing groups of strategy individuals.
"""

import random
from typing import List, Dict, Any, Optional

from individual import Individual
from enhancement_strategy import EnhancementStrategy

class Population:
    """Manages a group of individual strategies."""
    
    def __init__(self, size: int = 20, init_strategies: Optional[List[EnhancementStrategy]] = None):
        self.individuals: List[Individual] = []
        self.size = size
        self.generation = 0
        
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
    
    def select_tournament(self, tournament_size: int) -> Individual:
        """Select an individual using tournament selection."""
        if not self.individuals:
            raise ValueError("Population is empty, cannot select individuals")
        
        # Select random contestants
        contestants = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
        
        # Return the contestant with the highest fitness
        return max(contestants, key=lambda ind: ind.fitness.get("overall", 0))
    
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
    
    def add_individual(self, individual: Individual) -> None:
        """Add an individual to the population."""
        self.individuals.append(individual)
    
    def get_elite(self, n: int) -> List[Individual]:
        """Get the top n individuals."""
        sorted_individuals = sorted(
            self.individuals, 
            key=lambda ind: ind.fitness.get("overall", 0),
            reverse=True
        )
        return sorted_individuals[:n]
    def replace_with_offspring(self, offspring: List[Individual], elite_size: int = 2) -> None:
        """Replace the population with new offspring, preserving elites."""
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
        
        # Keep elite individuals
        elite_size = min(elite_size, len(sorted_individuals))
        elite = sorted_individuals[:elite_size]
        
        # Create new population with elites and offspring
        new_population = elite + offspring
        
        # If we have too many individuals, truncate
        if len(new_population) > self.size:
            new_population = new_population[:self.size]
        
        # If we don't have enough, add random individuals
        while len(new_population) < self.size:
            new_population.append(Individual())
        
        self.individuals = new_population
        
        # Increment age of all individuals
        for ind in self.individuals:
            ind.age += 1
    
    def apply_niching(self, similarity_threshold: float = 0.8) -> None:
        """Apply niching to maintain diversity in the population."""
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
                    
                    if fitness1 > fitness2:
                        # Penalize ind2
                        ind2.fitness["overall"] *= (1.0 - (similarity - similarity_threshold))
                    else:
                        # Penalize ind1
                        ind1.fitness["overall"] *= (1.0 - (similarity - similarity_threshold))
    
    def _calculate_similarity(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate similarity between two individuals."""
        # Simple similarity measure based on strategy parameters
        # In a more sophisticated system, this would use more advanced metrics
        
        same_count = 0
        total_count = 0
        
        # Compare boolean parameters
        for param in ["chain_of_thought", "semantic_check", "context_preservation"]:
            if getattr(ind1.strategy, param) == getattr(ind2.strategy, param):
                same_count += 1
            total_count += 1
        
        # Compare numeric parameters with tolerance
        for param in ["temperature"]:
            val1 = getattr(ind1.strategy, param)
            val2 = getattr(ind2.strategy, param)
            if abs(val1 - val2) < 0.1:  # If values are close
                same_count += 1
            total_count += 1
        
        # Compare optional numeric parameters
        for param in ["max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
            val1 = getattr(ind1.strategy, param)
            val2 = getattr(ind2.strategy, param)
            
            # If both are None or both are not None
            if (val1 is None and val2 is None) or (val1 is not None and val2 is not None):
                same_count += 0.5  # Partial match
                
                # If both are not None, check if they're close
                if val1 is not None and val2 is not None:
                    if param == "max_tokens":
                        if abs(val1 - val2) < 50:  # Within 50 tokens
                            same_count += 0.5
                    else:  # Other numeric params
                        if abs(val1 - val2) < 0.1:  # Within 0.1
                            same_count += 0.5
            
            total_count += 1
        
        # Very basic system prompt similarity (in reality, would use NLP)
        # Just check if they're exactly the same
        if ind1.strategy.system_prompt == ind2.strategy.system_prompt:
            same_count += 1
        elif len(set(ind1.strategy.system_prompt.split()) & set(ind2.strategy.system_prompt.split())) > 5:
            # If they share at least 5 words, they're somewhat similar
            same_count += 0.5
        total_count += 1
        
        return same_count / total_count if total_count > 0 else 0
    
    def to_json(self) -> List[Dict[str, Any]]:
        """Convert population to a JSON-serializable list."""
        return [ind.to_json() for ind in self.individuals]
    
    @classmethod
    def from_json(cls, data: List[Dict[str, Any]], size: int = 20) -> 'Population':
        """Create a population from a JSON list."""
        population = cls(size=size)
        population.individuals = [Individual.from_json(ind_data) for ind_data in data]
        return population