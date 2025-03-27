"""
Individual module for representing a single prompt enhancement strategy.
"""

import random
import uuid
from dataclasses import asdict
from typing import Dict, Any, List, Tuple, Optional

from enhancement_strategy import EnhancementStrategy

class Individual:
    """Represents a single prompt enhancement strategy."""
    
    def __init__(
        self, 
        strategy: Optional[EnhancementStrategy] = None,
        id: Optional[str] = None
    ):
        self.id = id if id else str(uuid.uuid4())
        self.strategy = strategy if strategy else self._generate_random_strategy()
        self.fitness: Dict[str, float] = {"overall": 0.0}
        self.prompt_result: Optional[str] = None
        self.age: int = 0
        self.parent_ids: List[str] = []
        self.generation: int = 0
        
    def _generate_random_strategy(self) -> EnhancementStrategy:
        """Generate a random strategy when no strategy is provided."""
        return EnhancementStrategy(
            name=f"Evolved-{self.id[:8]}",
            temperature=random.uniform(0.3, 0.9),
            chain_of_thought=random.choice([True, False]),
            semantic_check=random.choice([True, False]),
            context_preservation=random.choice([True, False]),
            system_prompt=self._generate_random_system_prompt(),
            max_tokens=random.choice([None, 150, 250, 350]),
            top_p=random.choice([None, 0.8, 0.9, 1.0]),
            frequency_penalty=random.choice([None, 0.0, 0.1, 0.2, 0.3]),
            presence_penalty=random.choice([None, 0.0, 0.1, 0.2, 0.3])
        )
    
    def _generate_random_system_prompt(self) -> str:
        """Generate a random system prompt for the strategy."""
        focus_elements = [
            "clarity and effectiveness",
            "innovative improvements",
            "maintaining original intent",
            "logical structure",
            "adding context",
            "brevity",
            "precision",
            "generating variations"
        ]
        
        styles = [
            "balanced",
            "creative",
            "conservative",
            "analytical",
            "detail-oriented",
            "concise",
            "technical",
            "exploratory"
        ]
        
        focus = random.choice(focus_elements)
        style = random.choice(styles)
        
        return f"You are a {style} prompt enhancement assistant. Focus on {focus} while improving the prompt."
    
    def mutate(self, mutation_rate: float) -> 'Individual':
        """Create a mutated copy of this individual."""
        if random.random() > mutation_rate:
            return Individual(strategy=self.strategy, id=str(uuid.uuid4()))
        
        # Create a copy of the strategy to mutate
        strategy_dict = asdict(self.strategy)
        
        # Choose which aspects to mutate
        if random.random() < 0.3:  # 30% chance to mutate system prompt
            strategy_dict["system_prompt"] = self._mutate_system_prompt(strategy_dict["system_prompt"])
        
        if random.random() < 0.5:  # 50% chance to mutate numeric parameters
            # Choose one or more parameters to mutate
            params_to_mutate = random.sample([
                "temperature", "max_tokens", "top_p", 
                "frequency_penalty", "presence_penalty"
            ], k=random.randint(1, 3))
            
            for param in params_to_mutate:
                if param == "temperature":
                    strategy_dict[param] = max(0.1, min(1.0, strategy_dict[param] + random.uniform(-0.2, 0.2)))
                elif param == "max_tokens":
                    if strategy_dict[param] is None:
                        strategy_dict[param] = random.choice([150, 250, 350])
                    else:
                        strategy_dict[param] = max(100, strategy_dict[param] + random.choice([-50, 50, 100]))
                elif param == "top_p":
                    if strategy_dict[param] is None:
                        strategy_dict[param] = random.uniform(0.7, 1.0)
                    else:
                        strategy_dict[param] = max(0.1, min(1.0, strategy_dict[param] + random.uniform(-0.1, 0.1)))
                elif param in ["frequency_penalty", "presence_penalty"]:
                    if strategy_dict[param] is None:
                        strategy_dict[param] = random.uniform(0.0, 0.5)
                    else:
                        strategy_dict[param] = max(0.0, min(2.0, strategy_dict[param] + random.uniform(-0.1, 0.1)))
        
        if random.random() < 0.3:  # 30% chance to mutate boolean parameters
            bool_params = ["chain_of_thought", "semantic_check", "context_preservation"]
            param = random.choice(bool_params)
            strategy_dict[param] = not strategy_dict[param]
        
        # Create new strategy with mutated parameters
        new_strategy = EnhancementStrategy(**strategy_dict)
        new_strategy.name = f"Evolved-{str(uuid.uuid4())[:8]}"
        
        # Create new individual with the mutated strategy
        new_individual = Individual(strategy=new_strategy)
        new_individual.parent_ids = [self.id]
        new_individual.generation = self.generation + 1
        
        return new_individual
    
    def _mutate_system_prompt(self, system_prompt: str) -> str:
        """Mutate the system prompt."""
        focus_elements = [
            "clarity and effectiveness",
            "innovative improvements",
            "maintaining original intent",
            "logical structure",
            "adding context",
            "brevity",
            "precision",
            "generating variations"
        ]
        
        modifiers = [
            "while preserving original meaning",
            "with special attention to detail",
            "using concise language",
            "with comprehensive explanations",
            "by breaking down complex concepts",
            "by adding relevant examples"
        ]
        
        # 50% chance to add a modifier, 50% chance to change the focus
        if random.random() < 0.5:
            # Add a modifier
            modifier = random.choice(modifiers)
            # Check if the prompt already has this modifier
            if modifier not in system_prompt:
                return system_prompt + " " + modifier
            return system_prompt
        else:
            # Change the focus
            new_focus = random.choice(focus_elements)
            # Simple replacement - in a more sophisticated system, this would use NLP
            return system_prompt.split("Focus on")[0] + "Focus on " + new_focus + "."
    
    @staticmethod
    def crossover(parent1: 'Individual', parent2: 'Individual') -> Tuple['Individual', 'Individual']:
        """Create two children by crossing over two parents."""
        # Create dictionaries of both parents' strategies
        p1_dict = asdict(parent1.strategy)
        p2_dict = asdict(parent2.strategy)
        
        # Create child dictionaries
        c1_dict = p1_dict.copy()
        c2_dict = p2_dict.copy()
        
        # Parameters that can be swapped
        numeric_params = ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
        bool_params = ["chain_of_thought", "semantic_check", "context_preservation"]
        
        # Crossover numeric parameters (random swapping)
        for param in numeric_params:
            if random.random() < 0.5:  # 50% chance to swap each parameter
                c1_dict[param], c2_dict[param] = c2_dict[param], c1_dict[param]
        
        # Crossover boolean parameters
        for param in bool_params:
            if random.random() < 0.5:  # 50% chance to swap each parameter
                c1_dict[param], c2_dict[param] = c2_dict[param], c1_dict[param]
        
        # Crossover system prompts (more complex)
        if random.random() < 0.3:  # 30% chance to combine system prompts
            # Simple combination - in a real system, this would be more sophisticated
            parts1 = c1_dict["system_prompt"].split('.')
            parts2 = c2_dict["system_prompt"].split('.')
            
            if len(parts1) > 1 and len(parts2) > 1:
                # Combine first part from parent1 and second part from parent2
                c1_dict["system_prompt"] = parts1[0] + '. ' + parts2[1] if len(parts2) > 1 else parts2[0]
                c2_dict["system_prompt"] = parts2[0] + '. ' + parts1[1] if len(parts1) > 1 else parts1[0]
        else:
            # Simple swap
            if random.random() < 0.5:
                c1_dict["system_prompt"], c2_dict["system_prompt"] = c2_dict["system_prompt"], c1_dict["system_prompt"]
        
        # Generate unique names for children
        c1_dict["name"] = f"Evolved-{str(uuid.uuid4())[:8]}"
        c2_dict["name"] = f"Evolved-{str(uuid.uuid4())[:8]}"
        
        # Create child strategies
        child1_strategy = EnhancementStrategy(**c1_dict)
        child2_strategy = EnhancementStrategy(**c2_dict)
        
        # Create child individuals
        child1 = Individual(strategy=child1_strategy)
        child2 = Individual(strategy=child2_strategy)
        
        # Set parent IDs and generation
        child1.parent_ids = [parent1.id, parent2.id]
        child2.parent_ids = [parent1.id, parent2.id]
        child1.generation = max(parent1.generation, parent2.generation) + 1
        child2.generation = max(parent1.generation, parent2.generation) + 1
        
        return child1, child2
    
    def to_json(self) -> Dict[str, Any]:
        """Convert individual to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "strategy": asdict(self.strategy),
            "fitness": self.fitness,
            "age": self.age,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "prompt_result": self.prompt_result
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Individual':
        """Create an individual from a JSON dictionary."""
        # Create a new individual
        individual = cls(
            strategy=EnhancementStrategy(**data["strategy"]),
            id=data["id"]
        )
        individual.fitness = data["fitness"]
        individual.age = data["age"]
        individual.parent_ids = data["parent_ids"]
        individual.generation = data["generation"]
        individual.prompt_result = data.get("prompt_result")
        return individual
    
    def __str__(self) -> str:
        return f"Individual(id={self.id[:8]}, fitness={self.fitness.get('overall', 0):.3f}, gen={self.generation})"