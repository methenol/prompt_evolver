"""
Evolution module for managing the evolutionary process.
"""

import os
import json
import copy # Added for deep copying
import asyncio
import random
import statistics
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt

from openai import OpenAI
from intent_analyzer import IntentAnalyzer
from fitness import FitnessEvaluator, EvaluationContext
from population import Population
from individual import Individual
from enhancement_strategy import EnhancementStrategy
from llm_breeder import LLMBreeder
from unified_fitness_metrics import UnifiedFitnessMetrics
from config import default_config

class Evolution:
    """Manages the evolutionary process for prompt enhancement strategies."""

    def __init__(
        self,
        client: OpenAI,
        intent_analyzer: IntentAnalyzer,
        output_dir: str = "output",
        save_frequency: int = 2,
        use_llm_breeding: bool = True,
        evaluation_type: str = default_config.evaluation_type # Add evaluation_type
    ):
        self.client = client
        self.intent_analyzer = intent_analyzer
        self.fitness_evaluator = FitnessEvaluator(client, intent_analyzer)
        self.population: Optional[Population] = None
        self.original_prompt: Optional[str] = None # Added
        self.generation = 0
        self.best_individual: Optional[Individual] = None # Tracks the reference to the best object found so far
        self.peak_individual_state: Optional[Dict[str, Any]] = None # Stores the state of the best individual at its peak
        self.history: List[Dict[str, Any]] = []
        self.output_dir = output_dir
        self.save_frequency = save_frequency
        self.use_llm_breeding = use_llm_breeding
        self.evaluation_type = evaluation_type # Store evaluation_type

        # Initialize metrics for fitness evaluation
        self.metrics = UnifiedFitnessMetrics()

        # Initialize LLM breeder if using LLM-based breeding
        if self.use_llm_breeding:
            self.llm_breeder = LLMBreeder(client)

    async def initialize(
        self,
        population_size: int = 20,
        seed_strategies: Optional[List[EnhancementStrategy]] = None
    ) -> None:
        """Initialize the evolution process."""
        # Create initial population
        strategies = seed_strategies if seed_strategies else []
        self.population = Population(size=population_size, init_strategies=strategies)
        self.generation = 0
        self.original_prompt = None # Reset on fresh initialize

    async def evolve(
        self,
        prompt: str,
        generations: int = 10,
        tournament_size: int = 3,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elite_size: int = 2
    ) -> Individual:
        """Run the evolutionary process."""
        if not self.population:
            raise ValueError("Population not initialized. Call initialize() first.")

        # Store original prompt if not already loaded
        if self.original_prompt is None:
            self.original_prompt = prompt
        elif prompt != self.original_prompt:
             # If a prompt is provided via args but we loaded one, use the loaded one.
             # Log a warning if a different prompt was also provided via args.
             if prompt:
                 print(f"Warning: Loaded state includes original prompt '{self.original_prompt}'. Ignoring provided prompt '{prompt}'.")
             prompt = self.original_prompt # Ensure we use the loaded original prompt

        # Analyze prompt intent using the original prompt
        intent_analysis = await self.intent_analyzer.analyze_multi_faceted(self.original_prompt)
        print(f"\nEvolutionary process starting/resuming with {len(self.population.individuals)} individuals")
        print("\nIntent Analysis:", json.dumps(intent_analysis, indent=2))

        # If using LLM breeding and it's the very first run (gen 0), initialize prompts
        if self.use_llm_breeding and self.generation == 0:
             await self._initialize_population_with_llm(self.original_prompt)

        # Track best individual
        best_fitness = self.best_individual.fitness.get("overall", 0) if self.best_individual else 0

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Calculate start and end generations for resuming
        start_generation = self.generation + 1
        end_generation = start_generation + generations -1 # Run for the specified number of *additional* generations
        print(f"Running from generation {start_generation} to {end_generation}")

        # Run for specified number of generations
        for gen in range(start_generation, end_generation + 1):
            self.generation = gen
            print(f"\n--- Generation {self.generation} ---")

            # Evaluate current population - use LLM evaluation if enabled
            if self.use_llm_breeding:
                await self._evaluate_population_with_llm(prompt)
            else:
                await self.fitness_evaluator.evaluate_population(
                    self.population,
                    prompt,
                    intent_analysis
                )

            # Apply niching to maintain diversity
            self.population.apply_niching(similarity_threshold=0.85)

            # Find best individual
            current_best = max(
                self.population.individuals,
                key=lambda ind: ind.fitness.get("overall", 0)
            )
            current_best_fitness = current_best.fitness.get("overall", 0)


            # Update best individual if improved
            if self.best_individual is None or current_best_fitness > best_fitness:
                # Update the reference and the peak fitness value
                self.best_individual = current_best
                best_fitness = current_best_fitness
                # Store the *state* of this peak individual
                self.peak_individual_state = copy.deepcopy(current_best.to_json())

                print(f"\nNew best individual state recorded (Gen {self.generation}):")
                print(f"  ID: {self.peak_individual_state['id'][:8]}")
                print(f"  Fitness: {best_fitness:.4f}")
                print(f"  Strategy: {self.peak_individual_state['strategy']['name']}")

                # Save the best individual's result (using the peak state)
                with open(os.path.join(self.output_dir, f"best_prompt_gen_{self.generation}.txt"), "w") as f:
                    f.write(f"# Best Prompt from Generation {self.generation}\n")
                    f.write(f"# Fitness: {best_fitness:.4f}\n")
                    f.write(f"# Strategy: {self.peak_individual_state['strategy']['name']}\n\n")
                    f.write(self.peak_individual_state.get('prompt_result', "") or "")
            # Log stats for this generation
            self._log_generation_stats()

            # Save state periodically
            if self.generation % self.save_frequency == 0:
                self.save_state(os.path.join(self.output_dir, f"evolution_state_gen_{self.generation}.json"))
                self.visualize_fitness_history(os.path.join(self.output_dir, f"fitness_history_gen_{self.generation}.png"))

            # Enhanced early stopping with multiple criteria
            if self._should_stop_early(best_fitness):
                break

            # Calculate adaptive parameters for next generation
            diversity = self.population.calculate_population_diversity()
            adaptive_mutation_rate = self._calculate_adaptive_mutation_rate(mutation_rate, diversity)
            adaptive_tournament_size = self.population.adaptive_tournament_size(tournament_size, diversity)
            
            print(f"Population diversity: {diversity:.3f}")
            print(f"Adaptive mutation rate: {adaptive_mutation_rate:.3f} (base: {mutation_rate:.3f})")
            print(f"Adaptive tournament size: {adaptive_tournament_size} (base: {tournament_size})")

            # Create next generation
            await self._create_next_generation(
                prompt=prompt,
                tournament_size=adaptive_tournament_size,
                mutation_rate=adaptive_mutation_rate,
                crossover_rate=crossover_rate,
                elite_size=elite_size
            )

        # No need for final evaluation - we've already evaluated each generation

        # Handle edge case: if no peak was recorded during loop (e.g., 1 gen run)
        # but a best individual exists, record it now
        if not self.peak_individual_state and self.best_individual:
            self.peak_individual_state = copy.deepcopy(self.best_individual.to_json())

        print("\n--- Evolution Complete ---")
        # Report based on the recorded peak state
        if self.peak_individual_state:
            print(f"Overall Best Individual (State Recorded at Peak):")
            print(f"  ID: {self.peak_individual_state['id'][:8]}")
            print(f"  Generation Found: {self.peak_individual_state['generation']}")
            print(f"  Strategy: {self.peak_individual_state['strategy']['name']}")
            print(f"  Peak Fitness: {best_fitness:.4f}") # Use the tracked best_fitness
        else:
             print("No best individual recorded.") # Should not happen in normal runs
        print(f"Best Fitness: {best_fitness:.4f}")

        # Save final results
        self.save_state(os.path.join(self.output_dir, "evolution_state_final.json"))
        self.visualize_fitness_history(os.path.join(self.output_dir, "fitness_history_final.png"))

        # Save the peak state information to the final file
        if self.peak_individual_state:
            with open(os.path.join(self.output_dir, "best_prompt_final.txt"), "w") as f:
                f.write(f"# Best Prompt (Final - State Recorded at Peak)\n")
                # Use the tracked best_fitness, as the score in peak_state might be slightly different due to float precision
                f.write(f"# Peak Fitness: {best_fitness:.4f}\n")
                f.write(f"# Strategy: {self.peak_individual_state['strategy']['name']}\n")
                f.write(f"# Generation Found: {self.peak_individual_state['generation']}\n\n")
                f.write(self.peak_individual_state.get('prompt_result', "") or "")
        else:
             print("Warning: No peak individual state recorded to save to best_prompt_final.txt")

        # Return an Individual object reconstructed from the peak state
        if self.peak_individual_state:
            # Reconstruct the Individual from the stored JSON state
            peak_individual = Individual.from_json(self.peak_individual_state)
            # Ensure the fitness reflects the actual peak score tracked
            peak_individual.fitness['overall'] = best_fitness
            return peak_individual
        else:
            # Fallback: return the last known best_individual reference if no peak state was ever recorded
            # This might happen in very short runs or if loading a state without peak info
            return self.best_individual

    async def _initialize_population_with_llm(self, original_prompt: str) -> None:
        """Initialize the population with LLM-generated prompts."""
        print("\nInitializing population with LLM-generated prompts...")

        # For each individual in the population
        tasks = []
        for individual in self.population.individuals:
            # Apply the individual's strategy to get the initial prompt
            task = self._initialize_individual_with_llm(individual, original_prompt)
            tasks.append(task)

        # Process initialization in batches to avoid rate limits
        batch_size = 5  # Process 5 initializations at a time (increased from 3)
        total_initialized = 0

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]

            # Execute batch of initializations
            await asyncio.gather(*batch)
            total_initialized += len(batch)

            # Add minimal delay between batches to respect rate limits
            if i + batch_size < len(tasks):
                await asyncio.sleep(0.2)  # 0.2 second delay between batches (reduced from 0.5)

        print(f"Initialized {total_initialized} individuals with LLM-generated prompts in {len(tasks) // batch_size + 1} batches")

    async def _initialize_individual_with_llm(self, individual: Individual, original_prompt: str) -> None:
        """Initialize a single individual with an LLM-generated prompt."""
        try:
            # Use the individual's strategy to generate a prompt
            if not individual.prompt_result:
                # Apply mutation to generate a new prompt
                mutated = await self.llm_breeder.mutate_prompt(
                    individual,
                    original_prompt,
                    mutation_strength=0.5  # Medium mutation for initialization
                )
                individual.prompt_result = mutated.prompt_result
        except Exception as e:
            print(f"Error initializing individual {individual.id}: {str(e)}")

    async def _evaluate_population_with_llm(self, original_prompt: str) -> None:
        """Evaluate the population using the LLM."""
        print("\nEvaluating population with LLM...")

        # Analyze prompt intent (moved here)
        intent_analysis = await self.intent_analyzer.analyze_multi_faceted(original_prompt)

        # Use the same evaluation method as non-LLM mode
        await self.fitness_evaluator.evaluate_population(
            self.population,
            original_prompt,
            intent_analysis # Pass intent_analysis here
        )
    
    def _calculate_adaptive_mutation_rate(self, base_rate: float, diversity: float) -> float:
        """Calculate adaptive mutation rate based on population diversity."""
        # Low diversity -> higher mutation rate (up to 2x base rate)
        # High diversity -> lower mutation rate (down to 0.5x base rate)
        min_rate = base_rate * 0.5
        max_rate = base_rate * 2.0
        
        # Inverse relationship: low diversity increases mutation rate
        adaptive_rate = min_rate + (max_rate - min_rate) * (1.0 - diversity)
        return max(min_rate, min(max_rate, adaptive_rate))
    
    def _get_convergence_metrics(self) -> Dict[str, float]:
        """Calculate multiple convergence metrics for sophisticated early stopping."""
        if not self.population or len(self.history) < 3:
            return {"fitness_stagnation": 0.0, "diversity": 1.0, "improvement_rate": 1.0}
        
        # Fitness stagnation: how long since last improvement
        recent_generations = min(5, len(self.history))
        recent_best_fitness = [gen.get("best_fitness", 0) for gen in self.history[-recent_generations:]]
        
        fitness_stagnation = 0.0
        if len(recent_best_fitness) > 1:
            improvements = sum(1 for i in range(1, len(recent_best_fitness)) 
                             if recent_best_fitness[i] > recent_best_fitness[i-1] + 0.001)
            fitness_stagnation = 1.0 - (improvements / (len(recent_best_fitness) - 1))
        
        # Population diversity
        diversity = self.population.calculate_population_diversity()
        
        # Improvement rate: slope of fitness over recent generations
        improvement_rate = 0.0
        if len(recent_best_fitness) >= 3:
            # Simple linear regression slope
            n = len(recent_best_fitness)
            x_sum = sum(range(n))
            y_sum = sum(recent_best_fitness)
            xy_sum = sum(i * recent_best_fitness[i] for i in range(n))
            x2_sum = sum(i * i for i in range(n))
            
            slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
            improvement_rate = max(0.0, slope)  # Only positive improvements
        
        return {
            "fitness_stagnation": fitness_stagnation,
            "diversity": diversity,
            "improvement_rate": improvement_rate
        }
    
    def _should_stop_early(self, best_fitness: float) -> bool:
        """Enhanced early stopping with multiple criteria."""
        # Original high fitness criterion
        if best_fitness > 0.95:
            return True
        
        # Get convergence metrics
        metrics = self._get_convergence_metrics()
        
        # Stop if fitness has stagnated for too long AND diversity is very low
        if (metrics["fitness_stagnation"] > 0.8 and 
            metrics["diversity"] < 0.1 and 
            self.generation > 5):
            print(f"\nEarly stopping: Population converged (stagnation: {metrics['fitness_stagnation']:.3f}, diversity: {metrics['diversity']:.3f})")
            return True
        
        # Stop if improvement rate is negligible and we're past minimum generations
        if (metrics["improvement_rate"] < 0.001 and 
            self.generation > 10 and 
            best_fitness > 0.7):
            print(f"\nEarly stopping: Improvement rate too low ({metrics['improvement_rate']:.6f})")
            return True
        
        return False
    
    def _select_parent(self, tournament_size: int, diversity: float) -> Individual:
        """Select parent using adaptive selection method based on population state."""
        # Use rank-based selection when diversity is low to encourage exploration
        # Use tournament selection when diversity is high to encourage exploitation
        if diversity < 0.3:
            return self.population.select_rank_based(selection_pressure=1.2)
        elif diversity > 0.7:
            return self.population.select_tournament(tournament_size)
        else:
            # Mixed strategy in middle range
            if random.random() < 0.5:
                return self.population.select_tournament(tournament_size)
            else:
                return self.population.select_rank_based(selection_pressure=1.5)

    async def _create_next_generation(
        self,
        prompt: str,
        tournament_size: int,
        mutation_rate: float,
        crossover_rate: float,
        elite_size: int
    ) -> None:
        """Create the next generation of individuals."""
        if not self.population:
            return

        print("\nCreating next generation...")

        offspring = []

        # Get elite individuals
        elite = self.population.get_elite(elite_size)

        # Always keep elite individuals
        offspring.extend(elite)

        # Fill the rest of the population with offspring using LLM breeding if enabled
        if self.use_llm_breeding:
            # Calculate current population diversity for adaptive selection
            current_diversity = self.population.calculate_population_diversity()
            
            # Create a list of tasks for concurrent breeding
            breeding_tasks = []

            # Calculate how many breeding operations we need
            num_breeding_ops = (self.population.size - len(offspring)) // 2
            additional_mutations = (self.population.size - len(offspring)) % 2

            # Create breeding tasks
            for _ in range(num_breeding_ops):
                # Select parents using adaptive selection strategy
                parent1 = self._select_parent(tournament_size, current_diversity)
                parent2 = self._select_parent(tournament_size, current_diversity)

                # Ensure we don't breed with self (try a few times)
                attempts = 0
                while parent1.id == parent2.id and attempts < 3:
                    parent2 = self._select_parent(tournament_size, current_diversity)
                    attempts += 1

                # Create breeding task
                if parent1.id != parent2.id and random.random() < crossover_rate:
                    # Crossover
                    task = self.llm_breeder.generate_offspring(parent1, parent2, prompt)
                    breeding_tasks.append(("crossover", task))
                else:
                    # Mutation (if we couldn't find different parents or by chance)
                    task = self.llm_breeder.mutate_prompt(
                        parent1, prompt, mutation_strength=mutation_rate
                    )
                    breeding_tasks.append(("mutation", task))

            # Add any needed mutation tasks
            for _ in range(additional_mutations):
                parent = self._select_parent(tournament_size, current_diversity)
                task = self.llm_breeder.mutate_prompt(
                    parent, prompt, mutation_strength=mutation_rate
                )
                breeding_tasks.append(("mutation", task))

            # Process breeding tasks in batches to avoid rate limits
            batch_size = 5  # Process 5 operations at a time (increased from 3)
            for i in range(0, len(breeding_tasks), batch_size):
                batch = breeding_tasks[i:i + batch_size]

                # Execute batch of tasks
                results = await asyncio.gather(*(task for _, task in batch))

                # Process batch results
                for j, (op_type, _) in enumerate(batch):
                    if op_type == "crossover":
                        child1, child2 = results[j]
                        offspring.append(child1)
                        offspring.append(child2)
                    else:  # mutation
                        child = results[j]
                        offspring.append(child)

                # Add minimal delay between batches to respect rate limits
                if i + batch_size < len(breeding_tasks):
                    await asyncio.sleep(0.2)  # 0.2 second delay between batches (reduced from 0.5)

        else:
            # Use traditional genetic algorithm operations with adaptive selection
            current_diversity = self.population.calculate_population_diversity()
            
            while len(offspring) < self.population.size:
                # Decide whether to do crossover
                if random.random() < crossover_rate and len(self.population.individuals) >= 2:
                    # Select parents using adaptive selection
                    parent1 = self._select_parent(tournament_size, current_diversity)
                    parent2 = self._select_parent(tournament_size, current_diversity)

                    # Ensure we don't crossover with self
                    attempts = 0
                    while parent1.id == parent2.id and attempts < 3:
                        parent2 = self._select_parent(tournament_size, current_diversity)
                        attempts += 1

                    if parent1.id != parent2.id:
                        # Perform crossover
                        child1, child2 = Individual.crossover(parent1, parent2)
                        offspring.append(child1)
                        if len(offspring) < self.population.size:
                            offspring.append(child2)
                    else:
                        # If we couldn't find different parents, just mutate one
                        child = parent1.mutate(mutation_rate)
                        offspring.append(child)
                else:
                    # Just mutate
                    parent = self._select_parent(tournament_size, current_diversity)
                    child = parent.mutate(mutation_rate)
                    offspring.append(child)

        # Make sure we don't exceed the population size
        offspring = offspring[:self.population.size]

        # Replace population with offspring
        self.population.replace_with_offspring(offspring, elite_size)

        print(f"Created new generation with {len(offspring)} individuals")

    def _log_generation_stats(self) -> None:
        """Log statistics for the current generation."""
        if not self.population or not self.population.individuals:
            return

        # Get fitness values
        fitness_values = [ind.fitness.get("overall", 0) for ind in self.population.individuals]

        # Calculate statistics
        stats = {
            "generation": self.generation,
            "max_fitness": max(fitness_values),
            "min_fitness": min(fitness_values),
            "avg_fitness": sum(fitness_values) / len(fitness_values),
            "median_fitness": statistics.median(fitness_values),
            "population_size": len(self.population.individuals),
            "best_id": max(self.population.individuals, key=lambda x: x.fitness.get("overall", 0)).id
        }

        # Print summary
        print(f"\nGeneration {stats['generation']} Stats:")
        print(f"Max Fitness: {stats['max_fitness']:.4f}")
        print(f"Avg Fitness: {stats['avg_fitness']:.4f}")
        print(f"Population Size: {stats['population_size']}")

        # Add to history
        self.history.append(stats)

    def save_state(self, filepath: str) -> None:
        """Save the current state of evolution to a file."""
        if not self.population:
            print("No population to save")
            return

        state = {
            "generation": self.generation,
            "original_prompt": self.original_prompt, # Added
            "population": self.population.to_json(),
            "history": self.history,
            "best_individual": self.best_individual.to_json() if self.best_individual else None, # Keep reference for potential resume logic
            "peak_individual_state": self.peak_individual_state # Save the peak state
        }

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)

            print(f"Evolution state saved to {filepath}")
        except Exception as e:
            print(f"Error saving evolution state: {str(e)}")

    def load_state(self, filepath: str) -> bool:
        """Load evolution state from a file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.generation = state.get("generation", 0)
            self.original_prompt = state.get("original_prompt") # Added
            self.history = state.get("history", [])

            population_data = state.get("population", [])
            self.population = Population.from_json(population_data)

            best_individual_data = state.get("best_individual")
            if best_individual_data:
                self.best_individual = Individual.from_json(best_individual_data)

            # Load the peak state as well
            self.peak_individual_state = state.get("peak_individual_state")

            print(f"Evolution state loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading evolution state: {str(e)}")
            return False

    def visualize_fitness_history(self, filepath: str) -> None:
        """Visualize fitness history over generations."""
        try:
            if not self.history:
                print("No history to visualize")
                return

            generations = [stats["generation"] for stats in self.history]
            max_fitness = [stats["max_fitness"] for stats in self.history]
            avg_fitness = [stats["avg_fitness"] for stats in self.history]

            plt.figure(figsize=(10, 6))
            plt.plot(generations, max_fitness, 'b-', label='Maximum Fitness')
            plt.plot(generations, avg_fitness, 'r-', label='Average Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness Evolution')
            plt.legend()
            plt.grid(True)

            # Save figure
            plt.savefig(filepath)
            plt.close()

            print(f"Fitness history visualized and saved to {filepath}")
        except ImportError:
            print("Matplotlib not available. Cannot visualize fitness history.")
        except Exception as e:
            print(f"Error visualizing fitness history: {str(e)}")