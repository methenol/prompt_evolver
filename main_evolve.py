"""
Main entry point for the evolutionary prompt enhancement system.
"""

import os
import asyncio
import argparse
from openai import OpenAI

from config import default_config
from intent_analyzer import IntentAnalyzer
from strategy_manager import StrategyManager
from evolution import Evolution

async def main():
    """Main function for the evolutionary prompt enhancement system."""
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Evolve prompts using evolutionary strategies")
    parser.add_argument("--prompt", type=str, help="Input prompt to evolve")
    parser.add_argument("--generations", type=int, default=default_config.num_generations, 
                        help="Number of generations to run")
    parser.add_argument("--population", type=int, default=default_config.population_size, 
                        help="Population size")
    parser.add_argument("--mutation", type=float, default=default_config.mutation_rate, 
                        help="Mutation rate")
    parser.add_argument("--crossover", type=float, default=default_config.crossover_rate, 
                        help="Crossover rate")
    parser.add_argument("--tournament", type=int, default=default_config.tournament_size, 
                        help="Tournament size")
    parser.add_argument("--elite", type=int, default=default_config.elite_size,
                        help="Number of elite individuals to keep")
    parser.add_argument("--load", type=str, help="Load evolution state from file")
    parser.add_argument("--output", type=str, default=default_config.output_dir, 
                        help="Output directory")
    parser.add_argument("--model", type=str, default=default_config.model_name,
                        help="Model name to use")
    parser.add_argument("--enable-llm-breeding", action="store_true", 
                        help="Enable LLM-based breeding (default: False)")
    parser.add_argument("--eval-type", type=str, default=default_config.evaluation_type,
                        choices=["llm", "rule", "combined"],
                        help="Evaluation type to use (llm, rule, combined)")
    
    args = parser.parse_args()
    
    # Update config from command line arguments
    config = default_config
    config.num_generations = args.generations
    config.population_size = args.population
    config.mutation_rate = args.mutation
    config.crossover_rate = args.crossover
    config.tournament_size = args.tournament
    config.elite_size = args.elite
    config.output_dir = args.output
    config.model_name = args.model
    config.evaluation_type = args.eval_type.lower() # Update config with eval_type
    
    try:
        # Set up OpenAI client
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        
        # Create intent analyzer
        intent_analyzer = IntentAnalyzer(client)
        
        # Create evolution manager
        evolution = Evolution(
            client, 
            intent_analyzer,
            output_dir=config.output_dir,
            save_frequency=config.save_frequency,
            use_llm_breeding=args.enable_llm_breeding,
            evaluation_type=config.evaluation_type # Pass evaluation_type
        )
        
        # Get initial strategies
        strategy_manager = StrategyManager()
        seed_strategies = strategy_manager.get_strategies()
        
        # If loading from a file
        if args.load:
            success = evolution.load_state(args.load)
            if not success:
                print(f"Failed to load state from {args.load}. Starting fresh.")
                await evolution.initialize(
                    population_size=config.population_size,
                    seed_strategies=seed_strategies
                )
            else:
                # Use the prompt from the loaded state if available
                if evolution.best_individual:
                    prompt = evolution.best_individual.prompt_result
                else:
                    prompt = args.prompt  # Fallback to command line prompt if not loaded
        else:
            # Initialize evolution
            await evolution.initialize(
                population_size=config.population_size,
                seed_strategies=seed_strategies
            )
            prompt = args.prompt
        
        if not prompt:
            prompt = input("Enter your prompt: ")
        
        print(f"\nEvolving prompt over {config.num_generations} generations...")
        print(f"Population size: {config.population_size}")
        print(f"Mutation rate: {config.mutation_rate}")
        print(f"Crossover rate: {config.crossover_rate}")
        print(f"Using LLM-based breeding: {args.enable_llm_breeding}")
        
        # Run the evolution
        best_individual = await evolution.evolve(
            prompt=prompt,
            generations=config.num_generations,
            tournament_size=config.tournament_size,
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate,
            elite_size=config.elite_size
        )
        
        # Display results
        print("\n--- Final Results ---")
        print(f"Best strategy: {best_individual.strategy.name}")
        print(f"Best fitness: {best_individual.fitness.get('overall', 0):.4f}")
        print("\nOriginal prompt:")
        print(prompt)
        print("\nEvolved prompt:")
        print(best_individual.prompt_result)
        
        # Print path to results
        print(f"\nDetailed results saved to: {os.path.abspath(config.output_dir)}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())