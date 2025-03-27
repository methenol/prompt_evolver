import gradio as gr
import asyncio
import os
import argparse
from config import default_config
from intent_analyzer import IntentAnalyzer
from strategy_manager import StrategyManager
from evolution import Evolution
from openai import OpenAI

class PromptEvolutionUI:
    def __init__(self):
        self.client = OpenAI(api_key=default_config.api_key, base_url=default_config.base_url)
        self.intent_analyzer = IntentAnalyzer(self.client)
        self.evolution = None
        self.current_generation = 0
        
    async def initialize_evolution(self, population_size, use_llm_breeding, evaluation_type):
        strategy_manager = StrategyManager()
        seed_strategies = strategy_manager.get_strategies()
        
        self.evolution = Evolution(
            self.client,
            self.intent_analyzer,
            output_dir=default_config.output_dir,
            save_frequency=default_config.save_frequency,
            use_llm_breeding=use_llm_breeding
        )
        
        await self.evolution.initialize(
            population_size=population_size,
            seed_strategies=seed_strategies
        )
        
    async def evolve_prompt(self,
                          prompt,
                          num_generations,
                          population_size,
                          mutation_rate,
                          crossover_rate,
                          tournament_size,
                          elite_size,
                          use_llm_breeding,
                          evaluation_type): # Add new parameter
        
        try:
            # Always reinitialize the evolution with the new settings
            await self.initialize_evolution(population_size, use_llm_breeding, evaluation_type)
            
            # Reset current generation counter
            self.current_generation = 0
            
            best_individual = await self.evolution.evolve(
                prompt=prompt,
                generations=num_generations,
                tournament_size=tournament_size,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elite_size=elite_size
            )
            
            result = {
                "Original Prompt": prompt,
                "Enhanced Prompt": best_individual.prompt_result,
                "Strategy Used": best_individual.strategy.name,
                "Fitness Score": f"{best_individual.fitness.get('overall', 0):.4f}",
                "Output Directory": os.path.abspath(default_config.output_dir)
            }
            
            return (result["Enhanced Prompt"], 
                   f"Strategy: {result['Strategy Used']}\nFitness: {result['Fitness Score']}\n\nResults saved in: {result['Output Directory']}")
            
        except Exception as e:
            return "", f"Error occurred: {str(e)}"

    def create_interface(self):
        with gr.Blocks() as interface:
            gr.Markdown("# Prompt Evolution System")
            
            with gr.Row():
                with gr.Column():
                    input_prompt = gr.Textbox(
                        label="Input Prompt",
                        placeholder="Enter the prompt you want to enhance...",
                        lines=3
                    )
                    
                    with gr.Row():
                        num_generations = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=default_config.num_generations,
                            step=1,
                            label="Number of Generations"
                        )
                        population_size = gr.Slider(
                            minimum=4,
                            maximum=50,
                            value=default_config.population_size,
                            step=2,
                            label="Population Size"
                        )
                    
                    with gr.Row():
                        mutation_rate = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=default_config.mutation_rate,
                            label="Mutation Rate"
                        )
                        crossover_rate = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=default_config.crossover_rate,
                            label="Crossover Rate"
                        )
                    
                    with gr.Row():
                        tournament_size = gr.Slider(
                            minimum=2,
                            maximum=10,
                            value=default_config.tournament_size,
                            step=1,
                            label="Tournament Size"
                        )
                        elite_size = gr.Slider(
                            minimum=0,
                            maximum=5,
                            value=default_config.elite_size,
                            step=1,
                            label="Elite Size"
                        )
                    
                    use_llm_breeding = gr.Checkbox(
                        label="Enable LLM-based Breeding",
                        value=False
                    )
                    
                    evaluation_type_dropdown = gr.Dropdown(
                        label="Evaluation Type",
                        choices=["llm", "rule", "combined"],
                        value=default_config.evaluation_type,
                        interactive=True
                    )
                    
                    evolve_button = gr.Button("Start Evolution")
                
                with gr.Column():
                    output_prompt = gr.Textbox(
                        label="Enhanced Prompt",
                        lines=3,
                        interactive=False
                    )
                    status_output = gr.Textbox(
                        label="Status",
                        lines=3,
                        interactive=False
                    )
            
            evolve_button.click(
                fn=lambda *args: asyncio.run(self.evolve_prompt(*args)),
                inputs=[
                    input_prompt,
                    num_generations,
                    population_size,
                    mutation_rate,
                    crossover_rate,
                    tournament_size,
                    elite_size,
                    use_llm_breeding,
                    evaluation_type_dropdown # Add new input
                ],
                outputs=[output_prompt, status_output]
            )
            
        return interface

def main():
    parser = argparse.ArgumentParser(description='Prompt Evolution Interface')
    parser.add_argument('--enable-public', action='store_true', 
                      help='Enable public link sharing')
    args = parser.parse_args()

    ui = PromptEvolutionUI()
    interface = ui.create_interface()
    interface.launch(share=args.enable_public, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()