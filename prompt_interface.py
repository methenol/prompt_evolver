import gradio as gr
import asyncio
import os
import argparse
import traceback # Added for better error logging
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
        self.current_generation = 0 # This might become less relevant if loading state
        self.state_loaded_successfully = False # Flag to track loaded state

    async def initialize_evolution(self, population_size, use_llm_breeding, evaluation_type):
        """Initializes a fresh evolution process."""
        print("Initializing fresh evolution...")
        strategy_manager = StrategyManager()
        seed_strategies = strategy_manager.get_strategies()

        # Ensure output directory exists
        os.makedirs(default_config.output_dir, exist_ok=True)

        self.evolution = Evolution(
            self.client,
            self.intent_analyzer,
            output_dir=default_config.output_dir,
            save_frequency=default_config.save_frequency,
            use_llm_breeding=use_llm_breeding,
            evaluation_type=evaluation_type
        )

        await self.evolution.initialize(
            population_size=population_size,
            seed_strategies=seed_strategies
        )
        print("Initialization complete.")

    async def load_and_update_ui(self, load_state_file):
        """Attempts to load state from file and returns values to update UI."""
        if load_state_file is None:
            self.state_loaded_successfully = False
            # Return current default values + empty status if no file is provided
            # This prevents resetting UI if the user clears the file input
            return (
                gr.update(), # Keep existing prompt
                gr.update(interactive=True), # Re-enable population size
                gr.update(), # Keep existing mutation rate
                gr.update(), # Keep existing crossover rate
                gr.update(), # Keep existing tournament size
                gr.update(), # Keep existing elite size
                gr.update(), # Keep existing llm breeding checkbox
                gr.update(), # Keep existing evaluation type
                "File input cleared. Ready for fresh run or new file."
            )

        print(f"Attempting to load state from: {load_state_file.name}")
        try:
            # Instantiate evolution object if it doesn't exist
            # We need it to call load_state. Use dummy values initially.
            if self.evolution is None:
                 self.evolution = Evolution(
                     self.client, self.intent_analyzer,
                     output_dir=default_config.output_dir,
                     save_frequency=default_config.save_frequency,
                     use_llm_breeding=False,
                     evaluation_type="combined"
                 )

            success = self.evolution.load_state(load_state_file.name)

            if success:
                print("State loaded successfully.")
                self.state_loaded_successfully = True
                # Assuming evolution object has these attributes after load
                # Or potentially a config dictionary e.g., self.evolution.config
                # Adjust access as needed based on Evolution class implementation
                loaded_prompt = getattr(self.evolution, 'original_prompt', '')
                pop_size = getattr(self.evolution, 'population_size', default_config.population_size)
                mut_rate = getattr(self.evolution, 'mutation_rate', default_config.mutation_rate)
                cross_rate = getattr(self.evolution, 'crossover_rate', default_config.crossover_rate)
                tourn_size = getattr(self.evolution, 'tournament_size', default_config.tournament_size)
                elite_size = getattr(self.evolution, 'elite_size', default_config.elite_size)
                llm_breed = getattr(self.evolution, 'use_llm_breeding', False)
                eval_type = getattr(self.evolution, 'evaluation_type', default_config.evaluation_type)

                # Update the default config output dir based on loaded state if possible
                # This ensures results save to the same place if continuing a run
                if hasattr(self.evolution, 'output_dir'):
                    default_config.output_dir = self.evolution.output_dir

                status_msg = f"State loaded successfully from {os.path.basename(load_state_file.name)}. Adjust parameters if needed."
                return (
                    gr.update(value=loaded_prompt),
                    gr.update(value=pop_size, interactive=False),
                    gr.update(value=mut_rate),
                    gr.update(value=cross_rate),
                    gr.update(value=tourn_size),
                    gr.update(value=elite_size),
                    gr.update(value=llm_breed),
                    gr.update(value=eval_type),
                    status_msg
                )
            else:
                print("Failed to load state.")
                self.state_loaded_successfully = False
                return (
                    gr.update(), # Keep prompt
                    gr.update(interactive=True), # Re-enable population slider on failure
                    gr.update(), # Keep others
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    f"Error: Failed to load state from {os.path.basename(load_state_file.name)}. Check file format."
                )
        except Exception as e:
            print(f"Exception during state load: {e}")
            traceback.print_exc() # Print detailed traceback
            self.state_loaded_successfully = False
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                f"Error loading file: {str(e)}"
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
                          evaluation_type):
        """Handles the evolution process, either starting fresh or continuing from loaded state."""
        state_file_path = None
        plot_file_path = None
        output_dir = default_config.output_dir # Use current config output dir

        try:
            if self.state_loaded_successfully:
                print("Continuing evolution from loaded state...")
                # Ensure evolution object exists (should have been loaded)
                if not self.evolution:
                     raise RuntimeError("Evolution state was marked as loaded, but the object is missing.")
                # Update parameters on the loaded evolution object from UI values
                print(f"Updating loaded evolution object: LLM Breeding={use_llm_breeding}, Eval Type={evaluation_type}")
                self.evolution.use_llm_breeding = use_llm_breeding
                self.evolution.evaluation_type = evaluation_type
                # Reset flag so next run is fresh unless loaded again
                self.state_loaded_successfully = False
                # Use the output directory from the loaded state
                output_dir = self.evolution.output_dir
                print(f"Using output directory from loaded state: {output_dir}")

            else:
                print("Starting fresh evolution run...")
                # Initialize based on current UI settings
                await self.initialize_evolution(population_size, use_llm_breeding, evaluation_type)
                output_dir = self.evolution.output_dir # Get output dir from new instance
                print(f"Using output directory: {output_dir}")


            # Reset current generation counter (might be useful for progress tracking later)
            self.current_generation = 0

            # Run the evolution process
            best_individual = await self.evolution.evolve(
                prompt=prompt, # Use prompt from UI (might be the loaded one or newly entered)
                generations=num_generations,
                tournament_size=tournament_size,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elite_size=elite_size
                # Note: use_llm_breeding and evaluation_type were set during init/load
            )

            # Prepare results
            enhanced_prompt = best_individual.prompt_result
            strategy_used = best_individual.strategy.name
            fitness_score = f"{best_individual.fitness.get('overall', 0):.4f}"
            abs_output_dir = os.path.abspath(output_dir)

            status_text = (f"Evolution Complete!\n"
                           f"Strategy: {strategy_used}\n"
                           f"Fitness: {fitness_score}\n\n"
                           f"Results saved in: {abs_output_dir}")

            # Define expected output file paths
            state_file_path = os.path.join(output_dir, "evolution_state_final.json")
            plot_file_path = os.path.join(output_dir, "fitness_history_final.png")

            # Check if files exist, set to None if not
            if not os.path.exists(state_file_path):
                print(f"Warning: Expected state file not found at {state_file_path}")
                state_file_path = None
            if not os.path.exists(plot_file_path):
                print(f"Warning: Expected plot file not found at {plot_file_path}")
                plot_file_path = None

            return (enhanced_prompt, status_text, state_file_path, plot_file_path)

        except Exception as e:
            print(f"Error during evolution: {str(e)}")
            traceback.print_exc() # Print detailed traceback
            error_message = f"Error occurred during evolution: {str(e)}"
            return ("", error_message, None, None)

    def create_interface(self):
        with gr.Blocks() as interface:
            gr.Markdown("# Prompt Evolution System")

            with gr.Row():
                with gr.Column(scale=2):
                    # --- Load State ---
                    load_state_input = gr.File(
                        label="Load Evolution State (Optional .json)",
                        file_types=[".json"],
                        type="filepath" # Use filepath to get the path
                    )

                    # --- Input Prompt ---
                    input_prompt = gr.Textbox(
                        label="Input Prompt",
                        placeholder="Enter the prompt you want to enhance, or load state...",
                        lines=5
                    )

                    # --- Parameters ---
                    gr.Markdown("### Evolution Parameters")
                    with gr.Row():
                        num_generations = gr.Slider(
                            minimum=1, maximum=20, value=default_config.num_generations,
                            step=1, label="Number of Generations"
                        )
                        population_size = gr.Slider(
                            minimum=4, maximum=50, value=default_config.population_size,
                            step=2, label="Population Size"
                        )
                    with gr.Row():
                        mutation_rate = gr.Slider(
                            minimum=0.0, maximum=1.0, value=default_config.mutation_rate,
                            label="Mutation Rate"
                        )
                        crossover_rate = gr.Slider(
                            minimum=0.0, maximum=1.0, value=default_config.crossover_rate,
                            label="Crossover Rate"
                        )
                    with gr.Row():
                        tournament_size = gr.Slider(
                            minimum=2, maximum=10, value=default_config.tournament_size,
                            step=1, label="Tournament Size"
                        )
                        elite_size = gr.Slider(
                            minimum=0, maximum=5, value=default_config.elite_size,
                            step=1, label="Elite Size"
                        )

                    use_llm_breeding = gr.Checkbox(
                        label="Enable LLM-based Breeding",
                        value=False # Default to False unless loaded
                    )

                    evaluation_type_dropdown = gr.Dropdown(
                        label="Evaluation Type",
                        choices=["llm", "rule", "combined"],
                        value=default_config.evaluation_type,
                        interactive=True
                    )

                    evolve_button = gr.Button("Start Evolution", variant="primary")

                with gr.Column(scale=1):
                    # --- Outputs ---
                    gr.Markdown("### Results")
                    output_prompt = gr.Textbox(
                        label="Enhanced Prompt",
                        lines=5,
                        interactive=False
                    )
                    status_output = gr.Textbox(
                        label="Status / Logs",
                        lines=5,
                        interactive=False,
                        placeholder="Status updates will appear here..."
                    )
                    # --- Download Links ---
                    gr.Markdown("#### Download Results")
                    output_state_file = gr.File(
                        label="Final State (.json)",
                        interactive=False
                        )
                    output_plot_file = gr.File(
                        label="Fitness Plot (.png)",
                        interactive=False
                        )

            # --- Event Handlers ---
            # Handle state loading
            load_state_input.upload(
                fn=lambda file: asyncio.run(self.load_and_update_ui(file)),
                inputs=[load_state_input],
                outputs=[
                    input_prompt,           # Update prompt box
                    population_size,        # Update sliders/inputs
                    mutation_rate,
                    crossover_rate,
                    tournament_size,
                    elite_size,
                    use_llm_breeding,       # Update checkbox
                    evaluation_type_dropdown, # Update dropdown
                    status_output
                    # Note: num_generations is intentionally left out - user sets how many *more* generations to run
                ]
            )

            # Handle evolution start
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
                    evaluation_type_dropdown
                ],
                outputs=[
                    output_prompt,
                    status_output,
                    output_state_file,
                    output_plot_file
                    ]
            )

        return interface

def main():
    parser = argparse.ArgumentParser(description='Prompt Evolution Interface')
    parser.add_argument('--enable-public', action='store_true',
                      help='Enable public link sharing')
    args = parser.parse_args()

    ui = PromptEvolutionUI()
    interface = ui.create_interface()
    print("Launching Gradio interface...")
    interface.launch(share=args.enable_public, server_name="0.0.0.0", server_port=7861)

if __name__ == "__main__":
    main()