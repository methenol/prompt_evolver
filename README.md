# Prompt Evolver

An AI-powered evolutionary system for enhancing and optimizing prompts through iterative refinement.

## Overview

Prompt Evolver uses evolutionary algorithms combined with LLM evaluation to automatically improve prompts through generations of refinement. The system analyzes your initial prompt, applies various enhancement strategies, and evolves the prompt over multiple generations to maximize effectiveness.

## Features

- **Evolutionary Optimization**: Uses genetic algorithms to evolve prompts over multiple generations
- **Intent Analysis**: Automatically identifies the intent behind your prompts
- **Strategy Selection**: Applies different enhancement strategies based on prompt type
- **LLM-Based Breeding**: Optional advanced breeding techniques using language models
- **Progress Tracking**: Saves evolution state, best prompts, and fitness history

## Installation

First, ensure you have Python 3.10+ installed on your system.

### Option 1: Using conda
```bash
conda create -n prompt_evolver python=3.10
conda activate prompt_evolver
pip install -r requirements.txt
```

### Option 2: Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## Using the Graphical Interface

You can use our Gradio-based web interface for a more user-friendly experience:

```bash
python prompt_interface.py
```

This will start a local web interface where you can:
- Input your prompt and adjust all evolution parameters
- Start the evolution process with a single click
- View real-time status updates
- See the enhanced prompt results
- Access all evolution parameters through intuitive sliders and inputs

To enable public link sharing (e.g., for team collaboration), use:
```bash
python prompt_interface.py --enable-public
```

The interface provides the same capabilities as the command line version but with an intuitive visual interface that makes it easier to:
- Experiment with different parameter combinations
- View results immediately
- Adjust settings without remembering command line arguments
- Monitor the evolution process in real-time

## Configuration

Before running the application, make sure you have:

1. Create a .env file or rename .env.example
2. Set up your OpenAI API key in an environment variable or in the .env file
3. Configured any custom settings in the .env file if needed

For OpenAI compatible endpoints, specify a BASE_URL ex. http://localhost:11434/v1

## Usage

Run the main evolution script with your initial prompt:

```bash
python main_evolve.py --prompt "Your initial prompt here"
```

Or run it interactively and input your prompt when prompted:

```bash
python main_evolve.py
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--prompt` | Input prompt to evolve | (Interactive prompt) |
| `--generations` | Number of generations to run | 20 |
| `--population` | Population size | 30 |
| `--mutation` | Mutation rate | 0.3 |
| `--crossover` | Crossover rate | 0.7 |
| `--tournament` | Tournament size for selection | 3 |
| `--elite` | Number of elite individuals to keep | 2 |
| `--load` | Load evolution state from file | None |
| `--output` | Output directory | ./output |
| `--model` | OpenAI model to use | gpt-4o-mini |
| `--enable-llm-breeding` | Enable LLM-based breeding | False |
| `--eval-type` | Evaluation type (llm, rule, combined) | combined |

### Examples

Basic usage:
```bash
python main_evolve.py --prompt "Create a marketing tagline for an eco-friendly water bottle"
```

Advanced usage with custom parameters:
```bash
python main_evolve.py --prompt "Write a Python function to calculate Fibonacci numbers" --generations 30 --population 50 --mutation 0.4 --enable-llm-breeding
```

Continue from a previous run (expiramental):
```bash
python main_evolve.py --load ./output/evolution_state_gen_10.json
```

## Output Files

The system saves various files to the output directory:

- `evolution_state_gen_X.json`: Evolution state after X generations
- `evolution_state_final.json`: Final evolution state
- `fitness_history_gen_X.png`: Fitness evolution graphs
- `fitness_history_final.png`: Final evolution graph
- `best_prompt_gen_X.txt`: Best prompt after X generations
- `best_prompt_final.txt`: Final best prompt

## How It Works

1. **Initialization**: The system analyzes your prompt and selects appropriate enhancement strategies
2. **Population Creation**: Creates a diverse population of prompt variations
3. **Evaluation**: Each prompt variant is evaluated for fitness based on multiple criteria
4. **Selection**: The best prompts are selected for breeding
5. **Breeding**: New prompt variants are created through crossover and mutation
6. **Evolution**: The process repeats over multiple generations
7. **Refinement**: Optional LLM-based breeding can be used for more sophisticated prompt refinement
