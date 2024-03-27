import random
from utils.ga_utils import *

def run_ga_optimization(stock_data, generate_individual, alpha_func, evaluate_strategy):
    # Other parameters
    population_size = 300
    num_generations = 7
    mutation_rate = 0.1  # not mutating
    elite_size = 100
    consecutive_threshold = 3  # Number of consecutive generations with no fitness improvement to consider saturation
    fitness_history = []

    # Initialize variables to track the best individual and its fitness score
    best_individual = None
    best_fitness_score = float('-inf')

    # Initialize population using known parameter values and randomly generated values
    population = [list(generate_individual(alpha_func).values()) for _ in range(population_size)]
    strategy_metrics = {}

    for generation in range(num_generations):
        
        evaluated_population = [evaluate_strategy(stock_data.copy(), alpha_func, *individual) for individual in population]

        # Extract fitness scores from the evaluated population
        fitness_scores = [individual["Allocated Funds"] for individual in evaluated_population]

        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

        # Store metrics for the best strategy in this generation
        best_strategy_metrics = evaluated_population[0]
        strategy_metrics[f"Generation {generation + 1}"] = best_strategy_metrics
        
        # Update best individual if a new one is found
        if fitness_scores[0] > best_fitness_score:
            best_individual = sorted_population[0]
            best_fitness_score = fitness_scores[0]

        print(best_fitness_score)
        new_population = []

        # Select the top K individuals (elite) for the next generation
        elite = sorted_population[:elite_size]

        # Check for convergence or other stopping criteria
        fitness_history.append(max(fitness_scores))

        # Check for convergence or other stopping criteria
        if len(fitness_history) >= consecutive_threshold:
            recent_improvements = fitness_history[-consecutive_threshold:]
            print(recent_improvements)
            if len(set(recent_improvements)) == 1:
                print("Saturation condition met, breaking the loop")
                break

        # Create new population by generating offspring from the elite individuals
        new_population = generate_offspring(elite, population_size, mutation_rate)

        population = new_population

    # Print metrics for the best strategy
    best_generation = max(strategy_metrics, key=lambda k: strategy_metrics[k]["Allocated Funds"])
    best_metrics = strategy_metrics[best_generation]

    print(f"\nBest Strategy Metrics (Generation {best_generation}):")
    print(f"\nBest Individual Parameters: {best_individual}")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value}")

    return best_metrics, best_individual


import random
from utils.ga_utils import *

def run_ga_optimization_updatedFor(stock_data, generate_individual, alpha_func, evaluate_strategy):
    # Other parameters
    population_size = 500
    num_generations = 10
    mutation_rate = 0.1  # not mutating
    elite_size = 80
    consecutive_threshold = 3  # Number of consecutive generations with no fitness improvement to consider saturation
    fitness_history = []

    # Initialize variables to track the best individual and its fitness score
    best_individual = None
    best_fitness_score = float('-inf')

    # Initialize population using known parameter values and randomly generated values
    population = [list(generate_individual(alpha_func).values()) for _ in range(population_size)]
    strategy_metrics = {}

    for generation in range(num_generations):
        
        evaluated_population = [evaluate_strategy(stock_data.copy(), alpha_func, *individual) for individual in population]

        # Extract fitness scores from the evaluated population
        fitness_scores = [individual["Allocated Funds"] for individual in evaluated_population]

        # Find the index of the best individual in the current generation
        best_index = max(range(len(fitness_scores)), key=fitness_scores.__getitem__)

        # Store metrics for the best strategy in this generation
        best_strategy_metrics = evaluated_population[best_index]
        strategy_metrics[f"Generation {generation + 1}"] = best_strategy_metrics

        # Update best individual if a new one is found
        if fitness_scores[best_index] > best_fitness_score:
            best_individual = population[best_index]
            best_fitness_score = fitness_scores[best_index]

        new_population = []

        # Select the top K individuals (elite) for the next generation
        elite = population[:elite_size]

        # Check for convergence or other stopping criteria
        fitness_history.append(max(fitness_scores))

        # Check for convergence or other stopping criteria
        if len(fitness_history) >= consecutive_threshold:
            recent_improvements = fitness_history[-consecutive_threshold:]
            print(recent_improvements)
            if len(set(recent_improvements)) == 1:
                print("Saturation condition met, breaking the loop")
                break

        # Create new population by generating offspring from the elite individuals
        new_population = generate_offspring(elite, population_size, mutation_rate)

        population = new_population

    # Print metrics for the best strategy
    best_generation = max(strategy_metrics, key=lambda k: strategy_metrics[k]["Allocated Funds"])
    best_metrics = strategy_metrics[best_generation]

    print(f"\nBest Strategy Metrics (Generation {best_generation}):")
    print(f"\nBest Individual Parameters: {best_individual}")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value}")

    return best_metrics, best_individual
