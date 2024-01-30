import random

def run_ga_optimization(stock_data,generate_individual):
    # Other parameters

    population_size = 500
    num_generations = 10
    mutation_rate = 0.1
    elite_size = 80
    consecutive_threshold = 4  # Number of consecutive generations with no fitness improvement to consider saturation
    fitness_history = [] 

    # Initialize variables to track the best individual and its fitness score
    best_individual = None
    best_fitness_score = float('-inf')

    # Initialize population using known parameter values and randomly generated values
    population = [generate_individual() for _ in range(population_size)]
    strategy_metrics = {}


    for generation in range(num_generations):
        fitness_scores = [evaluate_strategy(stock_data.copy(), *individual)["Allocated Funds"] for individual in population]    
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

        # Store metrics for the best strategy in this generation
        best_strategy_metrics = evaluate_strategy(stock_data.copy(), *sorted_population[0])
        strategy_metrics[f"Generation {generation + 1}"] = best_strategy_metrics

            # Update best individual if a new one is found
        if fitness_scores[0] > best_fitness_score:
            best_individual = sorted_population[0]
            best_fitness_score = fitness_scores[0]

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