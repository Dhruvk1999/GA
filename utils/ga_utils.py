import random
from inspect import signature


def generate_offspring(elite,population_size,mutation_rate):
    new_population = []
    elite_size = len(elite)
    
    # Create new individuals by performing uniform crossover and mutation
    while len(new_population) < population_size:
        parent1 = random.choice(elite)
        parent2 = random.choice(elite)
        
        # Perform uniform crossover
        child = uniform_crossover(parent1, parent2)
        
        # Apply mutation
        if random.random() < mutation_rate:
            child = mutate(child)
        
        new_population.append(child)
    
    return new_population

def uniform_crossover(parent1, parent2):
    child = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.choice([True, False]):
            child.append(gene1)
        else:
            child.append(gene2)
    return child

def mutate(individual):
    # ...
    return individual

def generate_individual(alpha_function):
    # Get the parameters accepted by the alpha function
    alpha_params = signature(alpha_function).parameters

    # Define parameter ranges for alpha functions
    parameter_ranges = {
        'param1': (2, 20),
        'param2': (3, 20),
        'threshold': (0, 1),
        'fast_period': (2, 10),
        'stop_loss_percentage': (-1, 2),
        'power': (1, 5),
        'argmax_window': (3, 10),
        'delta_lookback': (2, 10),
        'window_corr': (3, 15),
        'ts_rank_lookback' : (3,15),
        'vwap_lookback': (3,20),
        'adv_window': (3,20),
        'window_sum': (3,10),
        'window_delay': (3,15),
        'window_delta':(3,10),
        'window_volume':(2,10),
        'window_covariance':(2,10),
        'window_returns':(2,10),
        'window_covariance':(2,10),
        'window_rank':(2,5),
        'window_ts_rank':(5,15),
        'window_ts_rank_volume':(3,15),
        'window_std':(3,15),

        # Add more parameters as needed for other alpha functions
    }

    # Generate individual with random parameters within specified ranges
    individual = {param: random.randint(param_range[0], param_range[1]) for param, param_range in parameter_ranges.items() if param in alpha_params}

    individual['threshold'] = random.randint(parameter_ranges['threshold'][0],parameter_ranges['threshold'][1])
    individual['stop_loss_percentage'] = random.uniform(parameter_ranges['stop_loss_percentage'][0],parameter_ranges['stop_loss_percentage'][1])
    return individual