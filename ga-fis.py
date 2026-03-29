import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(42)  # for reproducibility

# Generating 1600 random points in (-π,π)
x = np.random.uniform(-np.pi, np.pi, 1600)
y = np.random.uniform(-np.pi, np.pi, 1600)

# compute output
f = np.sin(x) * np.cos(y)

# stack and split data

data = np.column_stack((x, y, f))
np.random.shuffle(data)  # shuffin'
train_data = data[0:1280]
test_data = data[1280:1600]

# building the GA
# note I am using gaussian membership functions, so each MF is defined by two parameters: mean (center) and standard deviation   (width).
n_mf = 7
n_inputs = 2
n_rules = n_mf**n_inputs
chrom_length = n_inputs * n_mf * 2 + n_rules
n_generations = 1000
mutation_rate = 0.01
crossover_rate = 0.20

# population
pop_size = 300
mf_params = np.random.uniform(-np.pi, np.pi, (pop_size, n_inputs * n_mf * 2))
rule_outputs = np.random.uniform(-1, 1, (pop_size, n_rules))
population = np.column_stack([mf_params, rule_outputs])


# the gaussian stuff

def gaussian_mf(x, center, sigma):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)

# decoding the chromosome


def decode_chromosome(chromosome):
    mf_params = chromosome[:n_inputs * n_mf * 2].reshape(n_inputs, n_mf, 2)
    rule_outputs = chromosome[n_inputs * n_mf * 2:].reshape(n_rules)
    return mf_params, rule_outputs


def fuzzy_inference(x, y, mf_params, rule_outputs):
    # fuzzifyyyyyyy

    mu_x = [gaussian_mf(x, mf_params[0, i, 0], mf_params[0, i, 1])
            for i in range(n_mf)]
    mu_y = [gaussian_mf(y, mf_params[1, j, 0], mf_params[1, j, 1])
            for j in range(n_mf)]

    # firin' my lazers

    firing_strengths = []
    for i in range(n_mf):
        for j in range(n_mf):
            strength = mu_x[i] * mu_y[j]
            firing_strengths.append(strength)

    # defuzzifyyyyyyy

    numerator = sum(s * r for s, r in zip(firing_strengths, rule_outputs))
    denominator = sum(firing_strengths)

    # we can't destroy the universe by dividing by zero!
    return numerator / (denominator + 1e-9)


# Fitness Function Time
def fitnesses(chromosome):
    mf_params, rule_outputs = decode_chromosome(chromosome)
    X_train = train_data[:, 0]
    Y_train = train_data[:, 1]
    F_actual = train_data[:, 2]

# fuzzy inference runnin'

    predictions = [fuzzy_inference(x, y, mf_params, rule_outputs)
                   for x, y in zip(X_train, Y_train)]
# RSME calculation

    rmse = np.sqrt(np.mean((np.array(predictions) - F_actual)**2))
    return rmse
# GA Details
# bracket style tournament


def tournament_selection(population, fitnesses, k=2):
    idx = np.random.choice(pop_size, k, replace=False)
    best = idx[np.argmin([fitnesses[i] for i in idx])]
    return population[best].copy()

# crossover


def crossover(parent1, parent2):
    child = np.empty_like(parent1)
    for i in range(len(parent1)):
        lo = min(parent1[i], parent2[i])
        hi = max(parent1[i], parent2[i])
        child[i] = np.random.uniform(lo, hi)
    return child


def mutate(chromosome, mutation_rate):
    mutant = chromosome.copy()
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            mutant[i] += np.random.normal(0, 0.5)

    # clip centers and widths separately
    n_mf_params = n_inputs * n_mf * 2

    # clip centers to [-π, π]
    for i in range(0, n_mf_params, 2):
        mutant[i] = np.clip(mutant[i], -np.pi, np.pi)

    # clip widths to [0.1, 2.0] ← key fix!
    for i in range(1, n_mf_params, 2):
        mutant[i] = np.clip(mutant[i], 0.1, 2.0)

    # clip rule outputs to [-1, 1]
    mutant[n_mf_params:] = np.clip(mutant[n_mf_params:], -1, 1)

    return mutant


def run_ga():
    global population
    best_chromosome = None
    best_fitness = float('inf')

    for gen in tqdm(range(n_generations), desc="Training GA"):
        fitness = [fitnesses(c) for c in population]
        # elitism found here... we keep best solution from current gen and add it to next gen
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_chromosome = population[best_idx].copy()

        sorted_pop = sorted(zip(population, fitness), key=lambda x: x[1])
        next_pop = [chrom.copy() for chrom, _ in sorted_pop[:3]]
        while len(next_pop) < pop_size:
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            if np.random.rand() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()
            child = mutate(child, mutation_rate)
            next_pop.append(child)

        population = next_pop
        tqdm.write(
            f"Gen {gen+1}/{n_generations} | Best RMSE: {best_fitness:.4f}")

    return best_chromosome, best_fitness


# run the GA
best_chromosome, best_fitness = run_ga()
print(f"Final RMSE: {best_fitness:.4f}")

# validate on test data
mf_params_best, rule_outputs_best = decode_chromosome(best_chromosome)
X_test = test_data[:, 0]
Y_test = test_data[:, 1]
F_test = test_data[:, 2]

test_preds = [fuzzy_inference(x, y, mf_params_best, rule_outputs_best)
              for x, y in zip(X_test, Y_test)]

test_rmse = np.sqrt(np.mean((np.array(test_preds) - F_test)**2))
print(f"Validation RMSE: {test_rmse:.4f}")

# plots
x_plot = np.linspace(-np.pi, np.pi, 50)
y_plot = np.linspace(-np.pi, np.pi, 50)
X_grid, Y_grid = np.meshgrid(x_plot, y_plot)

Z_actual = np.sin(X_grid) * np.cos(Y_grid)

Z_fis = np.array([[fuzzy_inference(xi, yi, mf_params_best, rule_outputs_best)
                   for xi in x_plot] for yi in y_plot])

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X_grid, Y_grid, Z_actual, cmap='viridis')
ax1.set_title('Actual: sin(x)cos(y)')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X_grid, Y_grid, Z_fis, cmap='viridis')
ax2.set_title('FIS Approximation')

plt.tight_layout()
plt.savefig('fis_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
