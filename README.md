# GA-FIS: Genetic Algorithm-Optimized Fuzzy Inference System

A Fuzzy Inference System (FIS) built entirely from scratch in Python, where membership function parameters and rule outputs are optimized using a custom Genetic Algorithm. No fuzzy libraries. No black-box optimizers. Pure NumPy.

Developed as part of graduate research in soft computing at the University of Cincinnati AI BIO Lab under Dr. Kelly Cohen.

---

## The Problem

Approximate the function **F(x,y) = sin(x)·cos(y)** over the domain [-π, π] × [-π, π] using a 2-input, 1-output Fuzzy Inference System whose structure — membership function parameters and rule outputs — is entirely determined by a Genetic Algorithm.

The challenge: placing membership function centers and widths by hand for a smooth, continuous 2D function is impractical. The search space is enormous, the function has many local minima and maxima, and there is no obvious expert knowledge to guide placement. This is exactly the kind of problem GAs are built for.

---

## Results

| Metric | Value |
|--------|-------|
| Best training RMSE | 0.0373 |
| Best validation RMSE | 0.0386 |
| Membership functions per input | 7 |
| Total rules | 49 |
| Generations | 1000 |
| Population size | 300 |

![FIS vs Actual](results/fis_comparison.png)

---

## Hyperparameter Exploration

13 trials were run systematically to find the best configuration. Key findings are documented below.

| Trial | pop_size | n_mf | n_generations | crossover | mutation | elitism | seed | RMSE |
|-------|----------|------|---------------|-----------|----------|---------|------|------|
| 1 | 75 | 3 | 300 | 0.09 | 0.01 | 1 | random | 0.5611 |
| 2 | 150 | 5 | 400 | 0.09 | 0.01 | 1 | random | 0.1356 |
| 3 | 200 | 7 | 500 | 0.09 | 0.01 | 1 | random | 0.1228 |
| 4 | 200 | 7 | 500 | 0.08 | 0.02 | 1 | random | 0.1349 |
| 5 | 200 | 7 | 700 | 0.09 | 0.01 | 1 | random | 0.0924 |
| 6 | 200 | 5 | 200 | 0.10 | 0.10 | 1 | random | 0.1935 |
| 7 | 200 | 5 | 200 | 0.075 | 0.05 | 1 | random | 0.1224 |
| 8 | 300 | 5 | 200 | 0.05 | 0.075 | 1 | random | 0.1735 |
| 9 | 300 | 7 | 300 | 0.09 | 0.01 | 5 | 42 | 0.1437 |
| 12 | 300 | 7 | 700 | 0.20 | 0.02 | 3 | 42 | 0.0453 |
| **13** | **300** | **7** | **1000** | **0.20** | **0.01** | **3** | **42** | **0.0373** |

**Key findings from hyperparameter search:**

- **n_mf=7 is non-negotiable** — trials with n_mf=5 consistently underperformed regardless of other settings. More membership functions means more expressive rules and a smoother approximation.
- **Crossover rate matters more than expected** — the original crossover rate of 0.09 was effectively no crossover for a blend crossover operator. Raising to 0.20 unlocked significantly better performance. Note: standard GA literature recommends 0.6-0.9 for discrete chromosomes, but blend crossover on continuous values behaves differently — 0.20 outperformed 0.95.
- **Elitism stabilizes convergence** — increasing elitism from 1 to 3 prevented good solutions from being lost to random noise, producing the characteristic staircase convergence pattern where the best fitness holds steady then improves in steps.
- **Fixed random seed enables reproducibility** — locking np.random.seed(42) ensures the same result every run, critical for a portfolio piece.
- **More generations keep paying off** — the GA was still improving at gen 628 of trial 12, which motivated pushing to 1000 generations in trial 13. Diminishing returns appear after ~800 gens.
- **Training/validation gap is tight** — 0.0373 vs 0.0386 indicates the GA learned the underlying function rather than memorizing training data.

---

## How It Works

```
Dataset: 1600 random points over [-π, π] × [-π, π]
   │
   ├── 1280 training / 320 test (80/20 split)
   │
   ▼
Population of chromosomes (size 300)
Each chromosome encodes:
   ├── 2 inputs × 7 MFs × 2 params (center, width) = 28 values
   └── 7² = 49 rule output values
   │
   ▼
For each chromosome:
   ├── Decode → MF parameters + rule outputs
   ├── Fuzzify inputs using Gaussian MFs
   ├── Compute 49 firing strengths (product T-norm)
   ├── Defuzzify via weighted average (TSK zeroth-order)
   └── Fitness = RMSE on training set
   │
   ▼
GA operators:
   ├── Tournament selection (k=2)
   ├── Blend crossover (rate 0.20)
   ├── Gaussian mutation (rate 0.01) with boundary clipping
   └── Elitism (top 3 carried unchanged)
   │
   ▼
Repeat for 1000 generations → best chromosome = trained FIS
```

---

## System Design

**Membership functions — Gaussian**

μ(x) = exp(-0.5 × ((x - c) / σ)²)

Gaussian MFs were chosen for three reasons: they are smooth and continuous like the target function, they only require 2 parameters per MF (center c and width σ), and unlike triangular MFs they have no ordering constraints which simplifies the chromosome encoding.

**Rule base — TSK zeroth-order**

F_output = Σ(sᵢ × rᵢ) / Σ(sᵢ)

Where sᵢ is the firing strength of rule i and rᵢ is the crisp rule output evolved by the GA. 49 rules total (7² from the two inputs). A small epsilon (1e-9) prevents division by zero. TSK was chosen over Mamdani for computational efficiency — it produces a crisp output directly without a separate defuzzification step.

**Chromosome encoding**

Each chromosome is a flat array of length 77:
- Indices 0–27: MF parameters (center, width) for both inputs across 7 MFs each
- Indices 28–76: 49 rule output values initialized in [-1, 1] to match the output range of sin(x)·cos(y)

**GA operators**

- Tournament selection with k=2 — balanced selection pressure, weaker chromosomes still have a chance, maintains diversity
- Blend crossover — children explore between and slightly beyond parent bounds, enabling the GA to find solutions outside current population limits
- Gaussian mutation with clipping — centers clipped to [-π, π], widths to [0.1, 2.0], rule outputs to [-1, 1]
- Elitism (top 3) — best chromosomes always carried forward, the GA never loses its best solutions

---

## Quickstart

**Clone**
```bash
git clone https://github.com/JetHayes/ga-fuzzy-inference-system.git
cd ga-fuzzy-inference-system
```

**Install**
```bash
pip install -r requirements.txt
```

**Run**
```bash
python ga_fis.py
```

Training progress prints per generation. On completion, a surface comparison plot is saved to `results/fis_comparison.png`.

Expected output:
```
Gen 1/1000 | Best RMSE: 0.4xxx
Gen 2/1000 | Best RMSE: 0.3xxx
...
Gen 1000/1000 | Best RMSE: ~0.037x
Final RMSE: 0.0373
Validation RMSE: 0.0386
```

> Note: with default settings, training takes approximately 90 minutes depending on your machine. Reduce `n_generations` to 300 for a faster run at slightly lower accuracy.

**Hyperparameters** (top of `ga_fis.py`):
```python
n_mf          = 7      # membership functions per input
n_generations = 1000   # generations to run
pop_size      = 300    # population size
mutation_rate = 0.01   # per-gene mutation probability
crossover_rate = 0.20  # crossover probability
```

---

## Requirements

```
numpy
matplotlib
tqdm
```

No fuzzy libraries. No optimization frameworks.

---

## Future Work

- **First-order TSK** — replacing constant rule outputs with linear functions (aᵢx + bᵢy + cᵢ) would give the FIS significantly more expressive power and likely close the remaining scale compression at the surface extremes. Chromosome length would increase from 77 to 175 values.
- **Adaptive mutation rate** — decaying mutation rate over generations to allow coarse exploration early and fine-grained refinement later.
- **Parallel fitness evaluation** — the fitness function is the main computational bottleneck. Vectorizing across the population would dramatically reduce runtime.

---

## Why a Genetic Fuzzy System

Standard fuzzy systems require manual tuning of membership function centers and widths — impractical for a smooth 2D function with no obvious expert knowledge to guide placement. The search space is high-dimensional, continuous, and full of local optima that trip up gradient-based methods.

GAs are well-suited here: diverse populations and mutation maintain exploration, elitism preserves the best solutions, and the result is fully interpretable — each rule and membership function can be inspected directly. This interpretability is especially important in the context of explainable AI for mission-critical systems, which is the broader focus of the UC AI BIO Lab.

---

## Citation

```
John Cavanaugh, "GA-Optimized Fuzzy Inference System from Scratch,"
University of Cincinnati AI BIO Lab, 2026.
GitHub: https://github.com/JetHayes/ga-fuzzy-inference-system
```

---

## License

MIT License. See `LICENSE` for details.

---

## Author

**[Your Name]**
PhD Candidate, Aerospace Engineering
University of Cincinnati — AI BIO Lab
Advisor: Dr. Kelly Cohen

[LinkedIn](https://www.linkedin.com/in/privacy-evangelist/) · [Email](johnthecavanaugh@gmail.com)
