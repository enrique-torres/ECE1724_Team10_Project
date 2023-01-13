from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import csv

def load_cost_progression(string_path):
    progression_path = Path(string_path)
    costs = []
    print("Opening cost progression file...")
    with open(progression_path, mode='r', encoding='utf-8') as costreader:
        csvcostreader = csv.reader(costreader, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_number = 0
        for row in csvcostreader:
            if line_number == 0:
                print("Reading cost progression file")
                line_number += 1
            else:
                costs.append(float(row[0]))
    print("Cost progression loaded! Found " + str(len(costs)) + " cost points.")
    return costs


solution_sa = load_cost_progression("./average_10_runs_sa_progression.csv")
solution_ga = load_cost_progression("./average_10_runs_ga_progression.csv")
solution_adapt_ga = load_cost_progression("./average_10_runs_adapt_ga_progression.csv")

max_cost_sa = max(solution_sa)
max_cost_ga = max(solution_ga)
max_cost_adapt_ga = max(solution_adapt_ga)
max_cost = max(max_cost_sa, max(max_cost_ga, max_cost_adapt_ga))

normalized_sa = [x / max_cost for x in solution_sa]
normalized_ga = [x / max_cost for x in solution_ga]
normalized_adapt_ga = [x / max_cost for x in solution_adapt_ga]

large = 32; med = 28; small = 24
params = {'axes.titlesize': large,
            'legend.fontsize': large,
            'figure.figsize': (16, 10),
            'axes.labelsize': med,
            'axes.titlesize': med,
            'xtick.labelsize': med,
            'ytick.labelsize': med,
            'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
#%matplotlib inline

plt.figure(figsize=(16,8), dpi= 80)
plt.ylabel("Cost", fontsize=med)  
plt.xlabel("Iteration", fontsize=med) 
x = range(len(normalized_sa))
plt.plot(x, normalized_sa, color="black", lw=2, marker='s', markersize=12, markevery=5, label = "Simulated Annealing") 
plt.plot(x, normalized_ga, color="black", lw=2, marker='o', markersize=12, markevery=5, label = "Genetic Algorithm") 
plt.plot(x, normalized_adapt_ga, color="black", lw=2, marker='^', markersize=12, markevery=5, label = "Adaptive Genetic Algorithm") 
plt.legend(loc='center right', frameon=False)
plt.savefig("sa_vs_ga_vs_adaptga_cost_progression.svg", format="svg")
plt.show()