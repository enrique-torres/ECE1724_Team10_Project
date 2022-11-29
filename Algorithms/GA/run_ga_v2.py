from pathlib import Path
import csv
import math
import numpy as np
import random
from heapq import nsmallest
import matplotlib.pyplot as plt
from copy import deepcopy
from joblib import Parallel, delayed
from matplotlib.patches import Polygon

class GASolveFacilityProblem:
    def __init__(
            self, 
            nodes_info,
            total_deliveries, 
            facility_capacity, 
            facility_cost, 
            population_size, 
            num_iterations, 
            num_parents, 
            mutation_prob, 
            facility_increase_prob, 
            facility_decrease_prob, 
            len_nodes_mutation,
            crossover_prob
    ):

        self.nodes_info = nodes_info
        self.total_deliveries = total_deliveries
        self.min_facilities = math.ceil(self.total_deliveries / float(facility_capacity)) # there have to be at least CAP / NUM_DELIVERIES facilities to serve demand
        self.facility_cost = facility_cost
        self.facility_capacity = facility_capacity
        self.cost_matrix = None
        self.ordered_matrix = None
        self.nodes_len_mutation = len_nodes_mutation # the number of closest nodes to be considered for the mutation

        self.pop_size = population_size
        self.iterations = num_iterations
        self.num_parents = num_parents
        self.mutation_prob = mutation_prob
        self.facility_increase_prob = facility_increase_prob
        self.facility_decrease_prob = facility_decrease_prob
        self.crossover_prob = crossover_prob
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0
        self.x_resolution = 0 # = (self.x_max - self.x_min) / len(self.nodes_info)
        self.y_resolution = 0 # = (self.y_max - self.y_min) / len(self.nodes_info)

        self.population = []
        self.costs = []
        self.states = []
        self.solution = None
        self.solution_cost = None

        self.infinity = float("inf")

        random.seed(100) # to keep consistency through different runs

    def calculate_closest_nodes(self, solution):
        facilities_closest_nodes = [[] for i in range(len(solution))]
        facilities_available_space = [self.facility_capacity for i in range(len(solution))]
        visited_nodes_set = set()
        for i, facility in enumerate(solution):
            closest_node_indexes = self.ordered_matrix[facility[2]]
            for node_id in closest_node_indexes:
                node_cost = self.nodes_info[node_id][2]
                if node_id not in visited_nodes_set and facilities_available_space[i] - node_cost >= 0:
                    facilities_closest_nodes[i].append([node_id, self.cost_matrix[facility[2]][node_id]])
                    facilities_available_space[i] -= node_cost
                    visited_nodes_set.add(node_id)
                elif node_id not in visited_nodes_set and facilities_available_space[i] - node_cost < 0:
                    break
        all_nodes_set = set(range(0, len(self.nodes_info)))
        not_visited_nodes_set = list(all_nodes_set.difference(visited_nodes_set))

        for node_id in not_visited_nodes_set:

            node_cost = self.nodes_info[node_id][2]

            most_available_facility = 0
            available_space_at_facility = facilities_available_space[most_available_facility]
            for i, facility in enumerate(solution):
                if available_space_at_facility < facilities_available_space[i]:
                    most_available_facility = i
                    available_space_at_facility = facilities_available_space[i]
            facilities_closest_nodes[most_available_facility].append([node_id, self.cost_matrix[most_available_facility][node_id]])
            facilities_available_space[most_available_facility] -= node_cost

        return facilities_closest_nodes

    def calculate_facility_cost(self, facility_closest_nodes):
        total_cost = 0
        for node in facility_closest_nodes:
            total_cost += node[1] # add the distance from node to facility and back, which is already squared
        return total_cost + self.facility_cost

    def gen_random_solution(self):
        gen_facilities = []
        copy_nodes_info = deepcopy(self.nodes_info)
        for i in range(0, self.min_facilities):
            random_node_id = random.randint(0, len(copy_nodes_info) - 1)
            selected_node = copy_nodes_info.pop(random_node_id)
            selected_node.append(random_node_id)
            gen_facilities.append(selected_node)
        return gen_facilities

    # Order one crossover implementation, with different sized list implementation
    def crossover(self, parent1, parent2):
        p1_length = len(parent1)
        p2_length = len(parent2)
        child = list()
        if p1_length > p2_length:
            start_segment = random.randint(0, p2_length // 2)
            end_segment = random.randint(p2_length // 2 + 1, p2_length - 1) + 1
            child.extend(parent2[start_segment : end_segment])
            child.extend(parent1[:start_segment])
            child.extend(parent1[end_segment:])
        else:
            start_segment = random.randint(0, p1_length // 2)
            end_segment = random.randint(p1_length // 2 + 1, p1_length - 1) + 1
            child.extend(parent1[start_segment : end_segment])
            child.extend(parent2[:start_segment])
            child.extend(parent2[end_segment:])
        return child

    # custom mutation function for facility location problem
    # first, we mutate facility locations to a nearby node
    # then we randomly decide if we increment the number of facilities, decrement it or keep it the same
    def mutate(self, pop_element):
        mutated_population_element = []
        for facility in pop_element:
            sorted_closest_nodes = np.argsort(self.cost_matrix[facility[2]])
            random_index = random.randint(0, self.nodes_len_mutation)
            new_node_index = sorted_closest_nodes[random_index]
            new_x = self.nodes_info[new_node_index][0]
            new_y = self.nodes_info[new_node_index][1]

            mutated_population_element.append([new_x, new_y, new_node_index])
        
        # if we randomly decide to eliminate a facility from the population element
        if len(pop_element) > self.min_facilities and random.random() > self.facility_decrease_prob:
            # remove the last element from the population element (last facility)
            mutated_population_element.pop(len(mutated_population_element) - 1)
        
        # if we randomly decide to add a facility to the population element
        if random.random() > self.facility_increase_prob:
            random_new_index_node = random.randint(0, len(self.nodes_info) - 1)
            new_facility_node = self.nodes_info[random_new_index_node]
            mutated_population_element.append([new_facility_node[0], new_facility_node[1], random_new_index_node])

        return mutated_population_element

    def fitness_one(self, solution):
        total_cost = 0
        facilities_closest_nodes = self.calculate_closest_nodes(solution)
        for closest_nodes in facilities_closest_nodes:
            total_cost += self.calculate_facility_cost(closest_nodes)
        total_cost = total_cost * (len(solution) ** 2) # heavily penalize the number of facilities to minimize them
        return total_cost

    def fitness_all(self, population):
        total_costs = Parallel(n_jobs=8)(delayed(self.fitness_one)(population[i]) for i in range(len(population)))
        #total_costs = [self.fitness_one(population[i]) for i in range(len(population))]
        return total_costs

    def init_population(self):
        for node in self.nodes_info:
            if node[0] > self.max_x:
                self.max_x = node[0]
            if node[0] < self.min_x:
                self.min_x = node[0]
            if node[1] > self.max_y:
                self.max_y = node[1]
            if node[1] < self.min_y:
                self.min_y = node[1]
        
        self.x_resolution = (self.max_x - self.min_x) / len(self.nodes_info)
        self.y_resolution = (self.max_y - self.min_y) / len(self.nodes_info)

        self.cost_matrix = np.zeros((len(self.nodes_info), len(self.nodes_info)))
        print("Initializing cost matrix...")
        for i, node_i in enumerate(self.nodes_info):
            for j, node_j in enumerate(self.nodes_info):
                x_dist = node_i[0] - node_j[0]
                y_dist = node_i[1] - node_j[1]
                self.cost_matrix[i, j] = x_dist * x_dist + y_dist * y_dist # distance without sqrt to increase performance (dirty distance)
        print("Initialized cost matrix")
        print("Initializing the ordered index closest nodes matrix...")
        temp_ordered_matrix = []
        for i in range(0, len(self.nodes_info)):
            node_i_ordered_indexes = np.argsort(self.cost_matrix[i])
            temp_ordered_matrix.append(node_i_ordered_indexes.tolist())
        self.ordered_matrix = np.array(temp_ordered_matrix)
        print("Initialized the ordered index closest nodes matrix")   

        for _ in range(0, self.pop_size):
            self.population.append(self.gen_random_solution())
        self.costs = self.fitness_all(self.population)
        print("Initialized initial population.")

    def solve(self):
        # Check that initial population exists:
        if self.population:
            # Show some information
            print("Initial Population costs:")
            print(self.costs)
        else:
            raise Exception("Population not initialized.")
        
        for iter in range(self.iterations):
            print("Iteration ", str(iter))
            self.costs = self.fitness_all(self.population)
            self.states.append(min(self.costs))

            parents = nsmallest(self.num_parents,self.population, key=lambda x: self.costs[self.population.index(x)])

            offspring = []
            new_population = []
            for p1, p2 in zip(parents[:len(parents)//2],parents[len(parents)//2:]):
                # Crossover probability
                if random.random() < self.crossover_prob:
                    offspring.append(self.crossover(p1,p2))
                    offspring.append(self.crossover(p2,p1))
                else:
                    offspring.append(p1)
                    offspring.append(p2)
            for child in offspring:
                if random.random() < self.mutation_prob:
                    new_population.append(self.mutate(child))
                else:
                    new_population.append(child)
            new_population.extend(parents)
            self.population = new_population
        
        # Show best solution
        self.costs = self.fitness_all(self.population)
        self.states.append(min(self.costs))
        self.solution = min(self.population, key=lambda x: self.costs[self.population.index(x)])
        print("Minimum: ", min(self.population, key=lambda x: self.costs[self.population.index(x)]))
        self.solution_cost = self.costs[self.population.index(self.solution)]
        print("Best Solution:", self.solution)
        print("Best Solution Cost:", self.solution_cost)

    def visualize_solution(self):
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

        plt.figure(figsize=(16,8), dpi= 80)
        plt.ylabel("Y Coordinate of Node", fontsize=med)  
        plt.xlabel("X Coordinate of Node", fontsize=med) 
        plt.title("Calculated Facilities Map - w/Clustered Nodes", fontsize=large)

        nodes_per_facility = self.calculate_closest_nodes(self.solution)
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(nodes_per_facility))]
        total_deliveries_served = 0
        for i, facility_close_nodes in enumerate(nodes_per_facility):
            x_coordinates = []
            y_coordinates = []
            total_deliveries_assigned = 0
            for node in facility_close_nodes:
                x_coordinates.append(self.nodes_info[node[0]][0])
                y_coordinates.append(self.nodes_info[node[0]][1])
                total_deliveries_assigned += self.nodes_info[node[0]][2]
                total_deliveries_served += self.nodes_info[node[0]][2]
            print("Facility " + str(i) + " has " + str(total_deliveries_assigned) + " deliveries assigned to it.")
            plt.scatter(x_coordinates, y_coordinates, color=color[i], alpha=0.5, edgecolors='none', s=200)
        print("The total number of deliveries served is " + str(total_deliveries_served))
        x_coordinates = []
        y_coordinates = []
        for facility in self.solution:
            x_coordinates.append(facility[0])
            y_coordinates.append(facility[1])
        plt.scatter(x_coordinates, y_coordinates, color="#FF0000")
        plt.savefig("solution.svg", format="svg")
        plt.show()

        plt.cla()
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

        plt.figure(figsize=(16,8), dpi= 80)
        plt.ylabel("Cost of Best Solution", fontsize=med)  
        plt.xlabel("# Iteration", fontsize=med) 
        plt.title("Cost of Best Solution Through The Iterations", fontsize=large)
        plt.plot(self.states)
        plt.savefig("solution_progression.svg", format="svg")
        plt.show()



# loads the synthetic dataset from a .csv file
# the synthetic dataset must have the following columns:
# Node ID, Node OSMID, X, Y, Node Weight, Number Deliveries
def load_dataset(string_path):
    dataset_path = Path(string_path)
    nodes_info = []
    print("Opening dataset file and loading dataset...")
    with open(dataset_path, mode='r', encoding='utf-8') as datasetreader:
        csvdatasetreader = csv.reader(datasetreader, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_number = 0
        for row in csvdatasetreader:
            if line_number == 0:
                print("Reading synthetic dataset file")
                line_number += 1
            else:
                node_id = int(row[0])
                node_osmid = int(float(row[1]))
                node_x = float(row[2])
                node_y = float(row[3])
                node_weight = float(row[4])
                node_deliveries = int(row[5])
                nodes_info.append([node_id, node_osmid, node_x, node_y, node_weight, node_deliveries])
    print("Dataset loaded! Found " + str(len(nodes_info)) + " nodes.")
    return nodes_info

# unwraps or expands the deliveries in each node, converting them to nodes with the same coordinates
# for ease of use in the algorithms
def prepare_dataset(nodes_info):
    prepared_dataset = []
    print("Preparing the dataset for use in the algorithms...")
    total_deliveries = 0
    for node in nodes_info:
        # for every delivery for that node, create a new node with same location
        #for _ in range(0, node[5]):
            # only save node X,Y coordinates, as after expanding the deliveries, this is all that matters
        prepared_dataset.append([node[2], node[3], node[5]])
        total_deliveries += node[5]
    print("Dataset preparation complete! Expanded " + str(len(prepared_dataset)) + " deliveries.")
    return total_deliveries, prepared_dataset

facility_capacity = 1200
facility_cost = 5000
population_size = 8
num_iterations = 100
num_parents = 4
mutation_prob = 0.6
crossover_prob = 0.7
facility_increase_prob = 0.5
facility_decrease_prob = 0.5
len_mutation_nodes_div = 10

# main function of the program
def main():
    dataset = load_dataset("../../DatasetGen/synthetic_dataset.csv")
    total_deliveries, prepared_dataset = prepare_dataset(dataset)
    solver = GASolveFacilityProblem(
        prepared_dataset, 
        total_deliveries,
        facility_capacity, 
        facility_cost, 
        population_size, 
        num_iterations, 
        num_parents, 
        mutation_prob, 
        facility_increase_prob, 
        facility_decrease_prob, 
        len(prepared_dataset) // len_mutation_nodes_div,
        crossover_prob
    )
    solver.init_population()
    solver.solve()
    solver.visualize_solution()
    return 0

if __name__ == '__main__':
    main()