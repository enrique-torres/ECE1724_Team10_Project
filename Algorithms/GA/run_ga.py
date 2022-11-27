from pathlib import Path
import csv
import math
import numpy as np
import random

class GASolveFacilityProblem:
    def __init__(
            self, 
            nodes_info, 
            facility_capacity, 
            facility_cost, 
            population_size, 
            num_iterations, 
            num_parents, 
            mutation_prob, 
            facility_increase_prob, 
            facility_decrease_prob, 
            crossover_prob
    ):

        self.nodes_info = nodes_info
        self.min_facilities = math.round(float(facility_capacity) / len(nodes_info)) # there have to be at least CAP / NUM_DELIVERIES facilities to serve demand
        self.facility_cost = facility_cost

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
        self.cost = []
        self.state = []
        self.solution = None

        random.seed(100) # to keep consistency through different runs
        
    def calculate_closest_nodes(self, solution):
        cost_matrix = np.zeros((len(self.nodes_info), len(solution)))
        for i, node in enumerate(self.nodes_info):
            for j, facility in enumerate(solution):
                x_dist = facility[0] - node[0]
                y_dist = facility[1] - node[1]
                cost_matrix[i, j] = x_dist * x_dist + y_dist * y_dist # distance without sqrt to increase performance
        
        facilities_closest_nodes = [[]] * len(solution)
        for i, node in enumerate(self.nodes_info):
            min_index = np.argmin(cost_matrix[i])
            facilities_closest_nodes[min_index].append([i, cost_matrix[i][min_index]]) # save node ID and distance squared
        return facilities_closest_nodes

    def calculate_facility_cost(self, facility_closest_nodes):
        total_cost = self.facility_cost
        for node in facility_closest_nodes:
            total_cost += 2 * node[1] # add the distance from node to facility and back, which is already squared
        return total_cost

    def gen_random_solution(self):
        gen_facilities = []
        for i in range(0, self.min_facilities):
            rand_x = random.uniform(self.min_x, self.max_x)
            rand_y = random.uniform(self.min_y, self.max_y)
            gen_facilities.append([rand_x, rand_y])
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
    # first, we mutate facility locations based on a normal distribution
    # then we randomly decide if we increment the number of facilities, decrement it or keep it the same
    def mutate(self, pop_element):
        mutated_population_element = []
        for facility in pop_element:
            new_x = np.random.normal(loc=facility[0], scale=self.x_resolution, size=1)[0]
            new_y = np.random.normal(loc=facility[1], scale=self.y_resolution, size=1)[0]
            mutated_population_element.append([new_x, new_y])
        
        # if we randomly decide to eliminate a facility from the population element
        if len(pop_element) > self.min_facilities and random.random() > self.facility_decrease_prob:
            # remove the last element from the population element (last facility)
            mutated_population_element.pop(len(mutated_population_element) - 1)
        
        # if we randomly decide to add a facility to the population element
        if random.random() > self.facility_increase_prob:
            rand_x = random.uniform(self.min_x, self.max_x)
            rand_y = random.uniform(self.min_y, self.max_y)
            mutated_population_element.append([rand_x, rand_y])

        return mutated_population_element

    def fitness_one(self, solution):
        total_cost = 0
        facilities_closest_nodes = self.calculate_closest_nodes(solution)
        for closest_nodes in facilities_closest_nodes:
            total_cost += self.calculate_facility_cost(closest_nodes)
        total_cost = total_cost * (len(solution) ** 2) # heavily penalize the number of facilities to minimize them
        return total_cost

    def fitness_all(self, population):
        total_costs = []
        for solution in population:
            total_cost = 0
            facilities_closest_nodes = self.calculate_closest_nodes(solution)
            for closest_nodes in facilities_closest_nodes:
                total_cost += self.calculate_facility_cost(closest_nodes)
            total_cost = total_cost * (len(solution) ** 2) # heavily penalize the number of facilities to minimize them
            total_costs.append(total_cost)
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

        for _ in range(0, self.pop_size):
            self.population.append(self.gen_random_solution())

    def solve(self):
        # TODO: implement main solve function
        return

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
    for node in nodes_info:
        # for every delivery for that node, create a new node with same location
        for _ in range(0, node[5]):
            # only save node X,Y coordinates, as after expanding the deliveries, this is all that matters
            prepared_dataset.append([node[2], node[3]])
    print("Dataset preparation complete! Expanded " + str(len(prepared_dataset)) + " deliveries.")
    return prepared_dataset

# main function of the program
def main():
    dataset = load_dataset("../../DatasetGen/synthetic_dataset.csv")
    prepared_dataset = prepare_dataset(dataset)
    return 0