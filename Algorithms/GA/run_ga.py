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
            crossover_prob
    ):

        self.nodes_info = nodes_info
        self.total_deliveries = total_deliveries
        self.min_facilities = math.ceil(self.total_deliveries / float(facility_capacity)) # there have to be at least CAP / NUM_DELIVERIES facilities to serve demand
        self.facility_cost = facility_cost
        self.facility_capacity = facility_capacity

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
        
        '''
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
        '''

    def calculate_closest_nodes(self, solution):
        facilities_closest_nodes = [[] for i in range(len(solution))]
        facilities_available_space = [self.facility_capacity for i in range(len(solution))]
        limbo_nodes = []
        for i, node in enumerate(self.nodes_info):
            min_index = 0
            min_value = None
            for j, facility in enumerate(solution):
                x_dist = facility[0] - node[0]
                y_dist = facility[1] - node[1]
                cost = x_dist * x_dist + y_dist * y_dist
                if min_value is None:
                    min_index = j
                    min_value = cost
                else:
                    if cost < min_value:
                        min_value = cost
                        min_index = j
            
            if facilities_available_space[min_index] - node[2] > 0:
                facilities_closest_nodes[min_index].append([i, min_value]) # save node ID and distance squared
                facilities_available_space[min_index] -= node[2]
            else:
                limbo_nodes.append(i)

        for node_id in limbo_nodes:
            distance_to_facility = []
            for j, facility in enumerate(solution):
                x_dist = facility[0] - self.nodes_info[node_id][0]
                y_dist = facility[1] - self.nodes_info[node_id][1]
                cost = x_dist * x_dist + y_dist * y_dist
                distance_to_facility.append(cost)
            found_facility = False
            facilities_searched = 0
            while not found_facility and facilities_searched < len(solution):
                index_min = min(range(len(distance_to_facility)), key=distance_to_facility.__getitem__)
                if facilities_available_space[index_min] - self.nodes_info[node_id][2] > 0:
                    facilities_closest_nodes[index_min].append([i, distance_to_facility[index_min]]) # save node ID and distance squared
                    facilities_available_space[index_min] -= self.nodes_info[node_id][2]
                    found_facility = True
                else:
                    distance_to_facility[index_min] = self.infinity
                facilities_searched += 1
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
            #rand_x = random.uniform(self.min_x, self.max_x)
            #rand_y = random.uniform(self.min_y, self.max_y)
            random_node_id = random.randint(0, len(copy_nodes_info) - 1)
            selected_node = copy_nodes_info.pop(random_node_id)
            gen_facilities.append(selected_node)
        return gen_facilities

    # Order one crossover implementation, with different sized list implementation
    def crossover(self, parent1, parent2):
        try:
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
        except Exception as ex:
            print(parent1)
            print(parent2)
            print(str(ex))
            exit(1)

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
        total_costs = Parallel(n_jobs=8)(delayed(self.fitness_one)(population[i]) for i in range(len(population)))
        '''
        total_costs = []
        for solution in population:
            total_cost = 0
            facilities_closest_nodes = self.calculate_closest_nodes(solution)
            for closest_nodes in facilities_closest_nodes:
                total_cost += self.calculate_facility_cost(closest_nodes)
            total_cost = total_cost * (len(solution) ** 2) # heavily penalize the number of facilities to minimize them
            total_costs.append(total_cost)
        '''
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
        print(self.population)
        self.costs = self.fitness_all(self.population)

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
        polygons_list = []
        for i, facility_close_nodes in enumerate(nodes_per_facility):
            x_coordinates = []
            y_coordinates = []
            polygon_coordinates = []
            total_deliveries_assigned = 0
            for node in facility_close_nodes:
                x_coordinates.append(self.nodes_info[node[0]][0])
                y_coordinates.append(self.nodes_info[node[0]][1])
                total_deliveries_assigned += self.nodes_info[node[0]][2]
                polygon_coordinates.append((self.nodes_info[node[0]][0], self.nodes_info[node[0]][1]))
            print("Facility " + str(i) + " has " + str(total_deliveries_assigned) + " deliveries assigned to it.")
            plt.scatter(x_coordinates, y_coordinates, color=color[i], alpha=0.5, edgecolors='none', s=200)
            poly = Polygon(polygon_coordinates, color=color[i])
            polygons_list.append(poly)
        x_coordinates = []
        y_coordinates = []
        for facility in self.solution:
            x_coordinates.append(facility[0])
            y_coordinates.append(facility[1])
        plt.scatter(x_coordinates, y_coordinates, color="#FF0000")
        plt.savefig("solution.svg", format="svg")
        plt.show()

        plt.cla()

        fig, ax = plt.subplots(1,1)
        for poly in polygons_list:
            ax.add_patch(poly)

        x_coordinates = []
        y_coordinates = []
        for facility in self.solution:
            x_coordinates.append(facility[0])
            y_coordinates.append(facility[1])

        ax.scatter(x_coordinates, y_coordinates, color="#FF0000")
        plt.savefig("solution_polygons.svg", format="svg")
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

facility_capacity = 10000
facility_cost = 5000
population_size = 8
num_iterations = 100
num_parents = 4
mutation_prob = 0.6
crossover_prob = 0.7
facility_increase_prob = 0.5
facility_decrease_prob = 0.5

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
        crossover_prob
    )
    solver.init_population()
    solver.solve()
    solver.visualize_solution()
    return 0

if __name__ == '__main__':
    main()