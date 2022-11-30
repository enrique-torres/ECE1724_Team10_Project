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
from matplotlib.animation import FuncAnimation, PillowWriter

class SASolveFacilityProblem:
    def __init__(
            self, 
            nodes_info,
            total_deliveries, 
            facility_capacity, 
            facility_cost, 
            facility_increase_prob, 
            facility_decrease_prob, 
            num_neighbours,
            num_iterations,
            k,
            lam
    ):

        self.nodes_info = nodes_info
        self.total_deliveries = total_deliveries
        self.facility_capacity = facility_capacity
        self.facility_cost = facility_cost
        self.facility_increase_prob = facility_increase_prob
        self.facility_decrease_prob = facility_decrease_prob
        self.num_neighbours = num_neighbours
        self.min_facilities = math.ceil(self.total_deliveries / float(facility_capacity)) # there have to be at least CAP / NUM_DELIVERIES facilities to serve demand
        self.cost_matrix = None
        self.ordered_matrix = None

        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

        self.num_iterations = num_iterations
        self.k = k
        self.lam = lam
        self.limit = num_iterations

        self.states = []
        self.states_full = []
        self.solution = None
        self.solution_cost = None

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
            total_cost += node[1] * self.nodes_info[node[0]][2]  # add the distance from node to facility and back, which is already squared, times the demand for that node
        return total_cost + self.facility_cost
        #return total_cost / len(facility_closest_nodes) + self.facility_cost

    def cost_function(self, solution):
        total_cost = 0
        facilities_closest_nodes = self.calculate_closest_nodes(solution)
        for closest_nodes in facilities_closest_nodes:
            total_cost += self.calculate_facility_cost(closest_nodes)
        total_cost = total_cost * (len(solution) ** 2) # heavily penalize the number of facilities to minimize them
        return total_cost

    def gen_random_solution(self):
        gen_facilities = []
        copy_nodes_info = deepcopy(self.nodes_info)
        for i in range(0, self.min_facilities):
            random_node_id = random.randint(0, len(copy_nodes_info) - 1)
            selected_node = copy_nodes_info.pop(random_node_id)
            selected_node.append(random_node_id)
            gen_facilities.append(selected_node)
        return gen_facilities

    def exponential_scheduling(k=20, lam=0.005, limit=100):
        function = lambda t: (k * np.exp(-lam*t) if t < limit else 0)
        return function

    def neighbour_function(self, solution):
        mutated_population_element = []
        for facility in solution:
            sorted_closest_nodes = self.ordered_matrix[facility[2]]
            random_index = random.randint(0, self.num_neighbours)
            new_node_index = sorted_closest_nodes[random_index]
            new_x = self.nodes_info[new_node_index][0]
            new_y = self.nodes_info[new_node_index][1]

            mutated_population_element.append([new_x, new_y, new_node_index])
        
        # if we randomly decide to eliminate a facility from the population element
        if len(solution) > self.min_facilities and random.random() > self.facility_decrease_prob:
            # remove the last element from the population element (last facility)
            mutated_population_element.pop(len(mutated_population_element) - 1)
        
        # if we randomly decide to add a facility to the population element
        if random.random() > self.facility_increase_prob:
            random_new_index_node = random.randint(0, len(self.nodes_info) - 1)
            new_facility_node = self.nodes_info[random_new_index_node]
            mutated_population_element.append([new_facility_node[0], new_facility_node[1], random_new_index_node])

        return mutated_population_element

    def initialize(self):
        first_node = True
        for node in self.nodes_info:
            if first_node:
                self.max_x = node[0]
                self.min_x = node[0]
                self.max_y = node[1]
                self.min_y = node[1]
                first_node = False
            else:
                if node[0] > self.max_x:
                    self.max_x = node[0]
                if node[0] < self.min_x:
                    self.min_x = node[0]
                if node[1] > self.max_y:
                    self.max_y = node[1]
                if node[1] < self.min_y:
                    self.min_y = node[1]
        
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
        print("Generating initial solution...")
        self.solution = self.gen_random_solution()
        self.solution_cost = self.cost_function(self.solution)
        print("Initial solution generated! Solution cost is " + str(self.solution_cost))

    def solve(self):
        self.states.append(self.solution_cost)
        self.states_full.append(self.solution)
        current = self.solution
        for iter in range(self.num_iterations):
            print("Iteration ", str(iter))
            T = SASolveFacilityProblem.exponential_scheduling(self.k, self.lam, self.limit)(iter)
            next_choice = self.neighbour_function(current)
            current_cost = self.cost_function(current)
            next_cost = self.cost_function(next_choice)
            delta_e = next_cost - current_cost
            if delta_e < 0 or np.exp(-1 * delta_e / T) > random.uniform(0.0, 1.0):
                current = next_choice
            self.states.append(self.cost_function(current))
            self.states_full.append(current)
        self.solution = current
        self.solution_cost = self.cost_function(current)
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
        plt.xlim((self.min_x, self.max_x))
        plt.ylim((self.min_y, self.max_y))

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
        plt.savefig("sa_solution.svg", format="svg")
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
        plt.savefig("sa_solution_progression.svg", format="svg")
        plt.show()

        plt.cla()

        writer = PillowWriter(fps=5)
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

        fig2 = plt.figure(figsize=(16,8), dpi= 80)
        plt.ylabel("Y Coordinate of Node", fontsize=med)  
        plt.xlabel("X Coordinate of Node", fontsize=med) 
        plt.title("Progression of Facility Locations Over Time", fontsize=large)
        plt.xlim((self.min_x, self.max_x))
        plt.ylim((self.min_y, self.max_y))
        animation_points = []
        with writer.saving(fig2, "sa_facility_locations_progression.gif", 100):
            for i in range(0, len(self.states_full)):
                x_coordinates = []
                y_coordinates = []
                for facility in self.states_full[i]:
                    x_coordinates.append(facility[0])
                    y_coordinates.append(facility[1])
                writer.grab_frame()
                animation_points.append(plt.scatter(x_coordinates, y_coordinates, color="#FF0000"))
                if len(animation_points) == 2:
                    animation_points[0].remove()
                    animation_points.pop(0)
                plt.show(block=False)

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
num_iterations = 100
facility_increase_prob = 0.2
facility_decrease_prob = 0.4
num_neighbours_nodes_div = 10
exp_schedule_k = 100
exp_schedule_lam = 0.005

# main function of the program
def main():
    dataset = load_dataset("../../DatasetGen/synthetic_dataset.csv")
    total_deliveries, prepared_dataset = prepare_dataset(dataset)
    solver = SASolveFacilityProblem(
        prepared_dataset,
        total_deliveries, 
        facility_capacity, 
        facility_cost, 
        facility_increase_prob, 
        facility_decrease_prob, 
        len(prepared_dataset) // num_neighbours_nodes_div,
        num_iterations,
        exp_schedule_k,
        exp_schedule_lam
    )
    solver.initialize()
    solver.solve()
    solver.visualize_solution()
    return 0

if __name__ == '__main__':
    main()