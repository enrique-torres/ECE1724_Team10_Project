from pathlib import Path
import csv
import math
from copy import deepcopy

class CostCalculatorKMeans:
    def __init__(
            self, 
            nodes_info,
            total_deliveries, 
            facility_capacity, 
            solution
    ):

        self.nodes_info = nodes_info
        self.total_deliveries = total_deliveries
        self.facility_capacity = facility_capacity
        self.solution = solution

        self.estimated_dtwn_x = -3.7178296565262885e-14
        self.estimated_dtwn_y = -2.6310964646970894e-13
        self.max_distance_to_dtwn = 0
        self.min_distance_to_dtwn = 0
        self.min_bid_rent_multiplier = 0.3
        self.distance_to_downtown_list = []

        self.overdemand_penalty = 100

    def get_order_nodes(self, point):
        # create a list of tuples containing each coordinate and its index in the original list
        indexed_coordinates = [(coord, i) for i, coord in enumerate(self.nodes_info)]

        # sort the indexed coordinates by their distance to the given point
        sorted_indexed_coordinates = sorted(indexed_coordinates, key=lambda coord_index: (coord_index[0][0] - point[0])**2 + (coord_index[0][1] - point[1])**2)

        # extract the indexes from the sorted indexed coordinates and return them in a list
        return [coord_index[1] for coord_index in sorted_indexed_coordinates]

    def calculate_closest_nodes(self, solution):
        facilities_closest_nodes = [[] for i in range(len(solution))]
        facilities_available_space = [self.facility_capacity for i in range(len(solution))]
        visited_nodes_set = set()
        for i, facility in enumerate(solution):
            distance_ordered_nodes = self.get_order_nodes(facility)
            for node_id in distance_ordered_nodes:
                node_cost = self.nodes_info[node_id][2]
                if node_id not in visited_nodes_set and facilities_available_space[i] - node_cost >= 0:
                    dist_x = self.nodes_info[node_id][0] - facility[0]
                    dist_y =self.nodes_info[node_id][1] - facility[1]
                    facilities_closest_nodes[i].append([node_id, dist_x * dist_x + dist_y * dist_y])
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
            distance_available_facility = 0
            for i, facility in enumerate(solution):
                if available_space_at_facility < facilities_available_space[i]:
                    dist_x = self.nodes_info[node_id][0] - facility[0]
                    dist_y = self.nodes_info[node_id][1] - facility[1]
                    distance_available_facility = dist_x * dist_x + dist_y * dist_y
                    most_available_facility = i
                    available_space_at_facility = facilities_available_space[i]
            facilities_closest_nodes[most_available_facility].append([node_id, distance_available_facility])
            facilities_available_space[most_available_facility] -= node_cost

        return facilities_closest_nodes

    def calculate_facility_cost(self, facility_closest_nodes):
        total_cost = 0
        total_demand = 0
        for node in facility_closest_nodes:
            total_cost += node[1] * self.nodes_info[node[0]][2]  # add the distance from node to facility and back, which is already squared, times the demand for that node
            total_demand += self.nodes_info[node[0]][2]
        return total_cost if total_demand <= self.facility_capacity else total_cost * self.overdemand_penalty
        #return total_cost / len(facility_closest_nodes) + self.facility_cost

    def facility_placement_cost(self, facility_id):
        distance_to_downtown = self.distance_to_downtown_list[facility_id]
        bid_rent_multiplier = 1 - self.min_bid_rent_multiplier * (distance_to_downtown - self.min_distance_to_dtwn) / (self.max_distance_to_dtwn - self.min_distance_to_dtwn)
        return bid_rent_multiplier

    def cost_function(self):
        for facility in self.solution:
            dist_x = self.estimated_dtwn_x - facility[0]
            dist_y = self.estimated_dtwn_y - facility[1]
            distance = dist_x * dist_x + dist_y * dist_y
            self.distance_to_downtown_list.append(distance)
        self.min_distance_to_dtwn = min(self.distance_to_downtown_list)
        self.max_distance_to_dtwn = max(self.distance_to_downtown_list)
        total_cost = 0
        facilities_closest_nodes = self.calculate_closest_nodes(self.solution)
        for i, closest_nodes in enumerate(facilities_closest_nodes):
            cost = self.calculate_facility_cost(closest_nodes)
            bid_rent_multiplier = self.facility_placement_cost(i)
            total_cost += cost * bid_rent_multiplier
        total_cost = total_cost * (len(self.solution) ** 2) # heavily penalize the number of facilities to minimize them
        return total_cost
                

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

def load_solution(string_path):
    solution_path = Path(string_path)
    facilities = []
    print("Opening dataset file and loading solution...")
    with open(solution_path, mode='r', encoding='utf-8') as solutionreader:
        csvsolutionreader = csv.reader(solutionreader, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_number = 0
        for row in csvsolutionreader:
            if line_number == 0:
                print("Reading K-means solution file")
                line_number += 1
            else:
                fac_x = float(row[0])
                fac_y = float(row[1])
                facilities.append([fac_x, fac_y])
    print("Solution loaded! Found " + str(len(facilities)) + " facilities.")
    return facilities

# main function of the program
def main():
    facility_capacity = 1200
    dataset = load_dataset("../../DatasetGen/synthetic_dataset.csv")
    total_deliveries, prepared_dataset = prepare_dataset(dataset)
    solution = load_solution("./Kmeans_cluster.csv")
    calculator = CostCalculatorKMeans(
        nodes_info=prepared_dataset,
        total_deliveries=total_deliveries,
        facility_capacity=facility_capacity,
        solution=solution
    )
    cost = calculator.cost_function()
    print("K-Means cost: " + str(cost))
    return 0

if __name__ == '__main__':
    main()
