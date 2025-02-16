import json
import itertools
import collections
import heapq
from functools import reduce

class Inference:
    def __init__(self, data):
        self.variables_count = data["VariablesCount"]
        self.k_value = data["k value (in top k)"]
        self.potentials = data["Cliques and Potentials"]
        self.graph = collections.defaultdict(set)
        self.cliques = []
        self.potential_map = {}
        
        # Construct graph adjacency list
        for entry in self.potentials:
            nodes = tuple(entry["cliques"])
            self.graph[nodes[0]].add(nodes[1])
            self.graph[nodes[1]].add(nodes[0])
            self.potential_map[nodes] = entry["potentials"]
    
    def triangulate_and_get_cliques(self):
        """ Triangulate the graph and extract maximal cliques """
        # Create a copy of the adjacency list
        graph_copy = {node: set(neighbors) for node, neighbors in self.graph.items()}
        eliminated_nodes = set()
        
        while len(eliminated_nodes) < self.variables_count:
            # Choose a node with the minimum degree
            available_nodes = [node for node in graph_copy if node not in eliminated_nodes]
            if not available_nodes:
                break  # Prevent accessing empty list
            
            min_node = min(available_nodes, key=lambda x: len(graph_copy[x]))

            # Ensure the node exists before accessing its neighbors
            if min_node not in graph_copy:
                continue

            neighbors = list(graph_copy[min_node])
            for u, v in itertools.combinations(neighbors, 2):
                if u in graph_copy and v in graph_copy:  # Ensure both nodes exist
                    graph_copy[u].add(v)
                    graph_copy[v].add(u)

            # Remove the node from graph
            eliminated_nodes.add(min_node)
            graph_copy.pop(min_node, None)  # Avoid KeyError
            
        # Extract maximal cliques
        self.cliques = list(self.potential_map.keys())

    
    def get_junction_tree(self):
        """ Construct the junction tree ensuring running intersection property """
        # Sort cliques by size (largest first) and use them to form a tree
        self.junction_tree = collections.defaultdict(set)
        sorted_cliques = sorted(self.cliques, key=len, reverse=True)
        visited = set()
        
        for clique in sorted_cliques:
            for other in sorted_cliques:
                if clique != other and set(clique).intersection(set(other)) and clique not in visited:
                    self.junction_tree[clique].add(other)
                    self.junction_tree[other].add(clique)
                    visited.add(clique)
    
    def assign_potentials_to_cliques(self):
        """ Assign potential values to corresponding cliques """
        self.assigned_potentials = {}
        for clique in self.cliques:
            self.assigned_potentials[clique] = self.potential_map.get(clique, [])
    
    def get_z_value(self):
        """ Compute partition function Z using message passing """
        Z = 0
        for potentials in self.assigned_potentials.values():
            Z += sum(potentials)
        return Z
    
    def compute_marginals(self):
        """ Compute marginal probabilities """
        marginals = []
        Z = self.get_z_value()
        for var in range(self.variables_count):
            marginal = [0, 0]
            for clique, potentials in self.assigned_potentials.items():
                if var in clique:
                    marginal[0] += sum(potentials[:len(potentials)//2])
                    marginal[1] += sum(potentials[len(potentials)//2:])
            marginals.append([m/Z for m in marginal])
        return marginals
    
    def compute_top_k(self):
        """ Compute top-k most probable assignments """
        assignment_probs = []
        Z = self.get_z_value()
        
        for assignment in itertools.product([0, 1], repeat=self.variables_count):
            prob = 1
            for clique, potentials in self.assigned_potentials.items():
                index = sum((assignment[i] * (2 ** idx)) for idx, i in enumerate(clique))
                prob *= potentials[index]
            prob /= Z
            assignment_probs.append((prob, list(assignment)))
        
        top_k = heapq.nlargest(self.k_value, assignment_probs, key=lambda x: x[0])
        return [{"assignment": assign, "probability": prob} for prob, assign in top_k]

########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)

if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('Sample_Testcase.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')
