# Get metrics for map reuse

import os
from prettytable import PrettyTable
from settings import DATA_PATH, EVAL_PATH

patient_A = {'map': 'labels_027.txt',
             'loc': 'labels_035.txt',
             'results': '035_reuse_results.txt'}

patients = {'A': patient_A}

def get_metrics(map, loc, results):
    map_labels = {}
    loc_labels = {}
    results_labels = {}
    non_localized = 0
    # Read the map
    with open(map, 'r') as f:
        map_data = f.readlines()
        for n, line in enumerate(map_data):
            # Split the line by guion
            line = line.split(':')
            # Get the area name
            area_name = line[0]
            # Get the node connections
            connections = line[1].split('-')
            # Convert the connections to integers
            connections = [int(c) for c in connections]
            
            # Add to the dictionary with connection as key
            for c in connections:
                map_labels[c] = area_name
        # Order the dictionary by key
        map_labels = dict(sorted(map_labels.items()))

    # Read the loc
    with open(loc, 'r') as f:
        loc_data = f.readlines()
        loc_set = set()
        for n, line in enumerate(loc_data):
            # print(line)
            # Split the line by guion
            line = line.split(':')
            # Get the area name
            area_name = line[0]
            # Get the node connections
            connections = line[1].split('-')
            # Convert the connections to integers
            connections = [int(c) for c in connections]

            # Add to the dictionary with connection as key
            for c in connections:
                loc_labels[c] = area_name
                loc_set.add(c)
    
    # Find max and min values in set, add the missing values as none
    max_val = max(loc_set)
    min_val = min(loc_set)
    for i in range(min_val, max_val):
        if i not in loc_set:
            loc_labels[i] = 'none'
    
    # Order the dictionary by key
    loc_labels = dict(sorted(loc_labels.items()))

    # Read the results and translate to the map
    with open(results, 'r') as f:
        results_data = f.readlines()
        for n, line in enumerate(results_data):
            # Split the line by guion
            line = line.split(':')
            # Get the area name
            node_id = int(line[0])
            # Get the node connections
            loc_id = int(line[1])

            if loc_id == -1:
                results_labels[node_id] = 'none' 
            else:
                results_labels[node_id] = map_labels[loc_id]  

    # Compare results with loc gt and get metrics
    P, R = calculate(loc_labels, results_labels)

    return P, R

def calculate(loc_labels, results_labels):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    non_relevant = 0
    for node_id, loc in results_labels.items():
        loc = loc.split(',')
        map = loc_labels[node_id].split(',')
        if loc_labels[node_id] == 'none':
            non_relevant += 1

        if loc[0] == 'none':
            # Non-localized
            # FN += 1
            if loc_labels[node_id] == 'none':
                TN += 1
                result = 'NL'
            else:
                FN += 1
                result = 'NL'
        else:
            # Localized somewhere
            if set(loc) & set(map):
                TP += 1
                result = 'CORRECT'
            else:
                FP += 1
                result = 'WRONG'

        # print(f'Node {node_id} - Loc: {loc_labels[node_id]} - Res: {results_labels[node_id]} - Result: {result} ')
        # print(f'{node_id}: {result} ')

    relevant = len(results_labels) - non_relevant
    P = TP / (TP + FP)
    R = TP / relevant

    return P, R



if __name__ == '__main__':            
    # Get results and plot in a pretty table
    folder_labels = DATA_PATH + 'reuse'
    folder_results = EVAL_PATH

    methods = sorted([name for name in os.listdir(folder_results) if os.path.isdir(os.path.join(folder_results, name))])
    methods = [method for method in methods if 'withdrawal' not in method]
    methods = ['debug']

    table = PrettyTable()
    # table.field_names = ['Method', 'P_A', 'R_A', 'P_B', 'R_B','P_avg', 'R_avg', 'F_beta'] 
    table.field_names = ['Method', 'P_avg', 'R_avg', 'F_beta'] 

    for method in methods:
        PR = []
        for patient in patients:
            map = os.path.join(folder_labels, patients[patient]['map'])
            loc = os.path.join(folder_labels, patients[patient]['loc'])
            results = os.path.join(folder_results, method, patients[patient]['results'])
            P, R = get_metrics(map, loc, results)
            PR.append([P, R])
    
        # Add the results to the table
        row = [method]

        # Average
        p_avg = sum([pr[0] for pr in PR]) / len(PR)
        r_avg = sum([pr[1] for pr in PR]) / len(PR)
        f_beta = (1 + 0.5**2) * p_avg * r_avg / (0.5**2 * p_avg + r_avg)
        row.append(round(p_avg, 2))
        row.append(round(r_avg, 2))
        row.append(round(f_beta, 2))
        table.add_row(row)
    
    print(table)




