import os
from prettytable import PrettyTable
from settings import DATA_PATH, EVAL_PATH

total_nodes = { '027': 97, 
                '035': 120,
                '036': 168,
                '097': 72,
                '098': 172}

def get_metrics(gt_file, estimation_file):
    gt_nodes = {}
    positive_labels = {}
    est_nodes = {}
    labelled_nodes = []
    starting_nodes_gt = []
    starting_nodes_est = []
    closing_nodes_gt = []
    closing_nodes_est = []
    # Read the ground truth file
    with open(gt_file, 'r') as file:
        # Each line is a ground truth node
        # Format: n0 - n1 - n2 - n3 - n4 - n5 - n6 - n7 - n8 - n9
        gt_lines = file.readlines()
        for n, line in enumerate(gt_lines):
            # Split the line by guion
            line = line.split('-')
            # Get the node id
            node_id = n
            # Get the node connections
            gt_nodes[node_id] = [int(c) for c in line]

            for c in line:
                excluded = line.copy()
                excluded.remove(c)
                positive_labels[int(c)] = [int(c) for c in excluded]

            # Append all labelled nodes
            labelled_nodes += gt_nodes[node_id]

            if len(line) > 1:
                starting_nodes_gt.append(int(line[0]))
                # Append the rest of the nodes
                for l in line[1:]:
                    closing_nodes_gt.append(int(l))
            else:
                starting_nodes_gt.append(int(line[0]))

    # Check how many nodes are not labelled
    unlabelled_nodes = [n for n in range(total_nodes[sequence]) if n not in labelled_nodes]
    # Append all unlabelled to starting nodes
    starting_nodes_gt += unlabelled_nodes
    # print(f'Unlabelled nodes: {unlabelled_nodes}')
    final_gt_size = len(gt_nodes) + len(unlabelled_nodes)

    # Read the estimation file
    with open(estimation_file, 'r') as file:
        # Each line is an estimation node
        # Format: id: [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]
        estimation_lines = file.readlines()
        for n, line in enumerate(estimation_lines):
            # Split the line by guion
            line = line.split(':')
            # Get the node id
            node_id = int(line[0])
            # Get the node connections
            connections = line[1].replace('[', '').replace(']', '').replace('\n', '').split(',')
            # Convert the connections to integers
            connections = [int(c) for c in connections]
            est_nodes[node_id] = connections

            if len(connections) > 1:
                starting_nodes_est.append(connections[0])
                # Append the rest of the nodes
                for c in connections[1:]:
                    closing_nodes_est.append(int(l))
            else:
                starting_nodes_est.append(connections[0])

    # We are counting edges twice
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for id, nodes in est_nodes.items():
        current_size = len(nodes)
        for idx_local, n in enumerate(nodes):
            if idx_local == 0:
                # First node
                if n in starting_nodes_gt:
                    # Correctly started a node
                    tn += 1
                else:
                    # Started a node but it shouldn't
                    fn += 1
            else:
                # This node answered a localization
                # Check if it has positives
                positives = positive_labels[n]
                previous_nodes = nodes[:idx_local]

                found_positives = [p for p in positives if p in previous_nodes]
                false_positives = [p for p in previous_nodes if p not in positives]

                # Check if found positives are mayority
                if len(found_positives) > len(false_positives):
                    tp += 1
                else:
                    fp += 1
                
    P = tp / (tp + fp)
    R = tp / (tp + fn)

    # print(f'Average Precision for sequence {sequence}: {P:.2f}')
    # print(f'Average Recall for sequence {sequence}: {R:.2f}')

    return P, R
    
if __name__ == '__main__':
    folder = EVAL_PATH

    methods = sorted([name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])
    print(methods)
    if 'withdrawal' in methods:
        methods.remove('withdrawal')

    sequences = ['027', '035', '036', '097', '098']

    # Get results and plot in a pretty table
    table = PrettyTable()
    table.field_names = ['Method', 'P_027', 'R_027', 'P_035', 'R_035', 'P_036', 'R_036', 'P_097', 'R_097', 'P_098', 'R_098','P_avg', 'R_avg', 'F_beta']

    for i, method in enumerate(methods):
        PR = []
        for sequence in sequences:
            gt_file = DATA_PATH + f'{sequence}.txt'

            estimation_file = EVAL_PATH + f'{method}/{sequence}_results.txt'

            P, R = get_metrics(gt_file, estimation_file)
            PR.append((P, R))

            # print(f'Sequence {sequence}: P:{P}, R:{R}')
        
        # Add the results to the table
        row = [method]
        for pr in PR:
            # Use only 2 decimals
            row.append(round(pr[0], 2))
            row.append(round(pr[1], 2))
        # Average
        p_avg = sum([pr[0] for pr in PR]) / len(PR)
        r_avg = sum([pr[1] for pr in PR]) / len(PR)
        f_beta = (1 + 0.5**2) * p_avg * r_avg / (0.5**2 * p_avg + r_avg)
        row.append(round(p_avg, 2))
        row.append(round(r_avg, 2))
        row.append(round(f_beta, 2))
        table.add_row(row)

    print(table)
