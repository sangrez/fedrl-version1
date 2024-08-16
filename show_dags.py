import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Step 2: Open the pickle file and load the list of DAGs
with open('data_list/test/task_list_combined.pickle', 'rb') as f:
    dag_list = pickle.load(f)

# Step 3: Extract the first DAG from the list
for i, dag in enumerate(dag_list):
    

# Step 4: Create a networkx graph from the first DAG
    G = nx.DiGraph(dag)

# Step 5: Use matplotlib to visualize the graph
    nx.draw(G, with_labels=True)
    plt.show()