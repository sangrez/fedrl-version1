import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Step 2: Open the pickle file and load the list of DAGs
with open('data_list/task_list_linear.pickle', 'rb') as f:
    dag_list = pickle.load(f)

# Step 3: Extract the first DAG from the list
first_dag = dag_list[3]

# Step 4: Create a networkx graph from the first DAG
G = nx.DiGraph(first_dag)

# Step 5: Use matplotlib to visualize the graph
nx.draw(G, with_labels=True)
plt.show()