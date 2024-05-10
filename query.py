import chromadb
import numpy as np
import time

#constant variables
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(name="my_collection")


def get_vectors(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Create a 2D array
    vectors = []
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        line = line[1:-1]  # Remove square brackets
        values = line.split(',')  # Split values by comma
        values = [float(value) for value in values]  # Convert values to float
        vectors.append(values)

    return (vectors)

def get_gnd_truth_ids(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Create a 2D array
    vectors = []
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        line = line[1:-1]  # Remove square brackets
        values = line.split(',')  # Split values by comma
        values = [int(value) for value in values]  # Convert values to int
        vectors.append(values)
    return (vectors) 


def query_db(embedding, n_results=10):
    return collection.query(query_embeddings=embedding, n_results=n_results)



#Get all the ground thruth vectors from the file
gnd_truth_ids = get_gnd_truth_ids("./gnd_truth.txt")
query_vectors = get_vectors("./query.txt")
#Define number of results per query
n_results = 4
num_executions = 2
query_hits = 0
query_count = 0
query_times = []

for i in range(len(query_vectors)):
    print(f'\nQuery #{query_count + 1}')
    query_start_time = time.time()
    query_result = query_db(query_vectors[i], n_results)
    query_end_time = time.time()
    query_times.append(query_end_time - query_start_time)

    query_execution_duration = query_end_time - query_start_time

    result_ids = query_result['ids'][0]
    result_ids = [int(x) for x in result_ids]
    query_hits += len(set(result_ids).intersection(set(gnd_truth_ids[i])))

    print(f'Number of hits : {query_hits}')
    print("Predicted results :", result_ids)
    print("Ground truth results :", gnd_truth_ids[i])
    query_count += 1
    if query_count >= num_executions:
        break

recall_ratio = query_hits / (num_executions * n_results)
print(f"\nCompleted {query_count} queries in {sum(query_times):.2f} seconds with recall ratio {recall_ratio}")