#0bdcd9d680f9009e85d94da9d68ca8e3c59a3443e78f99f5a3cca86331dd22045aec92623d0e255a501a834e23b8798fc4b96c7c686714c6d991bfbf496ec18d
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
import numpy as np
import math
import json

# Initialize Qiskit Runtime Service
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='0bdcd9d680f9009e85d94da9d68ca8e3c59a3443e78f99f5a3cca86331dd22045aec92623d0e255a501a834e23b8798fc4b96c7c686714c6d991bfbf496ec18d'  # Insert your real token here
)
class Result:
    def __init__(self, backend_name, backend_version, qobj_id, job_id, success, results):
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.qobj_id = qobj_id
        self.job_id = job_id
        self.success = success
        self.results = results

    def to_dict(self):
        return {
            'backend_name': self.backend_name,
            'backend_version': self.backend_version,
            'qobj_id': self.qobj_id,
            'job_id': self.job_id,
            'success': self.success,
            'results': [result.to_dict() for result in self.results]
        }

class ExperimentResult:
    def __init__(self, shots, success, data):
        self.shots = shots
        self.success = success
        self.data = data

    def to_dict(self):
        return {
            'shots': self.shots,
            'success': self.success,
            'data': self.data
        }

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)
def save_output_to_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, cls=CustomEncoder, indent=4)
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def apply_modulo(node, mod):
    if node is None:
        return
    node.value = node.value % mod
    apply_modulo(node.left, mod)
    apply_modulo(node.right, mod)

def build_tree_from_text(values):
   if not values:
        return None
    
    # Create the root node with the first value
   root = TreeNode(int(values[0]))
    
    # Queue to keep track of nodes for level order insertion
   queue = [root]
   index = 0
    
   while index < len(values):
       current = queue.pop(0)
        # Assign the left child
       if index < len(values):
            current.left = TreeNode(int(values[index]))
            queue.append(current.left)
            index += 1
        
        # Assign the right child
       if index < len(values):
            current.right = TreeNode(int(values[index]))
            queue.append(current.right)
            index += 1
   return root
def right_rotate(root):
    if root is None or root.left is None:
        return root
    new_root = root.left
    root.left = new_root.right
    new_root.right = root
    return new_root

def left_rotate(root):
    if root is None or root.right is None:
        return root
    new_root = root.right
    root.right = new_root.left
    new_root.left = root
    return new_root

def get_nodes(root):
    if root is None:
        return []
    return [root.value] + get_nodes(root.left) + get_nodes(root.right)
# Apply mod 8
def transform_tree(root):
    apply_modulo(root, 8)
    apply_modulo(root, math.pi * 8)
    
def tree_to_list(node):
    if node is None:
        return []
    return [node.value] + tree_to_list(node.left) + tree_to_list(node.right)

def list_to_tree(values, index=0):
    if index >= len(values):
        return None, index
    node = TreeNode(values[index])
    node.left, index = list_to_tree(values, index + 1)
    node.right, index = list_to_tree(values, index + 1)
    return node, index
def print_tree(node, level=0):
    if node is not None:
        print_tree(node.right, level + 1)
        print(' ' * 4 * level + '->', node.value)
        print_tree(node.left, level + 1)
def tree_to_text(node):
    if node is None:
        return ""
    left_text = tree_to_text(node.left)
    right_text = tree_to_text(node.right)
    node_text = f"{node.value}"
    return f"{node_text}, {left_text}, {right_text}"
def update_tree_with_quantum_output(root):
    values = tree_to_list(root)
    qc = create_quantum_circuit(values)
    print(qc)
    quantum_output = execute_quantum_circuit(qc)
    print(quantum_output)
    # Convert quantum output to new tree values
    new_values = [int(key, 2) for key in quantum_output.get_counts().keys()]
    
    # Rebuild the tree with new values
    new_tree, _ = list_to_tree(new_values)
    return new_tree, _, quantum_output
def create_quantum_circuit(values):
    num_qubits = len(values)
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Apply Hadamard gates to all qubits to create superposition
    qc.h(range(num_qubits))

    # Apply a simple quantum operation (for example, X gate if value is 1)
    for i, value in enumerate(values):
        if value % 2 == 1:
            qc.x(i)

    # Measure the qubits
    qc.measure(range(num_qubits), range(num_qubits))

    return qc

def execute_quantum_circuit(qc):
    # Use Aer's qasm_simulator
    backend = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(qc, backend)
    qobj = assemble(transpiled_qc)
    result = backend.run(qobj).result()
    return result

def analyze_with_watson(text):
    authenticator = IAMAuthenticator('9uXbqcuyAKvQ0PejO078z-lNPCbmRAWY1LnPKa2tHXH_')
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2021-08-01',
        authenticator=authenticator
    )
    natural_language_understanding.set_service_url('https://api.us-east.natural-language-understanding.watson.cloud.ibm.com/instances/b233e171-152d-4002-bc65-f9d5b7859451')

    try:
        response = natural_language_understanding.analyze(
            text=text,
            features=Features(
                entities=EntitiesOptions(),
                keywords=KeywordsOptions(),
                sentiment=SentimentOptions(),
                emotion=EmotionOptions()
            )
        ).get_result()
        print(json.dumps(response, indent=2))
        return response
    except Exception as e:
        print(f"Error analyzing text with Watson: {e}")
        return None

def combine_keywords_and_text(response):
    if response and 'keywords' in response:
        keywords = [keyword['text'] for keyword in response['keywords']]
        combined_text = " ".join(keywords)
        return combined_text
    return "No keywords found."

def main():
   # User input for building the tree
    input_text = input("Enter the tree values separated by spaces: ")
    values = input_text.split()
    # Build the tree from user input
    root = build_tree_from_text(values)
    print_tree(root)
    # Apply initial transformations
    print("Tree with mod 8 then  mod 8π,")
    transformed_output = transform_tree(root)
    print_tree(transformed_output)
    # Update the tree using quantum output
    new_root, index, quantum_output = update_tree_with_quantum_output(root)
    
    # Print the updated tree
    print("Updated tree with quantum output:")
    print_tree(new_root)
    quantum_tree_output = tree_to_text(new_root)
    
    # Convert the tree structure to text
    right_tree_text = tree_to_text(new_root.right)
    left_tree_text = tree_to_text(new_root.left)
    
    # Sample text for Watson analysis
    sample_text = f"form communication (  {right_tree_text}/{left_tree_text} = output )"



    output_data = {
            "values": values,
            "quantum_output": quantum_output,
            "quantum_tree_output": quantum_tree_output
        }
    
    # Save the results to a file
    save_output_to_file("output.txt", output_data)
    # run/train the python file to test the out put on the ML model
    
    # Analyze the results using IBM Watson
    response = analyze_with_watson(sample_text)
    # Combine keywords and text
    combined_text = combine_keywords_and_text(response)
    print("Combined Text:", combined_text)
    

if __name__ == "__main__":
    main()