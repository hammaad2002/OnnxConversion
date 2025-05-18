import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper
from onnx.helper import make_node

# Model file paths
model_path = "pts_voxel_encoder_centerpoint.onnx"
model = onnx.load(model_path)

# Create inference session for the original model
session_options = ort.SessionOptions()
session = ort.InferenceSession(model_path, sess_options=session_options, providers=['CPUExecutionProvider'])

# Check if model has dynamic input shapes
def static_graph_checker(model):
    for model_input in model.graph.input:
        for dim in model_input.type.tensor_type.shape.dim:
            if dim.dim_param != "" and dim.dim_value == 0:
                print(f"This dim {dim.dim_param} is not static!")
                return False
        return True

# Model's input shape staticizer lulz
random_true = False
user_desired_input_shape = [[40000, 4, 9]]  # hardcoded input shape

# Make dynamic shapes static
if not static_graph_checker(model):
    print("Input shapes are dynamic, making them static")
    if random_true:
        for model_input in model.graph.input:
            for dim in model_input.type.tensor_type.shape.dim:
                if dim.dim_param != "" and dim.dim_value == 0:
                    dim.dim_param = ""
                    dim.dim_value = np.random.randint(1, 10)
    else:
        for i, model_input in enumerate(model.graph.input):
            user_shape = user_desired_input_shape[i]
            for j, dim in enumerate(model_input.type.tensor_type.shape.dim):
                dim.dim_param = ""
                dim.dim_value = user_shape[j]  # This order is important that we set dimparam first and then dimvalue otherwise correct shape is not shown by "model.graph.output[0].type.tensor_type.shape.dim" for onnx version 1.18.0 and onnxruntime version 1.22.0

# Find Shape and ConstantOfShape nodes to replace
shape_nodes_dict = {}
for node in model.graph.node:
    if node.name.split('_')[0] == "Shape":
        shape_nodes_dict[node.output[0]] = node.name
    elif node.name.split('_')[0] == "ConstantOfShape":
        shape_nodes_dict[node.output[0]] = node.name

# Add these nodes as temporary outputs to capture their values
for output_name in list(shape_nodes_dict.keys()):
    intermediate_layer_value_info = helper.ValueInfoProto()
    intermediate_layer_value_info.name = output_name
    model.graph.output.append(intermediate_layer_value_info)

# Run model to get values of shape nodes
temp_session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
shape_node_outputs = list(shape_nodes_dict.keys())

# Create input data filled with ones
input_data = {}
for i, input_ in enumerate(temp_session.get_inputs()):
    input_data[input_.name] = np.ones((user_desired_input_shape[i])).astype(np.float32)

# Get values of shape nodes
shape_output_values = temp_session.run(shape_node_outputs, input_data)

# Replace Shape and ConstantOfShape nodes with Constant nodes
for i, node in enumerate(model.graph.node):
    if node.name.split('_')[0] == "Shape":
        output_node_name = node.output[0]
        model.graph.node.remove(node)
        index = shape_node_outputs.index(output_node_name)
        new_node = make_node(op_type='Constant', 
                          inputs=[], 
                          outputs=[output_node_name], 
                          name="shape_node_constant_replacement",
                          value=shape_output_values[index])
        model.graph.node.insert(i, new_node)
        print(f"Found shape onnx node with complete name {node.name} and this node outputs {node.output} and takes input {node.input}")

    elif node.name.split('_')[0] == "ConstantOfShape":
        output_node_name = node.output[0]
        model.graph.node.remove(node)
        index = shape_node_outputs.index(output_node_name)
        new_node = make_node(op_type='Constant', 
                          inputs=[], 
                          outputs=[output_node_name], 
                          name="shape_node_constantofshape_replacement",
                          value=shape_output_values[index])
        model.graph.node.insert(i, new_node)
        print(f"Found constantofshape onnx node with complete name {node.name} and this node outputs {node.output} and takes input {node.input}")

# Remove temporary outputs
for output_name in list(shape_nodes_dict.keys()):
    intermediate_layer_value_info = helper.ValueInfoProto()
    intermediate_layer_value_info.name = output_name
    model.graph.output.remove(intermediate_layer_value_info)

# Removing nodes that do not have any other node taking its output as input
last_state = [node for node in model.graph.node]
keep_removing = True
while keep_removing:
    extract_outputs = [output_name.name for output_name in model.graph.output]
    for node in model.graph.node:
        output_name = node.output[0]
        if output_name not in extract_outputs:
            has_other_node_taking_output_as_input = False
            for node_ in model.graph.node:
                if node_.name != node.name:
                    if output_name in node_.input:
                        has_other_node_taking_output_as_input = True
            if not has_other_node_taking_output_as_input:
                model.graph.node.remove(node)

    if last_state == [node for node in model.graph.node]:
        keep_removing = False
    else:
        last_state = [node for node in model.graph.node] # update last state

# Remove unused initializers    
used_inputs = set() # Using set to track each input name once even if multiple nodes use it. 200 iq eh?
for node in model.graph.node:
    for input_name in node.input:
        used_inputs.add(input_name)

initializers_to_remove = []
for initializer in model.graph.initializer:
    if initializer.name not in used_inputs:
        initializers_to_remove.append(initializer)
        print(f"Found unused initializer: {initializer.name}")

# Iterating twice to prevent "collection changed during iteration" error
for initializer in initializers_to_remove:
    model.graph.initializer.remove(initializer)

# Save modified model
print("Saving model to disk")
new_model_path = model_path.replace(".onnx", "_static.onnx")
onnx.save(model, new_model_path)

# Testing if original and modified models produce the same output
print("Testing original and modified models with the same input...")

# Generate random input for testing
random_input = {}
for i, input_ in enumerate(session.get_inputs()):
    random_input[input_.name] = np.random.random(user_desired_input_shape[i]).astype(np.float32)

# Run the original model
print("Running inference on original model...")
original_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
original_output = session.run(None, random_input)

# Run the modified model
print("Running inference on modified model...")
modified_session = ort.InferenceSession(new_model_path, providers=['CPUExecutionProvider'])
modified_output = modified_session.run(None, random_input)

# Check if outputs match
print("Comparing outputs:")
if np.all(np.equal(original_output, modified_output)):
    print("All outputs match")
else:
    print("Outputs differ")


# NOTES FOR MYSELF
'''
--------------------------------
Understanding ONNX Dimension Types
--------------------------------
ONNX has two ways to represent dimensions in tensor shapes:
1) dim_value (integer) - Used for fixed, known dimensions at model definition time
2) dim_param (string) - Used for symbolic/dynamic dimensions (like "batch_size", "sequence_length")

So when you see dim_value = 0 in ONNX's API, it typically means that dimension is dynamic (unknown at definition time) 
and there's likely a corresponding dim_param with a string value.
'''