import pyrtl
import numpy as np
import argparse

def gen_one_hot(lower=-5, upper=5, shape=(8, 8)):
    one_hot = np.random.randint(lower, upper, shape, dtype=np.int8)
    if shape[0] == shape[1]:
        for i in range(shape[0]):
            one_hot[i, i] = 64  # Set diagonal elements to 64
    else:
        for i in range(shape[0]):
            one_hot[i, 0] = np.random.randint(lower, upper, dtype=np.int8)  # Set first column
    return one_hot

def gen_nn(shape, lower=None, upper=None):
    one_hot_matrix = gen_one_hot(lower, upper, shape)
    print(one_hot_matrix)

    # Create a PyRTL memory block
    mem_size = np.prod(shape)
    addrwidth = int(np.ceil(np.log2(mem_size)))  # Calculate the required address width
    memory = pyrtl.MemBlock(bitwidth=32, addrwidth=addrwidth, name='my_memory', max_write_ports=mem_size)

    # Fill the memory with one-hot encoded values
    for i in range(mem_size):
        memory[i] <<= int(one_hot_matrix.flatten()[i])

    # Define the read address input
    address = pyrtl.Input(addrwidth, 'address')  # Use the calculated addrwidth
    read_data = pyrtl.WireVector(32, 'read_data')  # Match the bitwidth of memory

    # Connect memory read
    read_data <<= memory[address]

    output_wire = pyrtl.WireVector(32, 'output_data')
    output_wire <<= read_data

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', action='store', type=int, nargs='+', help='Shape of matrix to generate.')
    parser.add_argument('--debug', action='store_true', help='Switch debug prints.')
    parser.add_argument('--range', type=int, nargs=2, help='Generate random in [lower, upper)')
    args = parser.parse_args()

if __name__ == '__main__':
    parse_args()
    gen_nn(args.shape, *args.range)

# Export to Verilog
#with open('onehot.v', 'w') as f:
    #pyrtl.output_to_verilog(f)

print("Onehot mem generation export completed.")
