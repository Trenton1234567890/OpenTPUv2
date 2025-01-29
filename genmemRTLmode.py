import pyrtl
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', action='store', type=int, nargs='+', help='Shape of the matrix to generate.')
    return parser.parse_args()

def generate_memory(shape):
    # Create a random memory array
    mem = np.random.rand(*shape)

    # Calculate the number of memory addresses
    mem_size = np.prod(shape)

    # Calculate address width based on memory size
    addrwidth = int(np.ceil(np.log2(mem_size)))

    # Create a Pyrtl memory block with more write ports
    memory = pyrtl.MemBlock(bitwidth=32, addrwidth=addrwidth, name='my_memory', max_write_ports=mem_size)

    # Fill the memory with random values scaled to fit 32 bits
    for i in range(mem_size):
        memory[i] <<= int(mem.flatten()[i] * (2**32 - 1))  # Scale to integer

    return memory

def top_module(shape):
    # Calculate the number of memory addresses
    mem_size = np.prod(shape)

    # Calculate address width based on memory size
    addrwidth = int(np.ceil(np.log2(mem_size)))

    # Create an input for address
    address = pyrtl.Input(addrwidth, 'address')  
    read_data = pyrtl.WireVector(32, 'read_data')

    # Generate memory and connect to read data
    memory = generate_memory(shape)
    read_data <<= memory[address]

    # Instead of using pyrtl.Output, just make it a wire
    output_wire = pyrtl.WireVector(32, 'output_data')
    output_wire <<= read_data

if __name__ == '__main__':
    args = parse_args()
    top_module(args.shape)

    # Export to Verilog
    #with open('gen_mem.v', 'w') as f:
        #pyrtl.output_to_verilog(f)

    print("Memory generation export completed.")
