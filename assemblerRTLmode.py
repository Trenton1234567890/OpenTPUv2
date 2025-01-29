import pyrtl

# Define your opcodes and other constants here
OPCODE2BIN = {
    'NOP': (0x00, 0, 0, 0),
    'HLT': (0x01, 0, 0, 0),
    'RW':  (0x02, 1, 0, 0),   # Read Weights
    'RHM': (0x03, 2, 2, 1),   # Read Host Memory
    'WHM': (0x04, 2, 2, 1),   # Write Host Memory
    'MMC': (0x05, 2, 2, 1),
    'ACT': (0x06, 1, 1, 1),
    'POOL': (0x07, 2, 2, 1),   # Pooling
    'SOFTMAX': (0x08, 2, 2, 1),  # Softmax
    'ADD': (0x09, 2, 1, 0)      # Addition
}

# Constants for the instruction encoding
OP_START, OP_END = 0, 7
FLAGS_START, FLAGS_END = 8, 15
LEN_START, LEN_END = 16, 23
ADDR_START, ADDR_END = 24, 55
UBADDR_START, UBADDR_END = 56, 87

# Create inputs for the instruction components
opcode = pyrtl.Input(8, 'opcode')  # 8 bits for opcode
src = pyrtl.Input(32, 'src')        # 32 bits for source address
tar = pyrtl.Input(32, 'tar')        # 32 bits for target address
length = pyrtl.Input(8, 'length')   # 8 bits for length
flags = pyrtl.Input(8, 'flags')     # 8 bits for flags
instr_out = pyrtl.Output(112, 'instr_out')  # 112 bits for the complete instruction

def putbytes(val, lo, hi):
    """ Pack value 'val' into a bit range defined by lo..hi inclusive. """
    if isinstance(val, int):  # Handle constant values
        return pyrtl.Const(val, hi - lo + 1)[0:hi - lo + 1]
    else:  # Handle Pyrtl wires
        return val[0:hi - lo + 1]

def format_instr(op, flags, length, addr, ubaddr, operation_type=None):
    """ Format the instruction from the components. """
    instr = pyrtl.concat(
        putbytes(op, OP_START, OP_END),
        putbytes(flags, FLAGS_START, FLAGS_END),
        putbytes(length, LEN_START, LEN_END),
        putbytes(addr, ADDR_START, ADDR_END),
        putbytes(ubaddr, UBADDR_START, UBADDR_END)
    )
    
    if operation_type is not None:
        instr = pyrtl.concat(instr, putbytes(operation_type, 0, 7))  # Add operation type field
    
    return instr

# Logic to generate the instruction encoding
with pyrtl.conditional_assignment:
    for inst_name, (op, n_src, n_tar, n_len) in OPCODE2BIN.items():
        with opcode == op:
            if inst_name in ['NOP', 'HLT']:
                temp_instr_out = format_instr(op, 0, 0, 0, 0)
            elif inst_name == 'RW':  # Read Weights
                temp_instr_out = format_instr(op, flags, 0, src, 0)
            elif inst_name == 'RHM':  # Read Host Memory
                temp_instr_out = format_instr(op, flags, length, src, tar)
            elif inst_name == 'WHM':  # Write Host Memory
                temp_instr_out = format_instr(op, flags, length, src, tar)
            elif inst_name == 'ACT':  # Activation
                temp_instr_out = format_instr(op, flags, length, src, tar)
            elif inst_name == 'POOL':  # Pooling
                operation_type = flags  # Assume flags hold pooling type (e.g., max or average pooling)
                temp_instr_out = format_instr(op, flags, length, src, tar, operation_type)
            elif inst_name == 'SOFTMAX':  # Softmax
                operation_type = flags  # Assume flags hold softmax type (e.g., 0 for standard, 1 for approx.)
                temp_instr_out = format_instr(op, flags, length, src, tar, operation_type)
            elif inst_name == 'ADD':  # Addition
                temp_instr_out = format_instr(op, flags, 0, src, tar)  # No length for addition
            else:  # MMC (assumed to be another kind of memory access or operation)
                temp_instr_out = format_instr(op, flags, length, tar, src)

    instr_out <<= temp_instr_out  # Assign the result to instr_out

def main():
    # Example simulation to test functionality
    sim = pyrtl.Simulation()

    # Test instruction: RHM (Read Host Memory)
    sim.step({
        opcode: OPCODE2BIN['RHM'][0], 
        src: 0x00000001, 
        tar: 0x00000002, 
        length: 3, 
        flags: 0
    })
    print(f"Encoded instruction for RHM: {sim.value}")

    # Test instruction: RW (Read Weights)
    sim.step({
        opcode: OPCODE2BIN['RW'][0], 
        src: 0x00000010, 
        tar: 0x00000011, 
        length: 5, 
        flags: 0
    })
    print(f"Encoded instruction for RW: {sim.value}")

    # Test instruction: WHM (Write Host Memory)
    sim.step({
        opcode: OPCODE2BIN['WHM'][0], 
        src: 0x00000020, 
        tar: 0x00000021, 
        length: 6, 
        flags: 1  # Example flag for write operation
    })
    print(f"Encoded instruction for WHM: {sim.value}")

    # Test instruction: ACT (Activation)
    sim.step({
        opcode: OPCODE2BIN['ACT'][0], 
        src: 0x0000000A, 
        tar: 0x0000000B, 
        length: 1, 
        flags: 2  # Assuming some activation function
    })
    print(f"Encoded instruction for ACT: {sim.value}")

    # Test instruction: POOL (Pooling)
    sim.step({
        opcode: OPCODE2BIN['POOL'][0], 
        src: 0x0000000C, 
        tar: 0x0000000D, 
        length: 2, 
        flags: 0  # Pooling type (e.g., 0 for max pooling)
    })
    print(f"Encoded instruction for POOL: {sim.value}")

    # Test instruction: SOFTMAX (Softmax)
    sim.step({
        opcode: OPCODE2BIN['SOFTMAX'][0], 
        src: 0x00000010, 
        tar: 0x00000011, 
        length: 5, 
        flags: 1  # Softmax type (e.g., 1 for approximate softmax)
    })
    print(f"Encoded instruction for SOFTMAX: {sim.value}")

    # Test instruction: ADD (Addition)
    sim.step({
        opcode: OPCODE2BIN['ADD'][0], 
        src: 0x00000020, 
        tar: 0x00000021, 
        length: 0,  # No length needed for addition
        flags: 0  # No additional flag
    })
    print(f"Encoded instruction for ADD: {sim.value}")

    # Optional: Export Verilog to file (if needed)
    # with open('assembler.v', 'w') as f:
    #     pyrtl.importexport.output_to_verilog(f)
    print("Simulation complete!")

# Run the main function
if __name__ == "__main__":
    main()
