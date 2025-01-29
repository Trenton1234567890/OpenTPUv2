######################################################################################
# Note this file was created to get RTL for ISA, but it combines with decoder a bit. #
######################################################################################

import pyrtl

# Configuration Constants
ENDIANNESS = 'big'
INSTRUCTION_WIDTH_BYTES = 14

# Address Sizes
HOST_ADDR_SIZE = 8  # 64-bit addressing
DRAM_ADDR_SIZE = 5  # 33-bit addressing (TPU has 8 GB on-chip DRAM)
UB_ADDR_SIZE = 3    # 17-bit addressing for Unified Buffer
ACC_ADDR_SIZE = 2   # 12-bit addressing for accumulator

# Instruction Bit Sizes
OP_SIZE = 1
FLAGS_SIZE = 1
ADDR_SIZE = 8
LEN_SIZE = 1

# Instruction Parsing Bit Ranges
UBADDR_START = 0
UBADDR_END = 3
ADDR_START = 3
ADDR_END = 11
LEN_START = 11
LEN_END = 12
FLAGS_START = 12
FLAGS_END = 13
OP_START = 13
OP_END = 14

# Opcode mapping
OPCODE2BIN = {
    'NOP':  (0x0, 0, 0, 0),
    'WHM':  (0x1, UB_ADDR_SIZE, HOST_ADDR_SIZE, 1),
    'RW':   (0x2, DRAM_ADDR_SIZE, 0, 1),
    'MMC':  (0x3, UB_ADDR_SIZE, ACC_ADDR_SIZE, 1),
    'ACT':  (0x4, ACC_ADDR_SIZE, UB_ADDR_SIZE, 1),
    'SYNC': (0x5, 0, 0, 0),
    'RHM':  (0x6, HOST_ADDR_SIZE, UB_ADDR_SIZE, 1),
    'HLT':  (0x7, 0, 0, 0),
}

# Reverse mapping for binary to opcode
BIN2OPCODE = {v[0]: k for k, v in OPCODE2BIN.items()}

# Masks for flags
SWITCH_MASK =       0b00000001
CONV_MASK =         0b00000010
OVERWRITE_MASK =    0b00000100
ACT_FUNC_MASK =     0b00011000
FUNC_RELU_MASK =    0b00001000
FUNC_SIGMOID_MASK = 0b00010000

# Bit positions for flags
SWITCH_BIT        = 0
OVERWRITE_BIT     = 2
ACT_FUNC_BITS     = slice(3, 5)
FUNC_RELU_BIT     = 3
FUNC_SIGMOID_BIT  = 4

# Function to decode instruction
def decode_instruction(instruction):
    op = pyrtl.WireVector(OP_SIZE, 'op')
    flags = pyrtl.WireVector(FLAGS_SIZE, 'flags')
    addr = pyrtl.WireVector(ADDR_SIZE, 'addr')
    ub_addr = pyrtl.WireVector(UB_ADDR_SIZE, 'ub_addr')
    length = pyrtl.WireVector(LEN_SIZE, 'length')

    op |= instruction[OP_START:OP_END + 1]
    flags |= instruction[FLAGS_START:FLAGS_END + 1]
    addr |= instruction[ADDR_START:ADDR_END + 1]
    ub_addr |= instruction[UBADDR_START:UBADDR_END + 1]
    length |= instruction[LEN_START:LEN_END + 1]

    return op, flags, addr, ub_addr, length

# Example usage
def top_module():
    instruction = pyrtl.Input(INSTRUCTION_WIDTH_BYTES * 8, 'instruction')  # Input instruction in bits
    op, flags, addr, ub_addr, length = decode_instruction(instruction)

    # You can add further processing or outputs as needed

top_module()

# Export to Verilog (optional)
#with open('isaRTLmode.v', 'w') as f:
    #pyrtl.output_to_verilog(f)

print("ISA RTL mode export completed.")
