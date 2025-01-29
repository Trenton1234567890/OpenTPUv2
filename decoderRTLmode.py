import pyrtl
from pyrtl import *
import configRTLmode as config
import isaRTLmode as isa
from pyrtl import importexport

# Define sizes based on configuration
DATASIZE = config.DWIDTH
MATSIZE = config.MATSIZE
ACCSIZE = config.ACC_ADDR_SIZE

# Define the decode function
def decode(instruction):
    # Define input and output wires
    accum_raddr = WireVector(ACCSIZE, 'accum_raddr')
    accum_waddr = WireVector(ACCSIZE, 'accum_waddr')
    accum_overwrite = WireVector(1, 'accum_overwrite')
    switch_weights = WireVector(1, 'switch_weights')
    weights_raddr = WireVector(config.WEIGHT_DRAM_ADDR_SIZE, 'weights_raddr')
    weights_read = WireVector(1, 'weights_read')

    ub_addr = WireVector(24, 'ub_addr')
    ub_raddr = WireVector(config.UB_ADDR_SIZE, 'ub_raddr')
    ub_waddr = WireVector(config.UB_ADDR_SIZE, 'ub_waddr')

    whm_length = WireVector(8, 'whm_length')
    rhm_length = WireVector(8, 'rhm_length')
    mmc_length = WireVector(16, 'mmc_length')
    act_length = WireVector(8, 'act_length')
    act_type = WireVector(3, 'act_type')

    rhm_addr = WireVector(config.HOST_ADDR_SIZE, 'rhm_addr')
    whm_addr = WireVector(config.HOST_ADDR_SIZE, 'whm_addr')

    dispatch_mm = WireVector(1, 'dispatch_mm')
    dispatch_act = WireVector(1, 'dispatch_act')
    dispatch_rhm = WireVector(1, 'dispatch_rhm')
    dispatch_whm = WireVector(1, 'dispatch_whm')
    dispatch_halt = WireVector(1, 'dispatch_halt')
    
    # New operation dispatch signals
    dispatch_pool = WireVector(1, 'dispatch_pool')  # Dispatch for pooling
    dispatch_softmax = WireVector(1, 'dispatch_softmax')  # Dispatch for softmax
    dispatch_add = WireVector(1, 'dispatch_add')  # Dispatch for add operation

    # Pooling-specific parameters (you may need to extend with kernel size and stride if required)
    pool_length = WireVector(8, 'pool_length')  # Length parameter for pooling
    
    # Softmax-specific parameters
    softmax_length = WireVector(8, 'softmax_length')  # Length for softmax

    # Add-specific parameters (e.g., addresses and other config)
    add_length = WireVector(8, 'add_length')  # Length for addition operation
    ub_raddr_2 = WireVector(config.UB_ADDR_SIZE, 'ub_raddr_2')  # Second UB read address. we may need to read 2 addresses from the UB. one for the actual matrix, and one for the bias. however, we can use the bias section of the UB kind of line a heap memory i think. 
    #We can reuse ub_raddr to represent the actual matrix, and ub_raddr_2 for the bias. We only need to write the output to one UB address: Thats the original, ub_waddr where the actual matrix used to be stored. this could be modified if we need to reference the "pre-addition" matrix at another point
    #but why would we need to do this? We can always report essential matrices back to the host memory via WHM, and additions are commutative, so long as we dont try to do something weird with (a+b)+a. its the same logic with (A*B)+A not being possible here, because its stupid in context.

    # Parse instruction
    op = instruction[isa.OP_START*8 : isa.OP_END*8]  # Adjusted slicing
    iflags = instruction[isa.FLAGS_START*8 : isa.FLAGS_END*8]
    ilength = instruction[isa.LEN_START*8 : isa.LEN_END*8]
    memaddr = instruction[isa.ADDR_START*8 : isa.ADDR_END*8]
    ubaddr = instruction[isa.UBADDR_START*8 : isa.UBADDR_END*8]

    with conditional_assignment:
        with op == isa.OPCODE2BIN['NOP'][0]:
            pass

        with op == isa.OPCODE2BIN['WHM'][0]:
            dispatch_whm |= 1
            ub_raddr |= ubaddr
            whm_addr |= memaddr
            whm_length |= ilength

        with op == isa.OPCODE2BIN['RW'][0]:
            weights_raddr |= memaddr
            weights_read |= 1

        with op == isa.OPCODE2BIN['MMC'][0]:
            dispatch_mm |= 1
            ub_addr |= ubaddr
            accum_waddr |= memaddr
            mmc_length |= ilength
            accum_overwrite |= iflags[isa.OVERWRITE_BIT]
            switch_weights |= iflags[isa.SWITCH_BIT]

        with op == isa.OPCODE2BIN['ACT'][0]:
            dispatch_act |= 1
            accum_raddr |= memaddr
            ub_waddr |= ubaddr
            act_length |= ilength
            act_type |= iflags[isa.ACT_FUNC_BITS]

        with op == isa.OPCODE2BIN['SYNC'][0]:
            pass

        with op == isa.OPCODE2BIN['RHM'][0]:
            dispatch_rhm |= 1
            rhm_addr |= memaddr
            ub_raddr |= ubaddr
            rhm_length |= ilength

        with op == isa.OPCODE2BIN['HLT'][0]:
            dispatch_halt |= 1

        with op == isa.OPCODE2BIN['POOL'][0]:
            dispatch_pool |= 1
            pool_length |= ilength  # You can modify this depending on the specifics of your pool operation

        with op == isa.OPCODE2BIN['SOFTMAX'][0]:
            dispatch_softmax |= 1
            softmax_length |= ilength  # Modify as needed based on softmax implementation

        with op == isa.OPCODE2BIN['ADD'][0]:
            dispatch_add |= 1
            add_length |= ilength  # Set length for the addition operation

            # Assign UB read addresses for both matrices (adjust depending on the instruction format)
            ub_raddr |= memaddr  # First UB address for the first matrix
            ub_raddr_2 |= (memaddr + add_length)  # Second UB address for the second matrix, adjust based on length
            
            # Set the start addresses for UB fetches (optional depending on your system design)
            ub_addr |= ubaddr  # UB address for the first matrix

    # Return all the signals, including new ones for the additional operations
    return (dispatch_mm, dispatch_act, dispatch_rhm, dispatch_whm, dispatch_halt,
            ub_addr, ub_raddr, ub_waddr, rhm_addr, whm_addr,
            rhm_length, whm_length, mmc_length, act_length,
            act_type, accum_raddr, accum_waddr, accum_overwrite,
            switch_weights, weights_raddr, weights_read,
            dispatch_pool, dispatch_softmax, dispatch_add,
            pool_length, softmax_length, add_length, ) 

#add_length will correspond to the matrix size that should be added for dynamic scalability. e.g if I need to do a (16x16)+(16x16), make sure the ilength is 00010000. assume square matrices 

print("decoder export completed.")
