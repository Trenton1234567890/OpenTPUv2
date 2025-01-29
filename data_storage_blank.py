
from functools import reduce
import pyrtl
from pyrtl import *
from pyrtl import rtllib
from pyrtl.rtllib import multipliers

def MMU_top(data_width, matrix_size, accum_size, ub_size, start, start_addr, nvecs, dest_acc_addr, overwrite, swap_weights, ub_rdata, accum_raddr, weights_dram_in, weights_dram_valid, dispatch_add_signal, add_length, add_data):
    '''
    MMU_top function for managing vector issuance and accumulator addressing.
    Outputs:
        ub_raddr: read address for unified buffer
    '''
    accum_waddr = Register(accum_size, name="mmu_top_accum_waddr")
    # Create a new addition accumulator register (one per element, or a vector register if needed)
    addition_accumulator = [Register(8, name=f"addition_accumulator_{i}") for i in range(matrix_size)]
    vec_valid = WireVector(1, name="mmu_top_vec_valid")
    overwrite_reg = Register(1, name="mmu_top_overwrite_reg")
    last = WireVector(1, name="mmu_top_last_vec")
    swap_reg = Register(1, name="mmu_top_swap_reg")

    busy = Register(1, name="mmu_top_busy")
    N = Register(len(nvecs), name="mmu_top_nvecs_counter")
    ub_raddr = Register(ub_size, name="mmu_top_ub_raddr")

    rtl_assert(~(start & busy), Exception("Cannot dispatch new MM instruction while previous instruction is still being issued."))

    # Single conditional_assignment block for all logic
    with conditional_assignment:
        # Case 1: Handle the start of the operation
        with start:
            # Initialize control signals and start issuing operations
            accum_waddr.next |= dest_acc_addr
            overwrite_reg.next |= overwrite
            swap_reg.next |= swap_weights
            busy.next |= 1
            N.next |= nvecs
            ub_raddr.next |= start_addr  # Begin issuing on the next cycle

        # Case 2: Handle the busy state and vector processing
        with busy:
            vec_valid |= 1
            swap_reg.next |= 0
            N.next |= N - 1
            with N == 1:  # Last vector
                last |= 1
                overwrite_reg.next |= 0
                busy.next |= 0
            with otherwise:  # More vectors to issue in the next cycle
                ub_raddr.next |= ub_raddr + 1
                accum_waddr.next |= accum_waddr + 1
                last |= 0

        # Case 3: Handle the addition logic for matrix addition
        with dispatch_add_signal:
            # Loop over the matrix elements for element-wise addition
            for i in range(matrix_size):  # Assuming n elements (adjust as needed)
                # Perform element-wise addition and store in the addition accumulator
                addition_accumulator[i].next |= ub_rdata[i] + add_data[8 * i : 8 * (i + 1)]  # Add data for element i
                                                                                                                                                 #somehow I need to get a list of the addition_accumulator[]s to mux into acc_out in order to be parsed into the activation function layer

    # Case 4: Call the MMU for memory management and vector operations
    acc_out, done = MMU(
        data_width=data_width,
        matrix_size=matrix_size,
        accum_size=accum_size,
        vector_in=ub_rdata,          # Input vector from UB
        accum_raddr=accum_raddr,     # Read address for accumulator
        accum_waddr=accum_waddr,     # Write address for accumulator
        vec_valid=vec_valid,         # Validation signal for the vector
        accum_overwrite=overwrite_reg, # Overwrite control for accumulator
        lastvec=last,                # Last vector flag
        switch_weights=swap_reg,     # Control to switch weights
        ddr_data=weights_dram_in,    # Weights data from DDR
        ddr_valid=weights_dram_valid # DDR weights validity signal
    )

    # Return the necessary outputs from the vector issue logic and MMU
    return ub_raddr, acc_out, busy, done