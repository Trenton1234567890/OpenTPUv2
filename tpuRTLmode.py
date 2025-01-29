from pyrtl import *
from pyrtl.analysis import area_estimation, TimingAnalysis

from configRTLmode import *
from decoderRTLmode import decode
from matrixRTLmode import MMU_top
from activateRTLmode import act_top

############################################################
#  Control Signals
############################################################

accum_act_read_addr = WireVector(ACC_ADDR_SIZE, 'accum_act_read_addr')  # Activate unit read address for accumulator buffers
weights_dram_in = Input(64*8, "weights_dram_in")  # Input signal from weights DRAM controller
weights_dram_valid = Input(1, "weights_dram_valid")  # Valid bit for weights DRAM signal
system_halt = Output(1, "system_halt")  # When raised, stop simulation


############################################################
#  Instruction Memory and Program Counter (PC)
############################################################

IMem = MemBlock(bitwidth=INSTRUCTION_WIDTH, addrwidth=IMEM_ADDR_SIZE, name="InstructionMemory")
program_counter = Register(IMEM_ADDR_SIZE, "program_counter")
pc_increment = WireVector(1, "pc_increment")

# Program counter increment logic
with conditional_assignment:
    with pc_increment:
        program_counter.next |= program_counter + 1
pc_increment <<= 1  # right now, increment the PC every cycle
current_instruction = IMem[program_counter]

############################################################
#  Unified Buffer
############################################################

UnifiedBuffer = MemBlock(bitwidth=MATSIZE*DWIDTH, addrwidth=UB_ADDR_SIZE, max_write_ports=2, name="UnifiedBuffer")

# Address and data wires for MM read port
ub_mm_read_addr = WireVector(UnifiedBuffer.addrwidth, 'ub_mm_read_addr')  # MM UB read address
ub_mm_data_out = UnifiedBuffer[ub_mm_read_addr]

############################################################
#  Decoder
############################################################

(dispatch_mm_signal, dispatch_act_signal, dispatch_rhm_signal, dispatch_whm_signal,
 dispatch_halt_signal, ub_start_address, ub_decode_address, ub_destination_address, 
 rhm_decode_address, whm_decode_address, rhm_data_length, whm_data_length, 
 mmc_data_length, act_data_length, act_function_type, accum_read_address, 
 accum_write_address, accum_overwrite_enable, weights_switch_enable, 
 weights_read_address, weights_memory_read, dispatch_pool_signal, dispatch_softmax_signal, 
 dispatch_add_signal, pool_data_length, softmax_data_length, add_data_length) = decode(current_instruction)

system_halt <<= dispatch_halt_signal

############################################################
#  Addition Memory and Management
############################################################

# Memory block for storing addition matrix
AddMatrix = MemBlock(bitwidth=128, addrwidth=UB_ADDR_SIZE, name="AdditionMatrix") #each element should contain 128 bits/8 words
# Counter to track which element of AddMatrix is being accessed
add_counter = Register(UB_ADDR_SIZE, name="add_counter")
# Control signal to enable counter increment
add_increment = WireVector(1, "add_increment")
# Signal to determine if addition operation is complete
addition_done = WireVector(1, "addition_done")
# Signal to track whether addition is active
add_active = Register(1, "add_active")

# Trigger addition on dispatch_add_signal (pulse) and deactivate on completion
with conditional_assignment:
    with dispatch_add_signal:
        add_active.next |= 1  # Latch addition active on pulse
    with addition_done:
        add_active.next |= 0  # Clear addition active when done

# Determine when to increment the counter
add_increment <<= add_active & ~addition_done  # Increment as long as addition is active

# Increment the counter when addition is active
with conditional_assignment:
    with add_increment:
        add_counter.next |= add_counter + 1

# Access the addition matrix using the counter
addition_matrix_in = AddMatrix[add_counter]
addition_done <<= (add_counter == MATSIZE*DWIDTH)  # Example logic for completion


############################################################
#  Matrix Multiply Unit (MMU)
############################################################

ub_mm_addr_out, acc_mmu_output, add_mmu_output, mmu_busy_signal, mmu_done_signal, add_in_progress_wire = MMU_top(
    data_width=DWIDTH, matrix_size=MATSIZE, accum_size=ACC_ADDR_SIZE, ub_size=UB_ADDR_SIZE,
    start=dispatch_mm_signal, start_addr=ub_start_address, nvecs=mmc_data_length,
    dest_acc_addr=accum_write_address, overwrite=accum_overwrite_enable, 
    swap_weights=weights_switch_enable, ub_rdata=ub_mm_data_out, 
    accum_raddr=accum_act_read_addr, weights_dram_in=weights_dram_in, 
    weights_dram_valid=weights_dram_valid, 
    dispatch_add_signal=dispatch_add_signal, add_counter=add_counter,
    add_data=addition_matrix_in
#these last 3 control the addition operation
)

ub_mm_read_addr <<= ub_mm_addr_out

############################################################
#  Activation Unit
############################################################

act_accum_read_addr, ub_act_write_addr, act_output, ub_act_write_enable, act_unit_busy = act_top(
    start=dispatch_act_signal, start_addr=accum_read_address, dest_addr=ub_destination_address,
    nvecs=act_data_length, func=act_function_type, accum_out=acc_mmu_output, add_mmu_out=add_mmu_output
)
accum_act_read_addr <<= act_accum_read_addr

act_outputwv=WireVector(act_output.bitwidth,"act_outputW")
act_outputwv|=act_output
# Write the result of activation to the unified buffer
with conditional_assignment:
    with ub_act_write_enable:
        UnifiedBuffer[ub_act_write_addr] |= act_outputwv

############################################################
#  Read/Write Host Memory
############################################################

host_memory_read_addr = Output(HOST_ADDR_SIZE, "host_memory_read_addr")
host_memory_data_in = Input(DWIDTH*MATSIZE, "host_memory_data_in")
host_memory_read_enable = Output(1, "host_memory_read_enable")
host_memory_write_addr = Output(HOST_ADDR_SIZE, "host_memory_write_addr")
host_memory_data_out = Output(DWIDTH*MATSIZE, "host_memory_data_out")
host_memory_write_enable = Output(1, "host_memory_write_enable")

# Write Host Memory control logic
whm_vector_count = Register(len(whm_data_length), "whm_vector_count")
whm_ub_read_addr = Register(len(ub_decode_address), "whm_ub_read_addr")
whm_write_addr = Register(len(whm_decode_address), "whm_write_addr")
whm_unit_busy = Register(1, "whm_unit_busy")

ub_whm_output = UnifiedBuffer[whm_ub_read_addr]

host_memory_write_addr <<= whm_write_addr
host_memory_data_out <<= ub_whm_output

with conditional_assignment:
    with dispatch_whm_signal:
        whm_vector_count.next |= whm_data_length
        whm_ub_read_addr.next |= ub_decode_address
        whm_write_addr.next |= whm_decode_address
        whm_unit_busy.next |= 1
    with whm_unit_busy:
        whm_vector_count.next |= whm_vector_count - 1
        whm_ub_read_addr.next |= whm_ub_read_addr + 1
        whm_write_addr.next |= whm_write_addr + 1
        host_memory_write_enable |= 1
        with whm_vector_count == 1:
            whm_unit_busy.next |= 0

# Read Host Memory control logic
rhm_vector_count = Register(len(rhm_data_length), "rhm_vector_count")
rhm_read_addr = Register(len(rhm_decode_address), "rhm_read_addr")
rhm_unit_busy = Register(1, "rhm_unit_busy")
rhm_ub_write_addr = Register(len(ub_decode_address), "rhm_ub_write_addr")

with conditional_assignment:
    with dispatch_rhm_signal:
        rhm_vector_count.next |= rhm_data_length
        rhm_unit_busy.next |= 1
        host_memory_read_addr |= rhm_decode_address
        host_memory_read_enable |= 1
        rhm_read_addr.next |= rhm_read_addr + 1
        rhm_ub_write_addr.next |= ub_decode_address
    with rhm_unit_busy:
        rhm_vector_count.next |= rhm_vector_count - 1
        host_memory_read_addr |= rhm_read_addr
        host_memory_read_enable |= 1
        rhm_read_addr.next |= rhm_read_addr + 1
        rhm_ub_write_addr.next |= rhm_ub_write_addr + 1
        UnifiedBuffer[rhm_ub_write_addr] |= host_memory_data_in
        with rhm_vector_count == 1:
            rhm_unit_busy.next |= 0

############################################################
#  Weights Memory
############################################################

weights_dram_read_addr = Output(WEIGHT_DRAM_ADDR_SIZE, "weights_dram_read_addr")
weights_dram_memory_read = Output(1, "weights_dram_memory_read")

weights_dram_read_addr <<= weights_read_address
weights_dram_memory_read <<= weights_memory_read


############################################################
#  Synthesis and Analysis
############################################################

def run_synth():
    print("logic = {:2f} mm^2, mem={:2f} mm^2".format(*area_estimation()))
    t = TimingAnalysis()
    print("Max freq = {} MHz".format(t.max_freq()))
    print("\nRunning synthesis...")
    synthesize()
    print("logic = {:2f} mm^2, mem={:2f} mm^2".format(*area_estimation()))
    t = TimingAnalysis()
    print("Max freq = {} MHz".format(t.max_freq()))
    print("\nRunning optimizations...")
    optimize()
    gate_total = sum(1 for gate in working_block() if gate.op not in ('s', 'c'))
    print("Gate total: " + str(gate_total))
    print("logic = {:2f} mm^2, mem={:2f} mm^2".format(*area_estimation()))
    t = TimingAnalysis()
    print("Max freq = {} MHz".format(t.max_freq()))

# run_synth()
