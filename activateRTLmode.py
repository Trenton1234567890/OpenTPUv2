import pyrtl
from pyrtl import *
from functools import reduce
import math

# Define activation functions

# ReLU vector: Applies ReLU activation element-wise to a vector
def relu_vector(vec, offset):
    assert offset <= 24
    return concat_list([select(d[-1] == 1, falsecase=d, truecase=Const(0, len(d)))[24 - offset:32 - offset] for d in vec])

# Sigmoid function: Uses a ROM block for efficient sigmoid calculation
def sigmoid(x):
    sigmoid_rom = RomBlock(bitwidth=8, addrwidth=3, name="sigmoid_rom", asynchronous=True, romdata={
        0: 128, 1: 187, 2: 225, 3: 243, 4: 251, 5: 254, 6: 255, 7: 255, 8: 255
    })
    x_gt_7 = reduce(lambda a, b: a | b, x[3:])  # OR of bits 3 and up for early exit check
    return select(x_gt_7, falsecase=sigmoid_rom[x[:3]], truecase=Const(255, bitwidth=8))

# Sigmoid vector: Applies sigmoid activation to each element in a vector
def sigmoid_vector(vec):
    return concat_list([sigmoid(x) for x in vec])

def restoring_division(dividend, divisor):
    quotient = pyrtl.Register(32, reset_value=0)
    remainder = pyrtl.Register(32, reset_value=0)

    for i in range(32):
        # Create unique names for each iteration
        quotient_temp = pyrtl.WireVector(32)#, name=f"quotient_temp_{i}")
        remainder_temp = pyrtl.WireVector(32)#, name=f"remainder_temp_{i}")

        # Shift remainder left and add the current bit of the dividend
        remainder_temp <<= pyrtl.shift_left_logical(remainder, 1) | dividend[31 - i]

        # Check if remainder_temp < divisor
        remainder_sub = remainder_temp - divisor
        remainder_is_negative = pyrtl.signed_lt(remainder_temp, divisor)

        # Restore the remainder if negative
        restored_remainder = remainder_sub + (remainder_is_negative * divisor)

        # Set the quotient bit based on the remainder comparison
        quotient_bit = pyrtl.mux(remainder_is_negative, Const(0, 1), Const(1, 1))

        # Update quotient by shifting and adding the new quotient bit
        quotient_temp <<= pyrtl.shift_left_logical(quotient, 1) | quotient_bit

    remainder.next <<= restored_remainder
    # Final updated value after the loop
    quotient.next <<= quotient_temp

    return quotient

def softmax_vector(vec):
    # Register to store exponential values (pipelined access)
    exp_values = [pyrtl.Register(32, reset_value=0) for _ in range(len(vec))]

    exp_rom = RomBlock(bitwidth=64, addrwidth=6, name="exp_rom", asynchronous=True, romdata={
        0: 1000, 1: 2718, 2: 7393, 3: 20279, 4: 54881, 5: 148413, 6: 403219, 7: 1099623,
        8: 2989579, 9: 8111552, 10: 22026468, 11: 59874042, 12: 162754791, 13: 441978729, 
        14: 1202608224, 15: 3269017177, 16: 8872070506, 17: 24098610722, 18: 65470398255, 
        19: 177136894994, 20: 479156299045, 21: 1304139822805, 22: 3547422477115, 
        23: 9647654117542, 24: 26111409342161, 25: 70783559130606, 26: 191721578915904, 
        27: 520441472998999, 28: 1415402889796095, 29: 3842872553299269, 30: 10469686632780338, 
        31: 28436122372830257, 32: 0})


    # ROM access control
    read_addr = pyrtl.Register(6, name="exp_rom_read_addr")
    exp_data_out = pyrtl.WireVector(32, name="exp_data_out")

    # Sequentially read from ROM
    exp_data_out <<= exp_rom[read_addr]

    with pyrtl.conditional_assignment:
        for i in range(len(vec)):    
            with read_addr == i:
                exp_values[i].next |= exp_data_out
        with read_addr != len(vec)-1:  # Update condition to reflect length of the vector
            read_addr.next |= read_addr + 1  # Increment if within bounds
        with read_addr == len(vec)-1:  # Keep steady when reaching the end of the vector
            read_addr.next |= 0


    # Accumulate the sum of exponentials
    temp_sum_exp = pyrtl.WireVector(32, name="temp_sum_exp")
    temp_sum_exp <<= sum(exp_values)

    # Store the accumulated sum in a register
    sum_exp = pyrtl.Register(32, name="sum_exp")
    sum_exp.next <<= temp_sum_exp

    # Calculate the softmax values using restoring division
    softmax_out_parts = []
    for i in range(len(vec)):
        temp_softmax = restoring_division(exp_values[i], sum_exp)
        softmax_out_parts.append(temp_softmax)

    # Concatenate all results into a single WireVector
    softmax_out = pyrtl.concat(*softmax_out_parts)

    return softmax_out









def max_pool_vector(vec, kernel_size, stride):
    # Calculate the number of windows
    num_windows = (len(vec) - kernel_size) // stride + 1
    pooled = []

    for win_start in range(num_windows):
        start_idx = win_start * stride
        window = vec[start_idx : start_idx + kernel_size]

        # Temporary Register to hold the running max within the loop
        temp_max = pyrtl.Register(len(window[0]), reset_value=0)

        # Loop through the remaining values in the window and update temp_max
        for val in window[1:]:
            with pyrtl.conditional_assignment:
                # Use a temporary register for comparison and assignment
                temp_max.next <<= pyrtl.mux(pyrtl.signed_gt(val, temp_max), val, temp_max)

        # Create a register to store the max value for the window
        max_val = pyrtl.Register(len(window[0]), reset_value=0)

        # Perform a single `.next` assignment outside the loop
        max_val.next <<= temp_max

        # Append the maximum value for the current window
        pooled.append(max_val)

    # Concatenate the pooled results into a single WireVector
    return concat_list(pooled)



# Main activation function
def act_top(start, start_addr, dest_addr, nvecs, func, accum_out, add_mmu_out):
    # WireVector inputs for control signals and memory addresses
    startwv = WireVector(1, "start_activation")  # 1-bit signal to start activation
    startwv |= start
    startaddrwv = WireVector(len(start_addr), "start_act_addr")
    startaddrwv |= start_addr
    destaddrwv = WireVector(len(dest_addr), "dest_act_addr")
    destaddrwv |= dest_addr

    # Registers to hold states during the activation process
    busy = Register(1, name='busy_act')
    accum_addr = Register(len(start_addr), name='accum_addr')
    ub_waddr_activation = Register(len(dest_addr), name='ub_waddr_activation')
    N = Register(len(nvecs), name='N')
    act_func = Register(len(func), name='act_func')

    # Prevent starting a new instruction while a previous one is still running
    rtl_assert(~(startwv & busy), Exception("Dispatching new activate instruction while previous instruction is still running."))

    # Define the activation functions based on `act_func`
    input_vals = WireVector(128, "input_vals")
    act_out = WireVector(128, "act_out")
    input_vals |= concat_list([x[:8] for x in accum_out])  # Slice if necessary (depends on the input format)

    # Create separate wire outputs for each activation function
    relu_out = relu_vector(accum_out, 24)
    sigmoid_out = sigmoid_vector(accum_out)
    softmax_out = softmax_vector(accum_out)
    maxpool_out = max_pool_vector(accum_out, kernel_size=2, stride=2)

        # Conditional assignment to update the state based on control signals
    with pyrtl.conditional_assignment:
        with startwv:  # New instruction being dispatched
            accum_addr.next |= start_addr
            ub_waddr_activation.next |= dest_addr
            N.next |= nvecs
            act_func.next |= func
            busy.next |= 1
        with busy:  # Process a new vector every cycle
            accum_addr.next |= accum_addr + 1
            ub_waddr_activation.next |= ub_waddr_activation + 1
            N.next |= N - 1
            with N == 1:  # End of current instruction
                busy.next |= 0

    # Use MultiSelector to select one of the activation function outputs
    with pyrtl.rtllib.muxes.MultiSelector(act_func, act_out) as ms:
        ms.option(0, add_mmu_out)  # Default input (no activation) on mmu_output_data rather than on multiplication data
        ms.option(1, relu_out)  # ReLU activation
        ms.option(2, sigmoid_out)  # Sigmoid activation
        ms.option(3, softmax_out)  # Softmax activation
        ms.option(4, maxpool_out)  # Max pooling (example with 2x2 pool)
        ms.option(5, input_vals)  # Default input (no activation)
        ms.option(6, input_vals)  # Default input (no activation)
        ms.option(7, input_vals)  # Default input (no activation)

    # Enable memory write based on the busy signal (only write when not busy)
    ub_write_enable = busy

    return accum_addr, ub_waddr_activation, act_out, ub_write_enable, busy
