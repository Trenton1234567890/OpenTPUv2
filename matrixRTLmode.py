from functools import reduce
import pyrtl
from pyrtl import *
from pyrtl import rtllib
from pyrtl.rtllib import multipliers
import pyrtl.rtllib
from pyrtl.rtllib.muxes import MultiSelector
import pyrtl.rtllib.muxes


global_counter = 0  # To give unique numbers to each MAC
temp_count = temp_count_1 = temp_count_2=0
def MAC(data_width, matrix_size, data_in, acc_in, switchw, weight_in, weight_we, weight_tag):
    '''Multiply-Accumulate unit with programmable weight.
    Inputs
    data_in: The 8-bit activation value to multiply by weight.
    acc_in: 32-bit value to accumulate with product.
    switchw: Control signal; when 1, switch to using the other weight buffer.
    weight_in: 8-bit value to write to the secondary weight buffer.
    weight_we: When high, weights are being written; if tag matches, store weights.
               Otherwise, pass them through with incremented tag.
    weight_tag: If equal to 255, weight is for this row; store it.

    Outputs
    output_accum: Result of the multiply accumulate; moves one cell down to become acc_in.
    data_reg: data_in, stored in a pipeline register for cell to the right.
    switch_reg: switchw, stored in a pipeline register for cell to the right.
    weight_reg: weight_in, stored in a pipeline register for cell below.
    weight_we_reg: weight_we, stored in a pipeline register for cell below.
    weight_tag_reg: weight_tag, incremented and stored in a pipeline register for cell below
    '''
    global global_counter
    probe(weight_in,f"weight_inMAC{global_counter}")
    # Check lengths of inputs
    if len(weight_in) != len(data_in) != data_width:
        raise Exception("Expected 8-bit value in MAC.")
    if len(switchw) != len(weight_we) != 1:
        raise Exception("Expected 1-bit control signal in MAC.")

    # Use two buffers to store weight and next weight to use.
    current_weight_buffer = Register(len(weight_in), name=f"current_weight_buffer_{global_counter}")
    next_weight_buffer = Register(len(weight_in), name=f"next_weight_buffer_{global_counter}")
    current_buffer=WireVector(1,f"current_buffer_{global_counter}")
    # Track which buffer is current and which is secondary.
    current_buffer_reg = Register(1, name=f"current_buffer_reg_{global_counter}")
    with conditional_assignment:
        with switchw:
            current_buffer_reg.next |= ~current_buffer_reg
    current_buffer |= current_buffer_reg ^ switchw  # reflects change in the same cycle switchw goes high

    # When told, store a new weight value in the secondary buffer
    with conditional_assignment:
        with weight_we & (weight_tag == Const(matrix_size - 1)):
            with current_buffer == 0:  # If 0, current_weight_buffer is current; if 1, next_weight_buffer is current
                next_weight_buffer.next |= weight_in
            with otherwise:
                current_weight_buffer.next |= weight_in

    # Do the actual MAC operation    
    #weight = WireVector(,"weight")
    weight = select(current_buffer, next_weight_buffer, current_weight_buffer)
    probe(weight, f"weight_{global_counter}")  # Add a probe with a unique name for debugging
    probe(data_in, f"data_in_{global_counter}")  # Add a probe with a unique name for debugging
    #probe(weight, "weight" + str(globali))
    global_counter += 1
    #inlen = max(len(weight), len(data_in)) #NONNATIVE
    #product = weight.sign_extended(inlen*2) * data_in.sign_extended(inlen*2) #NONNATIVE
    #product = product[:inlen*2]
    product = signed_mult(weight, data_in)[:32] ##################################################ORIGINAL, BUT KINDA BROKEN
    probe(product, f"product_{global_counter}")  # Add a probe with a unique name for debugging
    #plen = len(weight) + len(data_in)
    #product = weight.sign_extended(plen) * data_in.sign_extended(plen)
    #product = product[:plen]
    l = max(len(product), len(acc_in)) + 1
    output_accum = (product.sign_extended(l) + acc_in.sign_extended(l))[:-1]

    if len(output_accum) > 32:
        output_accum = output_accum[:32]
                
    # For values that need to be forwarded to the right/bottom, store in pipeline registers
    data_reg = Register(len(data_in), name=f"data_reg_{global_counter}")  # Pipeline register, holds data value for cell to the right
    data_reg.next <<= data_in
    switch_reg = Register(1, name=f"switch_reg_{global_counter}")  # Pipeline register, holds switch control signal for cell to the right
    switch_reg.next <<= switchw
    accum_reg = Register(len(output_accum), name=f"accum_reg_{global_counter}")  # Output value for MAC below
    accum_reg.next <<= output_accum
    weight_reg = Register(len(weight_in), name=f"weight_reg_{global_counter}")  # Pipeline register, holds weight input for cell below
    weight_reg.next <<= weight_in
    weight_we_reg = Register(1, name=f"weight_we_reg_{global_counter}")  # Pipeline register, holds weight write enable signal for cell below
    weight_we_reg.next <<= weight_we
    weight_tag_reg = Register(len(weight_tag), name=f"weight_tag_reg_{global_counter}")  # Pipeline register, holds weight tag for cell below
    weight_tag_reg.next <<= (weight_tag + 1)[:len(weight_tag)]  # Increment tag as it passes down rows

    return accum_reg, data_reg, switch_reg, weight_reg, weight_we_reg, weight_tag_reg

    
def MMArray(data_width, matrix_size, data_in, new_weights, weights_in, weights_we):
    '''
    data_in: 256-array of 8-bit activation values from systolic_setup buffer
    new_weights: 256-array of 1-bit control values indicating that new weight should be used
    weights_in: output of weight FIFO (8 x matsize x matsize bit wire)
    weights_we: 1-bit signal to begin writing new weights into the matrix
    '''

    # For signals going to the right, store in a variable; for signals going down, keep a list
    weights_input_top = [WireVector(data_width) for _ in range(matrix_size)]  # input weights to top row
    weights_input_last = [x for x in weights_input_top]
    weights_enable_top = [WireVector(1) for _ in range(matrix_size)]  # weight write enable to top row
    weights_enable_last = [x for x in weights_enable_top]
    weights_tag_top = [WireVector(data_width) for _ in range(matrix_size)]  # weight row tag to top row
    weights_tag_last = [x for x in weights_tag_top]
    data_output = [Const(0) for _ in range(matrix_size)]  # will hold output from final row

    # Build array of MACs
    for i in range(matrix_size):  # for each row
        input_data = data_in[i]
        switch_input = new_weights[i]
        for j in range(matrix_size):  # for each column
            acc_output, input_data, switch_input, current_weight, current_weight_enable, current_weight_tag = MAC(
                data_width, matrix_size, input_data, data_output[j], switch_input, 
                weights_input_last[j], weights_enable_last[j], weights_tag_last[j]
            )
            weights_input_last[j] = current_weight
            weights_enable_last[j] = current_weight_enable
            weights_tag_last[j] = current_weight_tag
            data_output[j] = acc_output
    
    # Handle weight reprogramming
    programming_signal = Register(1, name="programming_signal")  # When 1, indicates that new weights are being loaded
    size_index = 1
    while pow(2, size_index) < matrix_size:
        size_index += 1
    programming_step_register = Register(size_index, name="programming_step_register")  # 256 steps to program new weights (also serves as tag input)

    with conditional_assignment:
        with weights_we & (~programming_signal):
            programming_signal.next |= 1
        with programming_signal & (programming_step_register == matrix_size - 1):
            programming_signal.next |= 0
        with otherwise:
            pass
        with programming_signal:  # while programming, increment state each cycle
            programming_step_register.next |= programming_step_register + 1
        with otherwise:
            programming_step_register.next |= Const(0)

    # Divide FIFO output into rows (each row datawidth x matrixsize bits)
    row_size = data_width * matrix_size
    weight_array = [weights_in[i * row_size: i * row_size + row_size] for i in range(matrix_size)]

    # Mux the wire for this row
    current_weights_wire = mux(programming_step_register, *weight_array)
    # Split the wire into an array of 8-bit values
    current_weights = [current_weights_wire[i * data_width:i * data_width + data_width] for i in reversed(range(matrix_size))]

    # Connect top row to input and control signals
    for i, win in enumerate(weights_input_top):
        # From the current 256-array, select the byte for this column
        win <<= current_weights[i]
    for weight_enable in weights_enable_top:
        # Whole row gets the same signal: high when programming new weights
        weight_enable <<= programming_signal
    for weight_tag in weights_tag_top:
        # Tag is the same for the whole row; use state index (runs from 0 to 255)
        weight_tag <<= programming_step_register

    return [x.sign_extended(32) for x in data_output]


def accum(size, data_in, waddr, wen, wclear, raddr, lastvec):
    '''A single 32-bit accumulator with 2^size 32-bit buffers.
    On wen, writes data_in to the specified address (waddr) if wclear is high;
    otherwise, it performs an accumulate at the specified address (buffer[waddr] += data_in).
    lastvec is a control signal indicating that the operation being stored now is the
    last vector of a matrix multiply instruction (at the final accumulator, this becomes
    a "done" signal).
    '''
    global temp_count
    mem = MemBlock(bitwidth=32, addrwidth=size,name='accum_mem')
    
    # Writes
    with conditional_assignment:
        with wen:
            with wclear:
                mem[waddr] |= data_in
            with otherwise:
                mem[waddr] |= (data_in + mem[waddr])[:mem.bitwidth]

    # Read
    data_out = WireVector(32,name=f"accum_data_out_{temp_count}")
    data_out|=mem[raddr]
    
    # Pipeline registers with distinct names
    write_addr_register = Register(len(waddr), name=f"write_addr_register{temp_count}")
    write_addr_register.next <<= waddr
    write_enable_register = Register(1, name=f"write_enable_register_{temp_count}")
    write_enable_register.next <<= wen
    write_clear_register = Register(1, name=f"write_clear_register_{temp_count}")
    write_clear_register.next <<= wclear
    last_vector_register = Register(1, name=f"last_vector_register_{temp_count}")
    last_vector_register.next <<= lastvec
    temp_count=temp_count+1
    return data_out, write_addr_register, write_enable_register, write_clear_register, last_vector_register

def accumulators(accsize, datas_in, waddr, we, wclear, raddr, lastvec):
    '''
    Produces array of accumulators of same dimension as datas_in.
    '''
    wein=WireVector(1,'accum_wen')
    wclearin=WireVector(1,'accum_wclear')
    lastvecin=WireVector(1,'accum_lastvec')

    accout = [ None for i in range(len(datas_in)) ]
    waddrin = waddr
    wein |= we
    wclearin |= wclear
    lastvecin |= lastvec

 # Create WireVectors to monitor accout values
    accout_wv = [WireVector(32, f'acc_out_{i}') for i in range(len(datas_in))]

    for i,x in enumerate(datas_in): #x is a date element
        dout, waddrin, wein, wclearin, lastvecin = accum(accsize, x, waddrin, wein, wclearin, raddr, lastvecin)
        accout[i] = dout
        done = lastvecin
        accout_wv[i] |= dout
        #should be 16 accouts with 32 bits each

    return accout, done


def FIFO(matsize, mem_data, mem_valid, advance_fifo):
    '''
    FIFO function for handling data from DRAM controller.
    '''

    # Calculate buffer sizes
    totalsize = matsize * matsize  # total size of a tile in bytes
    tilesize = totalsize * 8  # total size of a tile in bits
    ddrwidth = int(len(mem_data) / 8)  # width from DDR in bytes
    size = 1
    while pow(2, size) < (totalsize / ddrwidth):
        size += 1

    # State and control signals
    state_register = Register(size, name="fifo_state")
    startup_register = Register(1, name="fifo_startup")
    startup_register.next <<= 1

    # Top row buffer for DDR-width chunks
    topbuf = [Register(ddrwidth * 8, name=f"fifo_topbuf_{i}") for i in range(max(1, int(totalsize / ddrwidth)))]

    # FIFO advancement latch and clear signal
    droptile_register = Register(1, name="fifo_droptile")
    clear_droptile_wire = WireVector(1, name="fifo_clear_droptile")
    with conditional_assignment:
        with advance_fifo:
            droptile_register.next |= 1
        with clear_droptile_wire:
            droptile_register.next |= 0

    # Write to buffer on valid memory signal
    with conditional_assignment:
        with mem_valid:
            state_register.next |= state_register + 1
            for i, reg in enumerate(reversed(topbuf)):
                with state_register == Const(i, bitwidth=size):
                    reg.next |= mem_data

    # FIFO full control
    full_register = Register(1, name="fifo_full")
    cleartop_wire = WireVector(1, name="fifo_cleartop")
    with conditional_assignment:
        print(len(topbuf))
        with mem_valid & (state_register == Const(len(topbuf) - 1)):
            full_register.next |= 1
        with cleartop_wire:
            full_register.next |= 0

    # Secondary buffers and empty flags
    buf2, buf3, buf4 = (
        Register(tilesize, name="fifo_buf2"),
        Register(tilesize, name="fifo_buf3"),
        Register(tilesize, name="fifo_buf4"),
    )
    empty2, empty3, empty4 = (
        Register(1, name="fifo_empty2"),
        Register(1, name="fifo_empty3"),
        Register(1, name="fifo_empty4"),
    )

    # Move data between buffers based on state and empty/full flags
    with conditional_assignment:
        with ~startup_register:
            empty2.next |= 1
            empty3.next |= 1
            empty4.next |= 1
        with full_register & empty2:
            buf2.next |= concat_list(topbuf)
            cleartop_wire |= 1
            empty2.next |= 0
        with empty3 & ~empty2:
            buf3.next |= buf2
            empty3.next |= 0
            empty2.next |= 1
        with empty4 & ~empty3:
            buf4.next |= buf3
            empty4.next |= 0
            empty3.next |= 1
        with droptile_register:
            empty4.next |= 1
            clear_droptile_wire |= 1

    ready_signal = startup_register & (~empty4) & (~droptile_register)

    return buf4, ready_signal, full_register

def systolic_setup(data_width, matsize, vec_in, waddr, valid, clearbit, lastvec, switch):
    '''
    Systolic setup function for feeding vectors along diagonals.
    '''

    # Register for address, write enable, clear, and done signals
    addrreg = Register(len(waddr), name="sys_addrreg")
    addrreg.next <<= waddr
    wereg = Register(1, name="sys_wereg")
    wereg.next <<= valid
    clearreg = Register(1, name="sys_clearreg")
    clearreg.next <<= clearbit
    donereg = Register(1, name="sys_donereg")
    donereg.next <<= lastvec
    topreg = Register(data_width, name="sys_topreg")

    # Diagonal buffers for systolic feed
    firstcolumn = [topreg,] + [ Register(data_width, name=f"sys_firstcol_{i}") for i in range(matsize - 1)]
    lastcolumn = [ None for i in range(matsize) ]
    lastcolumn[0] = topreg

    # Generate switch signals to matrix; propagate down diagonally
    switchout = [ None for i in range(matsize) ]
    switchout[0] = Register(1, name="sys_switchout_0")
    switchout[0].next <<= switch
    for i in range(1, len(switchout)):
        switchout[i] = Register(1, name=f"sys_switchout_{i}")
        switchout[i].next <<= switchout[i - 1]

    # Control pipeline for address, write enable, clear, and done signals
    addrout = addrreg
    weout = wereg
    clearout = clearreg
    doneout = lastvec
    # for i in range(0, matsize):
    #     addrout = Register(len(addrout), name=f"sys_addrout_{i}")
    #     addrout.next <<= addrreg
    #     weout = Register(1, name=f"sys_weout_{i}")
    #     weout.next <<= wereg
    #     clearout = Register(1, name=f"sys_clearout_{i}")
    #     clearout.next <<= clearreg
    #     doneout = Register(1, name=f"sys_doneout_{i}")
    #     doneout.next <<= donereg

    for i in range(0, matsize):
        a = Register(len(addrout), name=f"sys_addrout_{i}")
        a.next <<= addrout
        addrout = a
        w = Register(1,name=f"sys_weout_{i}")
        w.next <<= weout
        weout = w
        c = Register(1, name=f"sys_clearout_{i}")
        c.next <<= clearout
        clearout = c
        d = Register(1, name=f"sys_doneout_{i}")
        d.next <<= doneout
        doneout = d

    # Generate buffers in a diagonal pattern
    for row in range(1, matsize):
        left = firstcolumn[row]
        lastcolumn[row] = left
        for column in range(0,row):
            buf = Register(data_width, name=f"sys_buf_{row}_{column}")
            buf.next <<= left
            left = buf
            lastcolumn[row] = left

    # Connect first column to input data
    datain = [ vec_in[i*data_width : i*data_width+data_width] for i in range(matsize) ]
    for din, reg in zip(datain, firstcolumn):
        reg.next <<= din

    return lastcolumn, switchout, addrout, weout, clearout, doneout



def MMU(data_width, matrix_size, accum_size, vector_in, accum_raddr, accum_waddr, vec_valid, accum_overwrite, lastvec, switch_weights, ddr_data, ddr_valid):
    '''
    MMU setup and execution function.
    '''

    # Calculate sizes for register bitwidths
    logn1 = 1
    while pow(2, logn1) < (matrix_size + 1):
        logn1 += 1
    logn = 1
    while pow(2, logn) < matrix_size:
        logn += 1

    # Registers and wire declarations with distinct names
    programming = Register(1, name="mmu_programming")
    waiting = WireVector(1, name="mmu_waiting")
        
    weights_wait = Register(logn1, name="mmu_weights_wait")
    weights_count = Register(logn, name="mmu_weights_count")
    startup = Register(1, name="mmu_startup")
    startup.next <<= 1

    weights_we = WireVector(1, name="mmu_weights_we")
    done_programming = WireVector(1, name="mmu_done_programming")
    first_tile = Register(1, name="mmu_first_tile")

    # FIFO setup
    weights_tile, tile_ready, full = FIFO(
        matsize=matrix_size,
        mem_data=ddr_data,
        mem_valid=ddr_valid,
        advance_fifo=done_programming
    )

    # Systolic setup
    matin, switchout, addrout, weout, clearout, doneout = systolic_setup(
        data_width=data_width,
        matsize=matrix_size,
        vec_in=vector_in,
        waddr=accum_waddr,
        valid=vec_valid,
        clearbit=accum_overwrite,
        lastvec=lastvec,
        switch=switch_weights
    )

    # Matrix Multiply Array and Accumulators
    mouts = MMArray(
        data_width=data_width,
        matrix_size=matrix_size,
        data_in=matin,
        new_weights=switchout,
        weights_in=weights_tile,
        weights_we=weights_we
    )

    accout, done = accumulators(
        accsize=accum_size,
        datas_in=mouts,
        waddr=addrout,
        we=weout,
        wclear=clearout,
        raddr=accum_raddr,
        lastvec=doneout
    )

    # Switch start signal and total wait count
    switchstart = switchout[0]
    totalwait = Const(matrix_size + 1)
    waiting <<= weights_wait != totalwait

    # Conditional assignments for programming weights and managing FIFO
    with conditional_assignment:
        with ~startup:
            weights_wait.next |= totalwait
        with waiting:
            weights_wait.next |= weights_wait + 1
        with ~first_tile & tile_ready:
            weights_wait.next |= totalwait
            programming.next |= 1
            weights_count.next |= 0
            first_tile.next |= 1
        with switchstart:
            weights_wait.next |= 0
            programming.next |= 1
            weights_count.next |= 0
        with programming:
            with weights_count == Const(matrix_size - 1):
                programming.next |= 0
                done_programming |= 1
            with otherwise:
                weights_count.next |= weights_count + 1
                weights_we |= 1

    return accout, done

def MMU_top(data_width, matrix_size, accum_size, ub_size, start, start_addr, nvecs, dest_acc_addr, overwrite, swap_weights, ub_rdata, accum_raddr, weights_dram_in, weights_dram_valid, dispatch_add_signal, add_counter, add_data):
    '''
    MMU_top function for managing vector issuance and accumulator addressing.
    Outputs:
        ub_raddr: read address for unified buffer
    '''
    accum_waddr = Register(accum_size, name="mmu_top_accum_waddr")
    addition_accumulator = Register(128, name=f"addition_accumulator")  # Keep as Register for storage
    vec_valid = WireVector(1, name="mmu_top_vec_valid")
    overwrite_reg = Register(1, name="mmu_top_overwrite_reg")
    last = WireVector(1, name="mmu_top_last_vec")
    swap_reg = Register(1, name="mmu_top_swap_reg")
    busy = Register(1, name="mmu_top_busy")
    N = Register(len(nvecs), name="mmu_top_nvecs_counter")
    ub_raddr = Register(ub_size, name="mmu_top_ub_raddr")

    rtl_assert(~(start & busy), Exception("Cannot dispatch new MM instruction while previous instruction is still being issued."))

    # New register to track addition in progress
    add_in_progress = Register(1, name="mmu_top_add_in_progress")
    add_in_progress_wire = WireVector(1, name="add_in_progress_wire")
    add_in_progress_wire |= add_in_progress

    # Single conditional_assignment block for all logic
    with conditional_assignment:
        # Case 1: Handle the start of the operation
        with start:
            accum_waddr.next |= dest_acc_addr
            overwrite_reg.next |= overwrite
            swap_reg.next |= swap_weights
            busy.next |= 1
            N.next |= nvecs
            ub_raddr.next |= start_addr

        # Case 2: Handle the busy state and vector processing
        with busy:
            vec_valid |= 1
            swap_reg.next |= 0
            N.next |= N - 1
            with N == 1:
                last |= 1
                overwrite_reg.next |= 0
                busy.next |= 0
            with otherwise:
                ub_raddr.next |= ub_raddr + 1
                accum_waddr.next |= accum_waddr + 1
                last |= 0

        # Case 3: Handle the addition logic for matrix addition
        # Latch the addition in progress when dispatch_add_signal pulses
        with dispatch_add_signal:
            add_in_progress.next |= 1
            ub_raddr.next |= start_addr  # Reset this to read an addition from the UB.

        # Perform addition if addition is in progress
        with add_in_progress:
            addition_accumulator.next |= ub_rdata + add_data  # Store the result in the accumulator 
            # Clear the addition in progress flag when addition is done
            with add_counter == matrix_size - 1:
                add_in_progress.next |= 0
            ub_raddr.next |= ub_raddr + 1

    # Call the MMU for memory management and vector operations
    mult_acc_out, done = MMU(
        data_width=data_width,
        matrix_size=matrix_size,
        accum_size=accum_size,
        vector_in=ub_rdata,
        accum_raddr=accum_raddr,
        accum_waddr=accum_waddr,
        vec_valid=vec_valid,
        accum_overwrite=overwrite_reg,
        lastvec=last,
        switch_weights=swap_reg,
        ddr_data=weights_dram_in,
        ddr_valid=weights_dram_valid
    )

    # Returning the wirevectors that are connected to the registers
    return ub_raddr, mult_acc_out, addition_accumulator, busy, done, add_in_progress_wire

    

'''
Do we need full/stall signal from Matrix? Would need to stop SRAM out from writing to systolic setup
Yes: MMU needs to track when both buffers used and emit such a signal

The timing systems for weights programming are wonky right now. Both rtl_asserts are failing, but the
right answer comes out if you ignore that. It looks like the state machine that counts time since the
last weights programming keeps restarting, so the MMU thinks it's always programming weights?

Control signals propagating down systolic_setup to accumulators:
-Overwrite signal (default: accumulate)
-New accumulator address value (default: add 1 to previous address)
-Done signal?
'''

def testall(input_vectors, weights_vectors):
    DATWIDTH = 8
    MATSIZE = 4
    ACCSIZE = 8

    L = len(input_vectors)

    ins = [probe(Input(DATWIDTH)) for i in range(MATSIZE)]
    invec = concat_list(ins)
    swap = Input(1, 'swap')
    waddr = Input(8,'waddr')
    lastvec = Input(1,'lastvec')
    valid = Input(1,'valid')
    raddr = Input(8, "raddr")  # accumulator read address to read out answers
    donesig = Output(1, "donesig")

    outs = [Output(32, name="out{}".format(str(i))) for i in range(MATSIZE)]

    # Flatten and concatenate weights into single chunks
    flat_weights = [item for sublist in weights_vectors for item in sublist]
    chunk_size = 64 * 8
    weight_chunks = [reduce(lambda x, y: (x << 8) + y, flat_weights[i:i + chunk_size // 8])
                     for i in range(0, len(flat_weights), chunk_size // 8)]

    weightsdata = Input(chunk_size,'weightsdata')
    weightsvalid = Input(1,'weightsvalid')

    accout, done = MMU(
        data_width=DATWIDTH,
        matrix_size=MATSIZE,
        accum_size=ACCSIZE,
        vector_in=invec,
        accum_raddr=raddr,
        accum_waddr=waddr,
        vec_valid=valid,
        accum_overwrite=Const(0),
        lastvec=lastvec,
        switch_weights=swap,
        ddr_data=weightsdata,
        ddr_valid=weightsvalid
    )

    donesig <<= done
    for out, accout in zip(outs, accout):
        out <<= accout

    sim_trace = SimulationTrace()
    sim = FastSimulation(tracer=sim_trace)

    # Initialize inputs
    din = {swap: 0, waddr: 0, lastvec: 0, valid: 0, raddr: 0, weightsdata: 0, weightsvalid: 0}
    din.update({ins[j]: 0 for j in range(MATSIZE)})

    # Simulate startup
    for _ in range(10):
        sim.step(din)

    # Load weights into FIFO via DDR
    for block in weight_chunks:
        d = din.copy()
        d.update({weightsdata: block, weightsvalid: 1})
        sim.step(d)

    # Allow for weights propagation
    for _ in range(MATSIZE * 2):
        sim.step(din)

    # Send first row of input data with swap signal
    d = din.copy()
    d.update({ins[j]: input_vectors[0][j] for j in range(MATSIZE)})
    d.update({swap: 1, valid: 1})
    sim.step(d)

    # Continue sending vectors
    for i in range(1, L):
        d = din.copy()
        d.update({ins[j]: input_vectors[i][j] for j in range(MATSIZE)})
        d.update({waddr: i, lastvec: 1 if i == L - 1 else 0, valid: 1})
        sim.step(d)

    # Wait for propagation to complete
    for _ in range(L * 2):
        sim.step(din)

    # Read out accumulated results
    for i in range(L):
        d = din.copy()
        d[raddr] = i
        sim.step(d)

    # Output trace to VCD file
    with open('trace.vcd', 'w') as f:
        sim_trace.print_vcd(f)


if __name__ == "__main__":
    #weights = [[1, 10, 10, 2], [3, 9, 6, 2], [6, 8, 2, 8], [4, 1, 8, 6]]  # transposed
    #weights = [[4, 1, 8, 6], [6, 8, 2, 8], [3, 9, 6, 2], [1, 10, 10, 2]]  # tranposed, reversed
    #weights = [[1, 3, 6, 4], [10, 9, 8, 1], [10, 6, 2, 8], [2, 2, 8, 6]]
    weights = [[2, 2, 8, 6], [10, 6, 2, 8], [10, 9, 8, 1], [1, 3, 6, 4]]  # reversed

    vectors = [[12, 7, 2, 6], [21, 21, 18, 8], [1, 4, 18, 11], [6, 3, 25, 15], [21, 12, 1, 15], [1, 6, 13, 8], [24, 25, 18, 1], [2, 5, 13, 6], [19, 3, 1, 17], [25, 10, 20, 10]]

    testall(vectors, weights)
#startflag=pyrtl.Input(1, 'start')
#nvecsIn=pyrtl.Input(3,'nvecs')
#dramValid=pyrtl.Input(1,'dramValid')
#weights_dram_in_wire=pyrtl.Input(256,'weights_dram_in')
#ubrdata=pyrtl.Input(256,'ub_rdata')
#MMU_top(data_width=8, matrix_size=4, accum_size=8, ub_size=128, start=startflag, start_addr=0, nvecs=nvecsIn, dest_acc_addr=3, overwrite=0, swap_weights=1, ub_rdata=ubrdata, accum_raddr=0, weights_dram_in=weights_dram_in_wire, weights_dram_valid=dramValid)

#with open('matrix.v', 'w') as f:
    #pyrtl.importexport.output_to_verilog(f)