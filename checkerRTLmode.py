import pyrtl
import argparse
import numpy as np

args = None

def equal(a1, a2):
    assert a1.shape == a2.shape, 'Result file shape mismatch.'
    if a1.dtype == np.int8:
        a1 = a1.astype(np.uint8)
    if a2.dtype == np.int8:
        a2 = a2.astype(np.uint8)
    for x, y in np.nditer([a1, a2]):
        assert x == y, 'Result value mismatch.'

def check(p1, p2, width=None):
    r1 = np.load(p1)
    r2 = np.load(p2)
    if not width:
        equal(r1, r2)
    else:
        r_width = r1.shape[1]
        if r_width <= width:
            r2 = r2[:, :r_width]
            equal(r1, r2)
        else:
            r2 = np.concatenate((r2[::2], r2[1::2]), axis=1)
            r2 = r2[:, :r_width]
            equal(r1, r2)

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', action='store', type=int, default=16,
                        help='HW WIDTH.')
    parser.add_argument('--gt32', action='store', default='gt32.npy',
                        help='Path to f32 ground truth result.')
    parser.add_argument('--sim32', action='store', default='sim32.npy',
                        help='Path to f32 simulator result.')
    parser.add_argument('--sim8', action='store', default='sim8.npy',
                        help='Path to i8 simulator result.')
    parser.add_argument('--hw8', action='store', default='hw8.npy',
                        help='Path to i8 hardware result.')
    args = parser.parse_args()

def main():
    parse_args()
    print(f'HW width set to {args.width}.')
    check(args.gt32, args.sim32, args.width)
    print('32-bit passed.')
    check(args.sim8, args.hw8)
    print('8-bit passed.')

    # Define hardware logic
    data_in = pyrtl.Input(8, 'data_in')
    data_out = pyrtl.Output(8, 'data_out')

    # Simple logic: output equals input
    data_out <<= data_in

    # Export to Verilog
    #with open('checker.v', 'w') as f:
        #pyrtl.importexport.output_to_verilog(f)
    print("Verilog export completed.")

# Run the main function
if __name__ == "__main__":
    main()
