import argparse
from S_run_aifeynman import run_aifeynman

parser = argparse.ArgumentParser()

parser.add_argument("--pathdir", type=str, help="Path to the directory containing the data file")
parser.add_argument("--filename", type=str, help="Name of the file containing the data")
parser.add_argument("--BF_try_time", type=float, default=60, help="Time limit for each brute force code call")
parser.add_argument("--BF_ops_file_type", type=str, default="14ops.txt", help="File containing the symbols to be used in the brute force code")
parser.add_argument("--polyfit_deg", type=int, default=3, help="Maximum degree of the polynomial tried by the polynomial fit routine")
parser.add_argument("--NN_epochs", type=int, default=2000, help="Number of epochs for the training")
parser.add_argument("--vars_name", type=list, default=[], help="List with the names of the variables")
parser.add_argument("--test_percentage", type=float, default=0, help="Percentage of the input data to be kept as the test set")

opts = parser.parse_args()

run_aifeynman(opts.pathdir, opts.filename, BF_try_time=opts.BF_try_time, BF_ops_file_type=opts.BF_ops_file_type, polyfit_deg=opts.polyfit_deg,
          NN_epochs=opts.NN_epochs, vars_name=opts.vars_name, test_percentage=opts.test_percentage)
          
