import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_iter", default=1000)

args = parser.parse_args()

print args

# parser = argparse.ArgumentParser()
# parser.add_argument('--input_svg', default='MLP_Annotated.svg')
# parser.add_argument('--output_pdf', default='myPlot.pdf')
# args = parser.parse_args()

# print "Converting: " + str(args.input_svg) + " to: " + str(args.output_pdf)

# drawing = svg2rlg(args.input_svg)
