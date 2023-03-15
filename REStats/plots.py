import os


def save_figs(figs, format="pdf", output_dir=None):
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    output_dir = output_dir or (curr_dir + os.sep + "../figs")
    
    for name in figs:
        outfile = f"{output_dir}/{name}.{format}"
        figs[name].savefig(outfile, bbox_inches="tight")