r"""

"""
import sys
if sys.version_info.major == 3:
    from pathlib import Path
else:
    from pathlib2 import Path
import argparse
from collections import defaultdict
from functools import partial
import json

import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)


DATACARD_TEMPLATE = r"""#
imax 1
jmax {jmax:d}
kmax *

--------------------------------------------------------------------------------
# shapes [process] [channel] [file] [histogram] [histogram_with_systematics]
shapes * * {input_file} $PROCESS
shapes {signal} * {input_file} $PROCESS

--------------------------------------------------------------------------------
# (no data)
bin {channel}

--------------------------------------------------------------------------------
# expected yields
{rate_block}

--------------------------------------------------------------------------------
# systematics
{syst_block}
"""

# TODO comment
MEMORY = set()


def read_hists_from_dir(plotter_output_dir,
                        step,
                        kinematic,
                        verbose):
    r"""

    """
    hist_path = '{}/{}'.format(step, kinematic)

    h_sig = None
    h_bkg_list = []
    for path in plotter_output_dir.glob('*.root'):
        process = path.stem

        # TODO use_data
        if process == 'Data':
            if verbose:
                print('skip {}'.format(path))
            continue

        if verbose:
            print('Found \'{}\''.format(process))

        root_file = ROOT.TFile(str(path), "READ")
        # TODO comment
        MEMORY.add(root_file)

        hist = root_file.Get(hist_path)
        hist.SetName(process)

        if process.startswith('VBF-EWKino_'):
            assert h_sig is None
            h_sig = hist
        else:
            h_bkg_list.append(hist)
    return h_sig, h_bkg_list


def retrieve_hists_from_stack(plotter_output_path, step, kinematic, verbose):
    raise NotImplementedError

    plotter_output_file = ROOT.TFile(plotter_output_path, "READ")
    # TODO comment
    MEMORY.add(plotter_output_file)

    canvas_path = '{}/{}'.format(step, kinematic)
    canvas = plotter_output_file.Get(canvas_path)
    stack_pad = canvas.GetPad(1)
    # TODO sanity-check

    h_sig = None
    h_bkg_stack = None
    for each in stack_pad.GetListOfPrimitives():
        if each.GetName() != kinematic:
            continue

        if verbose:
            print('Found \'{}\''.format(process))

        # TODO Entries are divided by the width of each bin.
        # https://github.com/BSM3G/Plotter/blob/master/src/Plotter.cc#L826-L832

        if isinstance(each, ROOT.TH1D):
            h_sig = each
        elif isinstance(each, ROOT.THStack):
            h_bkg_stack = each
        else:
            raise TypeError

    if h_sig is None:
        raise RuntimeError
    if h_bkg_stack is None:
        raise RuntimeError

    h_bkg_list = list(h_bkg_stack.GetHists())
    return h_sig, h_bkg_list


def manipulate_hist(hist, rebin_edges=None):
    # the name of input histogram should be the process.
    new_name = hist.GetName()

    # `combine` use '#' character for inline comments
    new_name = new_name.replace('#', '')

    if rebin_edges is None:
        hist = hist.Clone(new_name)
    else:
        rebin_nbins = len(rebin_edges) - 1
        hist = hist.Rebin(rebin_nbins, new_name, rebin_edges)
    return hist


def make_rate_block(channel_label, hist_list):
    r""" e.g.
    bin             bin1       bin1
    process         signal     background
    process         0          1
    rate            10         100
    """
    num_processes = len(hist_list)

    #
    process_row = [each.GetName() for each in hist_list]

    #
    process_id_row = [str(each) for each in range(0, num_processes)]

    # TODO
    rate_row = [str(each.Integral()) for each in hist_list]

    block = [
        ['bin'] + [channel_label] * len(hist_list),
        ['process'] + process_row,
        ['process'] + process_id_row,
        ['rate'] + rate_row
    ]
    block = [' '.join(each) for each in block]
    block = '\n'.join(block)
    return block

def make_systematics_scale_dict(data, process_list=None):
    if isinstance(data, dict):
        scale_dict = defaultdict(lambda: '-')
        scale_dict.update(data)
    else:
        scale_dict = defaultdict(lambda: data)

    # sanity-check
    if process_list is not None:
        for key in scale_dict.keys():
            if key not in process_list:
                err = "{} not in {}".format(key, process_list)
                raise RuntimeError(err)
    return scale_dict

def make_systematics_block(systematics_path, hist_list):
    r"""
    e.g.
    lumi     lnN    1.10       1.0
    bgnorm   lnN    1.00       1.3
    """
    process_list = [each.GetName() for each in hist_list]

    with open(systematics_path, 'r') as systematics_file:
        systematics_data = json.load(systematics_file, parse_float=str,
                                     parse_int=str)
    block = []
    for label, row_data in systematics_data.items():
        scale_dict = make_systematics_scale_dict(row_data['scale'],
                                                 process_list)
        row = [label, row_data['distribution']]
        row += [scale_dict[each] for each in process_list]
        row = ' '.join(row)
        block.append(row)
    block = '\n'.join(block)
    return block


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--plotter-output', type=Path, required=True,
                        help='')
    parser.add_argument('-s', '--systematics-file', type=str, required=True,
                        help='')
    parser.add_argument('--shape-file', type=str, default='shape.root',
                        help='')
    parser.add_argument('--datacard-file', type=str, default='datacard.combine',
                        help='')
    parser.add_argument('--step', type=str, default='QCDRejection',
                        help='')
    parser.add_argument('--kinematic', type=str, default='LargestDiJetMass',
                        help='')
    parser.add_argument('--channel', type=str, default='bin0',
                        help='channel id')
    parser.add_argument('--rebin-edges', type=float, nargs='+',
                        help='')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='')

    args = parser.parse_args()

    # TODO subparser
    # python make-datacard.py config -i some-dir -s syst.json -c config.file

    if args.rebin_edges is not None:
        args.rebin_edges = np.array(args.rebin_edges, dtype=np.float64)


    ############################################################################
    # Step1, read a plotter output directory/file
    ############################################################################
    if args.plotter_output.is_dir():
        h_sig, h_bkg_list = read_hists_from_dir(
            plotter_output_dir=args.plotter_output,
            step=args.step,
            kinematic=args.kinematic,
            verbose=args.verbose)
    else:
        raise NotImplementedError
        h_sig, h_bkg_list = retrieve_hists_from_stack(
            plotter_output_path=args.plotter_output,
            step=args.step,
            kinematic=args.kinematic,
            verbose=args.verbose)

    if h_sig is None:
        raise RuntimeError

    if len(h_bkg_list) == 0:
        raise RuntimeError

    ############################################################################
    # Step2, write a shape file
    ############################################################################
    shape_file = ROOT.TFile(args.shape_file , "RECREATE")
    shape_file.cd()

    manipulate_hist_fn = partial(manipulate_hist, rebin_edges=args.rebin_edges)

    h_sig = manipulate_hist_fn(h_sig)
    h_bkg_list = [manipulate_hist_fn(each) for each in h_bkg_list]

    shape_file.Write()

    if args.verbose:
        shape_file.Print()

    ############################################################################
    # Step3, write a datacard file
    ############################################################################
    # NOTE signal first
    # The first four lines labelled bin, process, process and rate give the
    # channel label, the process label, a process identifier (<=0 for signal, >0
    # for background) and the number of expected events respectively.
    hist_list = [h_sig] + h_bkg_list

    rate_block = make_rate_block(args.channel, hist_list)

    syst_block = make_systematics_block(args.systematics_file, hist_list)

    datacard = DATACARD_TEMPLATE.format(
        jmax=len(h_bkg_list),
        input_file=args.shape_file,
        signal=h_sig.GetName(),
        channel=args.channel,
        rate_block=rate_block,
        syst_block=syst_block,
    )

    if args.verbose:
        print(datacard)

    with open(args.datacard_file, 'w') as datacard_file:
        datacard_file.write(datacard)

    shape_file.Close()

if __name__ == '__main__':
    main()
