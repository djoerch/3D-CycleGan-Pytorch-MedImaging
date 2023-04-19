#!/usr/bin/env python

import os
import re

from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import pandas as pd
import seaborn as sns


DESC = dedent(
    """
    Read loss log file and plot metrics.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example calls:
      {filename} -i /path/to/loss_log.txt -o /path/to/output/folder --basename test
      {filename} -i loss_log.txt -o outputs --basename cycle_gan
    """.format(  # noqa: E501
        filename=os.path.basename(__file__)
    )
)


def build_argparser():

    p = ArgumentParser(
        description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter
    )
    p.add_argument(
        "-i",
        "--inputfile",
        required=True,
        type=Path,
        help="Path to text file containing one log output per line.",
    )
    p.add_argument(
        "-o",
        "--outputfolder",
        required=True,
        type=Path,
        help="Path to folder to which all plots will be written.",  # noqa: E501
    )
    p.add_argument(
        "--basename",
        required=False,
        default="loss",
        help="Basename of the created files. Files will be named '<basename>_[metric]'. (default: 'loss')",  # noqa: E501
    )

    return p


COL_EPOCH = "epoch"
COL_ITERS = "iters"
COL_TIME = "time"
COL_DATA_TIME = "data_time"
COL_D_A = "discriminator_A"
COL_G_A = "generator_A"
COL_CYCLE_A = "cycle_A"
COL_IDT_A = "identity_A"
COL_D_B = "discriminator_B"
COL_G_B = "generator_B"
COL_CYCLE_B = "cycle_B"
COL_IDT_B = "identity_B"

INTEGRAL_COLS = [COL_EPOCH, COL_ITERS]
FLOAT_COLS = [
    COL_TIME,
    COL_DATA_TIME,
    COL_D_A,
    COL_G_A,
    COL_CYCLE_A,
    COL_IDT_A,
    COL_D_B,
    COL_G_B,
    COL_CYCLE_B,
    COL_IDT_B,
]

METRIC_COLS = [
    COL_D_A,
    COL_G_A,
    COL_CYCLE_A,
    COL_IDT_A,
    COL_D_B,
    COL_G_B,
    COL_CYCLE_B,
    COL_IDT_B,
]


def parse_loss(path_to_file: Path) -> pd.DataFrame:
    """Get metrics from loss log file and return them as data frame.

    Parameters
    ----------
    path_to_file : Path
        path to loss log file

    Returns
    -------
    df : DataFrame
        rows per logged row; columns per metric
    """

    parser_regex = r".*" \
        r"epoch: (?P<epoch>\d+), " \
        r"iters: (?P<iters>\d+), " \
        r"time: (?P<time>[0-9.]+), " \
        r"data: (?P<data_time>[0-9.]+)\) " \
        r"D_A: (?P<discriminator_A>[0-9.]+) " \
        r"G_A: (?P<generator_A>[0-9.]+) " \
        r"cycle_A: (?P<cycle_A>[0-9.]+) " \
        r"idt_A: (?P<identity_A>[0-9.]+) " \
        r"D_B: (?P<discriminator_B>[0-9.]+) " \
        r"G_B: (?P<generator_B>[0-9.]+) " \
        r"cycle_B: (?P<cycle_B>[0-9.]+) " \
        r"idt_B: (?P<identity_B>[0-9.]+)" \

    list_of_records = list()

    with open(path_to_file, "r") as f:

        for line in f.readlines():
            m = re.match(parser_regex, line)
            if m:
                list_of_records.append(m.groupdict())

    # make data frame
    df = pd.DataFrame.from_records(list_of_records)

    # fix column data types
    for c in INTEGRAL_COLS:
        df[c] = df[c].astype(int)

    for c in FLOAT_COLS:
        df[c] = df[c].astype(float)

    return df


def make_plots(df: pd.DataFrame, path_to_output_folder: Path, basename: str) -> None:
    """Make plots from data frame and dump them to folder.

    Parameters
    ----------
    df : DataFrame
        input data
    """

    # make iterations column cumulative
    df[COL_ITERS] = df[COL_ITERS].cumsum()

    # make long format
    df = df.melt(
        id_vars=[COL_EPOCH, COL_ITERS, COL_TIME, COL_DATA_TIME],
        value_vars=METRIC_COLS,
    )

    # extract domain as column
    df[["variable", "domain"]] = df["variable"].str.split("_", expand=True)

    for time_var in ["epoch", "iters"]:

        # plot with one axes object per metric
        g: sns.FacetGrid = sns.relplot(
            kind="line", data=df, x=time_var, y="value", row="domain", col="variable",
        )
        g.savefig(Path(path_to_output_folder, basename + f"_individual_{time_var}.png"))

        # plot with one axes per domain
        g: sns.FacetGrid = sns.relplot(
            kind="line", data=df, x=time_var, y="value", col="domain", hue="variable",
        )
        g.savefig(Path(path_to_output_folder, basename + f"_domains_{time_var}.png"))


if __name__ == "__main__":

    # get command line arguments
    p = build_argparser()
    args = vars(p.parse_args())

    # 1. parse data from loss log file
    df = parse_loss(args["inputfile"])

    # 2. make plots from metric data
    make_plots(df, args["outputfolder"], basename=args["basename"])
