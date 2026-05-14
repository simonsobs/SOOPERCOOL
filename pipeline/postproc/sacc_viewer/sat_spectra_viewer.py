import matplotlib.pyplot as plt
from tkinter import filedialog
import streamlit as st
import numpy as np
import matplotlib
import sacc
import camb
import os

matplotlib.rcParams["text.color"] = "white"
matplotlib.rcParams["xtick.color"] = "white"
matplotlib.rcParams["ytick.color"] = "white"
matplotlib.rcParams["axes.facecolor"] = "white"
matplotlib.rcParams["axes.labelcolor"] = "white"
matplotlib.rcParams["axes.edgecolor"] = "white"
matplotlib.rcParams["axes.linewidth"] = 2.0
matplotlib.rcParams["xtick.major.width"] = 2.0
matplotlib.rcParams["ytick.major.width"] = 2.0
matplotlib.rcParams["xtick.minor.width"] = 1.5
matplotlib.rcParams["ytick.minor.width"] = 1.5
matplotlib.rcParams["legend.fontsize"] = 15
matplotlib.rcParams["axes.labelsize"] = 20
matplotlib.rcParams["axes.labelpad"] = 20
matplotlib.rcParams["xtick.labelsize"] = 15
matplotlib.rcParams["ytick.labelsize"] = 15
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Computer Modern Roman"
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"


def get_theory_cls(cosmo_params):
    """ """
    params = camb.set_params(**cosmo_params)
    params.set_for_lmax(2000, lens_potential_accuracy=2, lens_margin=500)
    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(
        params, CMB_unit="muK", raw_cl=False
    )
    lth = np.arange(powers["total"].shape[0])

    cl_th = {
        "TT": powers["total"][:, 0],
        "EE": powers["total"][:, 1],
        "TE": powers["total"][:, 3],
        "ET": powers["total"][:, 3],
        "BB": powers["total"][:, 2],
    }
    for spec in ["EB", "TB", "BE", "BT"]:
        cl_th[spec] = np.zeros_like(lth)

    return lth, cl_th


st.set_page_config(page_title="SOOPERpower", layout="wide")


def open_file_dialog():
    paths = filedialog.askopenfilenames()

    file_info = []
    for path in paths:
        label = os.path.basename(path)
        file_info.append((path, label))

    if file_info:
        st.session_state.file_info = file_info
        st.session_state.clicked = True


def parse_cli_files():
    import sys
    file_info = []

    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue

        if ":" in arg:
            path, label = arg.split(":", 1)
        else:
            path = arg
            label = os.path.basename(path)

        if os.path.exists(path):
            file_info.append((path, label))
        else:
            st.warning(f"File does not exist and will be skipped: `{path}`")

    return file_info


if "clicked" not in st.session_state:
    st.session_state.clicked = False

if "file_info" not in st.session_state:
    st.session_state.file_info = []

if "data" not in st.session_state:
    st.session_state.data = None

if "loaded_info" not in st.session_state:
    st.session_state.loaded_info = []

# Initialize from CLI arguments if files were provided
cli_file_info = parse_cli_files()

if len(cli_file_info) > 0 and len(st.session_state.file_info) == 0:
    st.session_state.file_info = cli_file_info
    st.session_state.clicked = True

# Sidebar: file input
st.sidebar.text("Select sacc file(s)")

if len(cli_file_info) == 0:
    st.sidebar.button("Browse", on_click=open_file_dialog)
else:
    st.sidebar.text("Using command-line files")

st.sidebar.markdown(
    "Selected files:\n" + "\n".join(
        [
            f"- `{label}` ({path})"
            for path, label in st.session_state.get("file_info", [])
        ]
    )
)

if st.session_state.clicked:

    # Load data
    def load_data():
        datasets = []

        for path, label in st.session_state.file_info:
            s = sacc.Sacc.load_fits(path)
            datasets.append((label, s))

        st.session_state.loaded_info = st.session_state.file_info.copy()
        st.session_state.data = datasets

    # Compute theory
    @st.cache_data
    def compute_theory(r=0.0):
        cosmo = {
            "cosmomc_theta": 0.0104073,
            "As": np.exp(3.056) * 1e-10,
            "ombh2": 0.02250,
            "omch2": 0.1193,
            "ns": 0.9709,
            "Alens": 1.0,
            "tau": 0.0603,
            "r": r,
        }
        return get_theory_cls(cosmo)

    def clear_ylims():
        st.session_state["ymin"] = None
        st.session_state["ymax"] = None

    if st.session_state.data is None:
        load_data()

    if (
        "loaded_info" in st.session_state
        and st.session_state.loaded_info != st.session_state.file_info
    ):
        load_data()

    datasets = st.session_state.data
    st.title("SOOPERpower")
    st.html(
        """
        <style>
            .stMainBlockContainer {
                max-width:150rem;
            }
        </style>
        """
    )

    # Per-file tracer selection
    file_pairs = []

    for i, (label, s) in enumerate(datasets):
        tracer_pairs = s.get_tracer_combinations()
        ms1ms2 = [f"{x[0]} x {x[1]}" for x in tracer_pairs]

        pairs_to_display = st.sidebar.multiselect(
            f"{label} pairs",
            ms1ms2,
            default=ms1ms2[0],
            key=f"pairs_{i}",
        )

        file_pairs.append((label, s, pairs_to_display))

    field_pair = st.sidebar.pills(
        "Field pair",
        ["TT", "TE", "TB", "ET", "EE", "EB", "BT", "BE", "BB"],
        default="EE",
        on_change=clear_ylims,
    )
    fp = field_pair.replace("T", "0").lower()

    lth, cl_th = compute_theory(r=0.)

    # X-axis
    logscale_xaxis = st.sidebar.checkbox("Log scale x-axis", value=False)
    # Y-axis
    logscale_yaxis = st.sidebar.checkbox("Log scale y-axis", value=False)

    # Rescale theory
    rescale_th = st.sidebar.number_input(
        "Theory rescaling factor",
        value=1e-12,
        format="%.2E",
        key="th_rescale"
    )

    range_x = st.sidebar.slider("X-axis range", 0, 1000, (40, 600), step=10)

    stcolor = "#0e1117"
    fig = plt.figure(figsize=(9, 7), facecolor=stcolor)
    if fp in ["eb", "be", "0b", "b0"]:
        plt.axhline(0., color="white", ls="--")
    plt.plot(
        lth,
        cl_th[fp.upper().replace("0", "T")] * rescale_th,
        color="white",
    )

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{%s}$" % field_pair)

    n_files = len(file_pairs)

    for file_id, (label, s, pairs_to_display) in enumerate(file_pairs):
        for id12, ms1ms2 in enumerate(pairs_to_display):
            ms1, ms2 = ms1ms2.split(" x ")

            try:
                lb, cl, cov = s.get_ell_cl(
                    f"cl_{fp}", ms1, ms2, return_cov=True
                )
                err = np.sqrt(cov.diagonal())
                has_cov = True
                return_cov = True

            except Exception:
                try:
                    lb, cl = s.get_ell_cl(
                        f"cl_{fp}", ms1, ms2, return_cov=False
                    )
                    err = None
                    has_cov = False
                    return_cov = False

                except Exception:
                    continue

            offset = (
                (
                    id12 - len(pairs_to_display) / 2 + 0.5
                    + file_id - n_files / 2 + 0.5
                )
                * (range_x[1] - range_x[0])
                * 0.002
            )

            mask = (lb >= range_x[0]) & (lb <= range_x[1])

            if return_cov:
                plt.errorbar(
                    lb[mask] + offset,
                    cl[mask],
                    yerr=err[mask],
                    marker="o",
                    label=f"{label} | {ms1} x {ms2}",
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    markersize=4,
                    elinewidth=1.0,
                    capsize=2.0,
                    ls="None",
                )
            else:
                plt.plot(
                    lb[mask],
                    cl[mask],
                    marker="o",
                    label=f"{label} | {ms1} x {ms2}",
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    markersize=4,
                    ls="-",
                )

    if logscale_xaxis:
        plt.xscale("log")
    if logscale_yaxis:
        plt.yscale("log")
    plt.xlim(*range_x)
    # Dynamically define y axis limits
    min_y, max_y = plt.gca().get_ylim()

    min_y = st.sidebar.number_input(
        "Y-axis min", value=None, format="%.2E", key="ymin"
    )
    max_y = st.sidebar.number_input(
        "Y-axis max", value=None, format="%.2E", key="ymax"
    )
    range_y = (min_y, max_y)
    plt.ylim(*range_y)

    plt.legend(
        frameon=False,
        ncol=1,
        bbox_to_anchor=(0.5, -0.13),
        loc="upper center"
    )

    plt.gca().set_facecolor(stcolor)
    st.pyplot(fig, width="content", dpi=400)
