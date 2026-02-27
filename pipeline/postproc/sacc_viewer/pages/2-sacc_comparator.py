import matplotlib.pyplot as plt
from tkinter import filedialog
import streamlit as st
import tkinter as tk
import numpy as np
import matplotlib
import sacc

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
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"

st.set_page_config(page_title="SOOPERpower", layout="wide")


if "clicked1" not in st.session_state:
    st.session_state.clicked1 = False
if "clicked2" not in st.session_state:
    st.session_state.clicked2 = False


def open_file_dialog1():
    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    root.destroy()
    st.session_state.file_path1 = file_path
    st.session_state.clicked1 = True


def open_file_dialog2():
    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    root.destroy()
    st.session_state.file_path2 = file_path
    st.session_state.clicked2 = True


st.sidebar.text("Select SACC n°1")
st.sidebar.button("Browse", on_click=open_file_dialog1, key="browse1")
st.sidebar.markdown(
    "Selected file: `%s`" % st.session_state.get("file_path1", "None")
)
st.sidebar.text("Select SACC n°2")
st.sidebar.button("Browse", on_click=open_file_dialog2, key="browse2")
st.sidebar.markdown(
    "Selected file: `%s`" % st.session_state.get("file_path2", "None")
)
if st.session_state.clicked1 and st.session_state.clicked2:

    # Load data
    # @st.cache_data
    def load_data():
        sacc1 = sacc.Sacc.load_fits(st.session_state.file_path1)
        sacc2 = sacc.Sacc.load_fits(st.session_state.file_path2)
        st.session_state.loaded_path1 = st.session_state.file_path1
        st.session_state.loaded_path2 = st.session_state.file_path2
        st.session_state.data = (sacc1, sacc2)

    def clear_ylims():
        st.session_state["ymin"] = None
        st.session_state["ymax"] = None

    if "data" not in st.session_state:
        load_data()
    if (
        "loaded_path1" in st.session_state
        and st.session_state.loaded_path1 != st.session_state.file_path1
    ):
        load_data()
    if (
        "loaded_path2" in st.session_state
        and st.session_state.loaded_path2 != st.session_state.file_path2
    ):
        load_data()
    s1, s2 = st.session_state.data
    tracer_pairs1 = s1.get_tracer_combinations()
    tracer_pairs2 = s2.get_tracer_combinations()
    st.title("SACC comparator")
    st.html(
        """
        <style>
            .stMainBlockContainer {
                max-width:150rem;
            }
        </style>
        """
    )

    pair_options1 = [f"{x[0]} x {x[1]}" for x in tracer_pairs1]
    pair_options2 = [f"{x[0]} x {x[1]}" for x in tracer_pairs2]

    pairs_to_display1 = st.sidebar.selectbox(
        "Which map set pairs", pair_options1, index=0, key="pair1"
    )
    pairs_to_display2 = st.sidebar.selectbox(
        "Which map set pairs", pair_options2, index=0, key="pair2"
    )

    field_pair = st.sidebar.pills(
        "Field pair",
        ["TT", "TE", "TB", "ET", "EE", "EB", "BT", "BE", "BB"],
        default="EE",
        on_change=clear_ylims,
    )
    fp = field_pair.replace("T", "0").lower()

    # X-axis
    logscale_xaxis = st.sidebar.checkbox("Log scale x-axis", value=False)
    # Y-axis
    logscale_yaxis = st.sidebar.checkbox("Log scale y-axis", value=False)

    range_x = st.sidebar.slider("X-axis range", 0, 1000, (40, 600), step=10)

    stcolor = "#0e1117"
    fig = plt.figure(figsize=(9, 7), facecolor=stcolor)
    if fp in ["eb", "be", "0b", "b0"]:
        plt.axhline(0., color="white", ls="--")

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{%s}$" % field_pair)

    lb1, cl1, cov1 = s1.get_ell_cl(
        f"cl_{fp}",
        *pairs_to_display1.split(" x "),
        return_cov=True
    )
    lb2, cl2, cov2 = s2.get_ell_cl(
        f"cl_{fp}",
        *pairs_to_display2.split(" x "),
        return_cov=True
    )
    err1 = np.sqrt(cov1.diagonal())
    err2 = np.sqrt(cov2.diagonal())

    mask1 = (lb1 >= range_x[0]) & (lb1 <= range_x[1])
    mask2 = (lb2 >= range_x[0]) & (lb2 <= range_x[1])
    plt.errorbar(
        lb1[mask1]-1,
        cl1[mask1],
        yerr=err1[mask1],
        marker="o",
        label=f"SACC1 {pairs_to_display1}",
        markerfacecolor="white",
        markeredgewidth=1.0,
        markersize=4,
        elinewidth=1.0,
        capsize=2.0,
        ls="None",
    )
    plt.errorbar(
        lb2[mask2]+1,
        cl2[mask2],
        yerr=err2[mask2],
        marker="o",
        label=f"SACC2 {pairs_to_display2}",
        markerfacecolor="white",
        markeredgewidth=1.0,
        markersize=4,
        elinewidth=1.0,
        capsize=2.0,
        ls="None",
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
