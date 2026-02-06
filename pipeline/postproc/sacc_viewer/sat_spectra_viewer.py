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

st.set_page_config(page_title="SAT ISO viewer", layout="wide")


if "clicked" not in st.session_state:
    st.session_state.clicked = False


def open_file_dialog():
    root = tk.Tk()
    # root.withdraw()
    # root.mainloop()
    file_path = filedialog.askopenfilename()
    root.destroy()
    st.session_state.file_path = file_path
    st.session_state.clicked = True


st.sidebar.text("Select a sacc file")
st.sidebar.button("Browse", on_click=open_file_dialog)
st.sidebar.markdown(
    "Selected file: `%s`" % st.session_state.get("file_path", "None")
)
if st.session_state.clicked:

    # Load data
    # @st.cache_data
    def load_data():
        s = sacc.Sacc.load_fits(st.session_state.file_path)
        st.session_state.loaded_path = st.session_state.file_path
        st.session_state.data = s

    def clear_ylims():
        st.session_state["ymin"] = None
        st.session_state["ymax"] = None

    if "data" not in st.session_state:
        load_data()
    if (
        "loaded_path" in st.session_state
        and st.session_state.loaded_path != st.session_state.file_path
    ):
        load_data()
    s = st.session_state.data
    tracer_pairs = s.get_tracer_combinations()
    st.title("SAT SACC visualizer")
    st.html(
        """
        <style>
            .stMainBlockContainer {
                max-width:150rem;
            }
        </style>
        """
    )

    ms1_unique = np.unique([x[0] for x in tracer_pairs])
    ms2_unique = np.unique([x[1] for x in tracer_pairs])

    ms1ms2 = [f"{x[0]} x {x[1]}" for x in tracer_pairs]

    pairs_to_display = st.sidebar.multiselect(
        "Which map set pairs", ms1ms2, default=ms1ms2[0]
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
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{%s}$" % field_pair)

    for ms1ms2 in pairs_to_display:
        ms1, ms2 = ms1ms2.split(" x ")
        lb, cl, cov = s.get_ell_cl(f"cl_{fp}", ms1, ms2, return_cov=True)
        err = np.sqrt(cov.diagonal())

        plt.errorbar(
            lb,
            cl,
            yerr=err,
            marker="o",
            label=f"{ms1} x {ms2}",
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
        "Y-axis min", value=None, format="%.3f", key="ymin"
    )
    max_y = st.sidebar.number_input(
        "Y-axis max", value=None, format="%.3f", key="ymax"
    )
    range_y = (min_y, max_y)
    plt.ylim(*range_y)

    plt.legend(
        frameon=False, ncol=1, bbox_to_anchor=(1.0, 1.0), loc="upper left"
    )

    plt.gca().set_facecolor(stcolor)
    st.pyplot(fig, width="content", dpi=400)
