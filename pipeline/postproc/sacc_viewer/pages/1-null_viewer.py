import matplotlib.pyplot as plt
from tkinter import filedialog
import streamlit as st
import tkinter as tk
import numpy as np
import matplotlib
import sacc
import scipy.stats

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

st.set_page_config(page_title="SAT null viewer", layout="wide")


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
    st.title("SAT null viewer")
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

    which_ms1 = st.sidebar.selectbox(
        "Which map set 1?", options=ms1_unique, index=0
    )

    ms2_unique = ms2_unique[ms2_unique != which_ms1]
    ms2_unique = [x for x in ms2_unique if (which_ms1, x) in tracer_pairs]

    which_ms2 = st.sidebar.selectbox(
        "Which map set 2?", options=ms2_unique, index=0
    )

    which_diff = st.sidebar.selectbox(
        "Which residual?",
        options=["Map residual", "Auto-spectrum residual"],
        index=0,
    )

    field_pair = st.sidebar.pills(
        "Field pair",
        ["TT", "TE", "TB", "ET", "EE", "EB", "BT", "BE", "BB"],
        default="EE",
        on_change=clear_ylims,
    )
    fp = field_pair.replace("T", "0").lower()

    range_x_plot = st.sidebar.slider(
        "X-axis range (for plot)", 0, 1000, (0, 600), step=10
    )
    range_x_chi2 = st.sidebar.slider(
        "X-axis range for chi2", 0, 1000, (30, 400), step=10
    )

    idx_ms1_ms1 = s.indices(f"cl_{fp}", (which_ms1, which_ms1))
    idx_ms1_ms2 = s.indices(f"cl_{fp}", (which_ms1, which_ms2))
    idx_ms2_ms2 = s.indices(f"cl_{fp}", (which_ms2, which_ms2))

    idx = np.concatenate([idx_ms1_ms1, idx_ms1_ms2, idx_ms2_ms2])
    nl = len(idx) // 3

    P = np.zeros((nl, len(idx)))
    if which_diff == "Map residual":
        P[:nl, :nl] = np.eye(nl)
        P[:nl, nl: 2 * nl] = -2 * np.eye(nl)
        P[:nl, 2 * nl:] = np.eye(nl)
    elif which_diff == "Auto-spectrum residual":
        P[:nl, :nl] = np.eye(nl)
        P[:nl, nl: 2 * nl] = 0.0
        P[:nl, 2 * nl:] = -np.eye(nl)

    res = P @ s.mean[idx]
    rescov = P @ s.covariance.covmat[np.ix_(idx, idx)] @ P.T
    reserr = np.sqrt(rescov.diagonal())

    stcolor = "#0e1117"
    fig = plt.figure(figsize=(9, 7), facecolor=stcolor)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\Delta D_\ell^{%s}$" % field_pair)

    plt.axhline(0, color="white", ls="--", lw=1.0)
    plt.axvline(range_x_chi2[0], color="gray", ls="--", lw=1.0)
    plt.axvline(range_x_chi2[1], color="gray", ls="--", lw=1.0)

    lb = s.get_ell_cl(f"cl_{fp}", which_ms1, which_ms2)[0]

    idx_chi2 = (lb >= range_x_chi2[0]) & (lb <= range_x_chi2[1])
    chi2 = (
        res[idx_chi2].T
        @ np.linalg.inv(rescov[np.ix_(idx_chi2, idx_chi2)])
        @ res[idx_chi2]
    )
    dof = len(res[idx_chi2])

    pval = 1 - scipy.stats.chi2.cdf(chi2, dof)
    plt.errorbar(
        lb,
        res,
        yerr=reserr,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.0,
        markersize=4,
        elinewidth=1.0,
        capsize=2.0,
        ls="None",
    )
    plt.text(
        0.98,
        0.98,
        f"$\\chi^2$/dof = {chi2:.1f}/{dof} (PTE={pval:.3f})",
        transform=plt.gca().transAxes,
        fontsize=15,
        verticalalignment="top",
        horizontalalignment="right",
    )
    plt.xlim(*range_x_plot)

    min_y = (
        np.min(
            (res - reserr)[
                idx_chi2 & (lb >= range_x_plot[0]) & (lb <= range_x_plot[1])
            ]
        )
        * 1.1
    )
    max_y = (
        np.max(
            (res + reserr)[
                idx_chi2 & (lb >= range_x_plot[0]) & (lb <= range_x_plot[1])
            ]
        )
        * 1.1
    )
    plt.ylim(min_y, max_y)

    plt.title(f"{which_diff} for {which_ms1} x {which_ms2}", fontsize=15)

    plt.gca().set_facecolor(stcolor)
    st.pyplot(fig, width="content", dpi=400)
