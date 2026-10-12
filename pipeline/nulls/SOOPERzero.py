from pipeline.nulls import NullArchive
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import matplotlib
import sys
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
# matplotlib.rcParams["font.family"] = "serif"
# matplotlib.rcParams["font.serif"] = "cmr10"
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
stcolor = "#0e1117"

st.set_page_config(page_title="SOOPERzero", layout="wide")

args = sys.argv
if len(args) <= 1:
    st.error(
        "No sacc files provided. "
        "Please run with `streamlit run SOOPERzero.py /path/to/sacc.fits"
    )
    valid_input = False
if len(args) > 2:
    st.error(
        "Multiple inputs provided. "
        "Please run with `streamlit run SOOPERzero.py /path/to/sacc.fits`"
    )
    valid_input = False

if os.path.exists(args[1]):
    sacc_file = args[1]
    valid_input = True
else:
    st.error(f"File does not exist: `{args[1]}`")
    sacc_file = None
    valid_input = False

if "data" not in st.session_state:
    st.session_state.data = None

if valid_input:

    # Load the data
    st.sidebar.markdown(
        f"Selected files: {sacc_file}"
    )

    # Load data
    def load_data():
        st.session_state.data = NullArchive.from_file(sacc_file)

    def clear_ylims():
        st.session_state["ymin"] = None
        st.session_state["ymax"] = None

    if st.session_state.data is None:
        load_data()

    null_archive = st.session_state.data

    # Page layout
    st.title("SOOPERzero: Null SACC viewer")
    st.html(
        """
        <style>
            .stMainBlockContainer {
                max-width:150rem;
            }
        </style>
        """
    )
    mode = st.pills(
        "Mode",
        options=["Residuals", "Summary statistics"],
        default="Residuals",
        width=300,
        label_visibility="hidden"
    )

    # List possible filters to select from
    dtypes = null_archive.dtype
    unique_dtypes = list(set(dtypes))
    field_pairs = [
        dtype.split("_")[1].upper().replace("0", "T")
        for dtype in unique_dtypes
    ]
    null_groups = null_archive.group
    unique_groups = sorted(list(set(null_groups)))

    map_diffs = null_archive.map_diffs

    mapAs = [x[0] for x in map_diffs]
    mapBs = [x[1] for x in map_diffs]
    #####

    if mode == "Residuals":
        # Activate sidebar widgets
        field_pair = st.sidebar.pills(
            "Field pair",
            field_pairs,
            default="EE",
        )
        mapA = st.sidebar.selectbox(
            "Map set 1",
            mapAs,
            index=0
        )
        submapBs = [
            x for x in mapBs if (mapA, x) in map_diffs
        ]
        submapBs = list(set(submapBs))
        mapB = st.sidebar.selectbox(
            "Map set 2",
            submapBs,
            index=0
        )
        range = st.sidebar.slider(
            "Multipole range",
            0,
            600,
            (100, 500),
            step=10
        )

        fig = plt.figure(figsize=(6, 4), facecolor=stcolor)
        import time
        t0 = time.time()
        x, y, ycov = null_archive.get_ps_residual(
            mapA,
            mapB,
            field_pair=field_pair
        )
        mask_ht = (x >= range[0])
        mask_lt = (x <= range[1])
        mask = mask_ht & mask_lt
        compmask_ht = (x < range[0])
        compmask_lt = (x > range[1])

        idx_ht = np.flatnonzero(mask_ht)
        idx_lt = np.flatnonzero(mask_lt)
        compmask_ht[idx_ht[0]] = True
        compmask_lt[idx_lt[-1]] = True

        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(r"$\Delta D_\ell^{%s}$" % field_pair, fontsize=14)
        plt.title(f"$\\rm {mapA} - {mapB}$".replace("_", "\_"), fontsize=14) # noqa
        plt.axhline(0, color="w", lw=1, zorder=1)
        plt.axvline(range[0], color="w", lw=1, ls="--", zorder=1)
        plt.axvline(range[1], color="w", lw=1, ls="--", zorder=1)
        t0 = time.time()
        plt.errorbar(
            x[mask], y[mask], yerr=np.sqrt(np.diag(ycov))[mask],
            marker="o",
            markersize=4.7,
            markerfacecolor="white",
            ls="None",
            zorder=5,
            color="darkorange",
            capsize=1.7
        )
        ylims = plt.gca().get_ylim()
        for compmask in [compmask_ht, compmask_lt]:
            plt.errorbar(
                x[compmask], y[compmask],
                yerr=np.sqrt(np.diag(ycov))[compmask],
                marker="o",
                markersize=4.7,
                markerfacecolor="white",
                ls="None",
                zorder=5,
                color="darkorange",
                capsize=1.7,
                alpha=0.4,
            )

        chi2 = y[mask] @ np.linalg.inv(ycov[np.ix_(mask, mask)]) @ y[mask]

        chi2_sims = []
        sims = []
        for iii in null_archive.sim_ids:
            t0 = time.time()
            x, y, _ = null_archive.get_ps_residual(
                mapA,
                mapB,
                field_pair=field_pair,
                sim_id=iii
            )
            chi2_sims.append(
                y[mask] @ np.linalg.inv(ycov[np.ix_(mask, mask)]) @ y[mask]
            )

            sims.append(y)

        one_sigma_band = np.percentile(sims, [16, 84], axis=0)
        two_sigma_band = np.percentile(sims, [2.5, 97.5], axis=0)

        plt.fill_between(
            x[mask],
            one_sigma_band[0][mask],
            one_sigma_band[1][mask],
            color="gray",
            alpha=0.5,
            zorder=0
        )
        plt.fill_between(
            x[mask],
            two_sigma_band[0][mask],
            two_sigma_band[1][mask],
            color="gray",
            alpha=0.2,
            zorder=0
        )
        for compmask in [compmask_ht, compmask_lt]:
            plt.fill_between(
                x[compmask],
                one_sigma_band[0][compmask],
                one_sigma_band[1][compmask],
                color="gray",
                alpha=0.5*0.5,
                zorder=0
            )
            plt.fill_between(
                x[compmask],
                two_sigma_band[0][compmask],
                two_sigma_band[1][compmask],
                color="gray",
                alpha=0.2*0.5,
                zorder=0
            )

        plt.xlim(0., 600.)
        plt.ylim(ylims)

        pte = np.sum(np.array(chi2_sims) > chi2) / len(chi2_sims)

        plt.text(
            0.05, 0.95,
            f"$\chi^2 = {chi2:.2f}/{np.sum(mask)}$\n${{\\rm PTE}} = {pte:.3f}$", # noqa
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            color="white",
            bbox=dict(
                facecolor=stcolor,
                alpha=0.6,
                edgecolor="none",
                pad=0.5,
                boxstyle="round"
            ),
        )

        plt.gca().set_facecolor(stcolor)
        st.pyplot(fig)
        plt.close(fig)

    elif mode == "Summary statistics":
        # Activate sidebar widgets
        field_pair = st.sidebar.pills(
            "Field pair",
            field_pairs,
            default=["EE", "EB", "BB"],
            selection_mode="multi"
        )
        sel_groups = st.sidebar.multiselect(
            "Null groups",
            unique_groups,
            default=unique_groups,
        )

        range = st.sidebar.slider(
            "Multipole range",
            0,
            600,
            (100, 500),
            step=10
        )

        figs = null_archive.summary_stats(
            field_pairs=field_pair,
            null_groups=sel_groups,
            ellmin=range[0],
            ellmax=range[1],
            return_fig=True
        )

        nrows = 3
        ncols = len(figs) // nrows + (len(figs) % nrows > 0)
        cols = st.columns(ncols)

        for i, fig in enumerate(figs):
            fig.patch.set_facecolor(stcolor)
            ax = fig.axes[0]
            ax.set_facecolor(stcolor)
            try:
                cols[i % ncols].pyplot(fig)
            finally:
                plt.close(fig)
