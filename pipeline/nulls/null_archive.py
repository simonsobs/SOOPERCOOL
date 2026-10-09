import sacc
import numpy as np
import matplotlib.pyplot as plt


class NullArchive:
    """
    This is a class to handle bookkeping of null tests via sacc files.
    Instantiate an object with a list of sacc files, formatted as
    ```
    sacc_files = [
        sacc.Sacc.load_fits("null_sacc_file_1.fits"),
        sacc.Sacc.load_fits("null_sacc_file_2.fits"),
        ...
    ]
    ```
    They will be concatenated into a single sacc object, containing
    all residuals for data and simulations.

    The class provides methods to interact with power spectrum
    residuals and compute chi, chi2 and PTE statistics.
    """
    def __init__(self, sacc_files):

        saccs = self._read_from_fname_or_sacc(sacc_files)
        self.sacc = sacc.concatenate_data_sets(*saccs)

        self.parse_sacc()

    def _read_from_fname_or_sacc(self, sacc_files):
        """
        """
        if isinstance(sacc_files, (str, sacc.Sacc)):
            inputs = [sacc_files]
        elif isinstance(sacc_files, list):
            inputs = sacc_files
        else:
            raise ValueError(
                "Expected a path, SACC object or a list of either."
            )
        if not inputs:
            raise ValueError("At least one SACC input required.")

        if all(isinstance(item, sacc.Sacc) for item in inputs):
            self.sacc_files = None
            return inputs
        elif all(isinstance(item, str) for item in inputs):
            self.sacc_files = inputs
            return [sacc.Sacc.load_fits(f) for f in inputs]

        raise TypeError(
            "SACC inputs must all be all paths or all SACC objects."
        )

    @classmethod
    def from_file(cls, fnames):
        """
        Instantiate a NullArchive object from saved sacc file.
        """
        return cls(fnames)

    def to_file(self, fname):
        """
        Save the NullArchive object to a sacc file.
        """
        self.sacc.save_fits(fname)

    def parse_sacc(self):
        """
        """
        self.map_diffs = self.sacc.get_tracer_combinations()

        # Read the SACC content once
        dps = self.sacc.get_data_points()
        sim_tag = np.array(
            [-1 if dp.tags.get("sim") is None else int(dp.tags["sim"])
             for dp in dps]
        )
        mean = self.sacc.mean

        data_idx = np.where(sim_tag == -1)[0]
        self.sim_ids = sorted(set(sim_tag[sim_tag >= 0]))
        self.n_sims = len(self.sim_ids)
        self._sim_row = {sim_id: i for i, sim_id in enumerate(self.sim_ids)}

        # Layout of a single realization
        self.group = np.array([dps[i].tags["group"] for i in data_idx])
        self.freq = np.array([dps[i].tags["freq"] for i in data_idx])
        self.dtype = np.array([dps[i].data_type for i in data_idx])
        self.ell = np.array(
            [dps[i].tags["ell"] for i in data_idx],
            dtype=np.float32
        )
        self.tracer1 = np.array([dps[i].tracers[0] for i in data_idx])
        self.tracer2 = np.array([dps[i].tracers[1] for i in data_idx])

        self.data = mean[data_idx]
        # Concatenate simulations into matrices
        self.X = np.stack(
            [mean[np.where(sim_tag == sim_id)[0]] for sim_id in self.sim_ids]
        )
        if self.X.shape[1] != self.data.size:
            raise ValueError("Sims and data do not have the same layout.")

        self.cov = np.cov(self.X.T)
        self.var = self.cov.diagonal()

    def inspect(self):
        """
        Inspect the null archive to indicate which nulls, map sets
        are present.
        """
        unique_groups = list(set(self.group))
        diffs = [
            (ms1, ms2)
            for ms1, ms2 in zip(
                self.tracer1,
                self.tracer2
            )
        ]
        unique_diffs = list(set(diffs))
        field_pairs = [
            dtype.split("_")[-1].upper().replace("0", "T")
            for dtype in self.dtype
        ]
        unique_field_pairs = list(set(field_pairs))

        # Run some inspection of the null archive to
        # indicate which nulls, map sets are present.
        print("Inspecting null archive")
        print("-----------------------")
        print("  Provided files:")
        for f in self.sacc_files:
            print(f"    {f}")
        print("  Available null groups:")
        for group in unique_groups:
            print(f"    {group}")
        print("  Available map set differences:")
        for ms1, ms2 in unique_diffs:
            print(f"    {ms1} -- {ms2}")
        print("  Available field pairs:")
        for field_pair in unique_field_pairs:
            print(f"    {field_pair}")
        print(f"  Number of simulations: {self.n_sims}")

    def select(self, null_groups=None,
               field_pairs=None,
               freqs=None,
               ell=None,
               ellmin=None,
               ellmax=None):
        """
        Return boolean mask for the data points
        matching several selection criteria.

        Parameters
        ----------
        null_groups : list of str, optional
            List of null groups to select. If None, all groups are selected.
        field_pairs : list of str, optional
            List of field pairs to select. If None, all field pairs
            are selected.
        freqs : list of int, optional
            List of frequencies to select. If None, all frequencies
            are selected.
        ell : int, optional
            Ell bin to select. If None, all ell bins are selected.
        ellmin : int, optional
            Minimum ell bin to select. If None, no minimum is applied.
        ellmax : int, optional
            Maximum ell bin to select. If None, no maximum is applied.

        Returns
        -------
        m: np.ndarray, dtype=bool
            Boolean mask matching selection criteria.
        """
        m = np.ones(self.group.size, dtype=bool)
        if null_groups is not None:
            m &= np.isin(self.group, null_groups)
        if freqs is not None:
            m &= np.isin(self.freq, freqs)
        if field_pairs is not None:
            m &= np.isin(
                self.dtype,
                [f"cl_{fp.lower().replace('t', '0')}" for fp in field_pairs]
            )
        if ell is not None:
            m &= self.ell == ell

        if ellmin is not None:
            m &= self.ell >= ellmin
        if ellmax is not None:
            m &= self.ell <= ellmax
        return m

    def _values(self, sim_id=None):
        """
        Return the data/sim residual data vector

        Parameters
        ----------
        sim_id : int, optional
            Simulation ID to select. If None, data residual is returned.

        Returns
        -------
        values : np.ndarray
            Data vector of residuals for the selected simulation or data.
        """
        return self.data if sim_id is None else self.X[self._sim_row[sim_id]]

    def get_ps_residual(self, map_A, map_B, field_pair, sim_id=None):
        """
        Get the residual power spectrum of the map difference
        map_A - map_B for a given simulation id (None for the data)

        Parameters
        ----------
        map_A : str
            Name of the first map in the difference.
        map_B : str
            Name of the second map in the difference.
        field_pair : str
            Field pair to select. Must be one of
            "TT", "TE", "TB", "EE", "EB", "BB".
        sim_id : int, optional
            Simulation ID to select. If None, data residual is returned.

        Returns
        -------
        ell : np.ndarray
            Array of ell bins for the selected residual power spectrum.
        values : np.ndarray
            Array of residual power spectrum values for the selected
            simulation or data.
        cov : np.ndarray
            Covariance matrix of the residual power spectrum for the
            selected simulation or data.
        """
        dtype = f"cl_{field_pair.lower().replace('t', '0')}"
        m = (self.dtype == dtype)
        m &= (self.tracer1 == map_A) & (self.tracer2 == map_B)

        if not m.any():
            m = (self.dtype == dtype)
            m &= (self.tracer1 == map_B) & (self.tracer2 == map_A)
            if not m.any():
                raise ValueError(
                    f"No data found for {map_A} - {map_B} with"
                    f" field pair {field_pair}."
                    f" Existing map differences are {self.map_diffs}."
                )
        return self.ell[m], self._values(sim_id)[m], self.cov[np.ix_(m, m)]

    def get_chi_chi2(self, null_groups=None,
                     field_pairs=None,
                     freqs=None,
                     ell=None,
                     ellmin=None,
                     ellmax=None,
                     on_sims=False,
                     diag_cov=True):
        """
        Return (chi, chi2) for the data, or, if `on_sims` is True, for each
        simulation: chi has shape (n_sims, n_sel) and chi2 shape (n_sims,).
        """
        m = self.select(null_groups, field_pairs, freqs, ell, ellmin, ellmax)

        if diag_cov:
            cov = np.diag(self.var[m])
        else:
            cov = self.cov[np.ix_(m, m)]
        res = self.X[:, m] if on_sims else self.data[m][None, :]

        chi = res / np.sqrt(self.var[m])
        chi2 = np.einsum("si,is->s", res, np.linalg.solve(cov, res.T))
        if on_sims:
            return chi, chi2
        return chi[0], chi2[0]

    def _summary_stats_per_field_pair(self, null_groups=None,
                                      field_pairs=None,
                                      ellmin=None,
                                      ellmax=None,
                                      diag_cov=True):
        """
        """
        stats = []
        if field_pairs is None:
            field_pairs = list(set(self.dtype))
        if null_groups is None:
            null_groups = list(set(self.group))
        stats_sims = np.zeros((len(field_pairs), self.n_sims))

        for fp in field_pairs:
            chi, chi2 = self.get_chi_chi2(
                null_groups=null_groups,
                field_pairs=[fp],
                ellmin=ellmin,
                ellmax=ellmax,
                diag_cov=diag_cov
            )
            stats.append(chi2)

            chi_sims, chi2_sims = self.get_chi_chi2(
                null_groups=null_groups,
                field_pairs=[fp],
                ellmin=ellmin,
                ellmax=ellmax,
                diag_cov=diag_cov,
                on_sims=True

            )
            stats_sims[field_pairs.index(fp)] = chi2_sims

        max_chi2 = np.max(stats)
        max_chi2_sims = np.max(stats_sims, axis=0)

        pte = np.sum(max_chi2_sims > max_chi2) / self.n_sims
        pte_sims = [
            np.sum(max_chi2_sims > max_chi2_sims[i]) / self.n_sims
            for i in range(self.n_sims)
        ]

        return max_chi2, max_chi2_sims, pte, pte_sims

    def _summary_stats_per_test(self, null_groups=None,
                                field_pairs=None,
                                ellmin=None,
                                ellmax=None,
                                diag_cov=True):
        """
        """
        stats = []
        if null_groups is None:
            null_groups = list(set(self.group))
        if field_pairs is None:
            field_pairs = list(set(self.dtype))
        stats_sims = np.zeros((len(null_groups), self.n_sims))

        for ng in null_groups:
            chi, chi2 = self.get_chi_chi2(
                null_groups=[ng],
                field_pairs=field_pairs,
                ellmin=ellmin,
                ellmax=ellmax,
                diag_cov=diag_cov
            )
            stats.append(chi2)

            chi_sims, chi2_sims = self.get_chi_chi2(
                null_groups=[ng],
                field_pairs=field_pairs,
                ellmin=ellmin,
                ellmax=ellmax,
                diag_cov=diag_cov,
                on_sims=True
            )
            stats_sims[null_groups.index(ng)] = chi2_sims

        max_chi2 = np.max(stats)
        max_chi2_sims = np.max(stats_sims, axis=0)

        pte = np.sum(max_chi2_sims > max_chi2) / self.n_sims
        pte_sims = [
            np.sum(max_chi2_sims > max_chi2_sims[i]) / self.n_sims
            for i in range(self.n_sims)
        ]

        return max_chi2, max_chi2_sims, pte, pte_sims

    def _summary_stats_per_ell(self, null_groups=None,
                               field_pairs=None,
                               ellmin=None,
                               ellmax=None,
                               diag_cov=True):
        """
        """
        if field_pairs is None:
            field_pairs = list(set(self.dtype))
        if null_groups is None:
            null_groups = list(set(self.group))

        stats = {
            "chisq": [],
            "avg_chi": []
        }

        used_ells = np.array(list(set(self.ell)))
        m = np.ones(used_ells.size, dtype=bool)
        if ellmin is not None:
            m &= used_ells >= ellmin
        if ellmax is not None:
            m &= used_ells <= ellmax
        used_ells = used_ells[m]

        stats_sims = {
            "chisq": np.zeros((used_ells.size, self.n_sims)),
            "avg_chi": np.zeros((used_ells.size, self.n_sims))
        }
        for ell in used_ells:
            chi, chi2 = self.get_chi_chi2(
                null_groups=null_groups,
                field_pairs=field_pairs,
                ell=ell,
                ellmin=ellmin,
                ellmax=ellmax,
                diag_cov=diag_cov
            )
            stats["chisq"].append(chi2)
            stats["avg_chi"].append(np.mean(chi))

            chi_sims, chi2_sims = self.get_chi_chi2(
                null_groups=null_groups,
                field_pairs=field_pairs,
                ell=ell,
                ellmin=ellmin,
                ellmax=ellmax,
                diag_cov=diag_cov,
                on_sims=True
            )
            stats_sims["chisq"][
                np.where(used_ells == ell)[0][0]
            ] = chi2_sims
            stats_sims["avg_chi"][
                np.where(used_ells == ell)[0][0]
            ] = np.mean(chi_sims, axis=1)

        max_chi2 = np.max(stats["chisq"])
        max_chi2_sims = np.max(stats_sims["chisq"], axis=0)
        pte_max = np.sum(max_chi2_sims > max_chi2) / self.n_sims
        pte_max_sims = [
            np.sum(max_chi2_sims > max_chi2_sims[i]) / self.n_sims
            for i in range(self.n_sims)
        ]

        avg_chi = np.mean(stats["avg_chi"])
        avg_chi_sims = np.mean(stats_sims["avg_chi"], axis=0)
        pte_avg = np.sum(avg_chi_sims > avg_chi) / self.n_sims
        pte_avg_sims = [
            np.sum(avg_chi_sims > avg_chi_sims[i]) / self.n_sims
            for i in range(self.n_sims)
        ]

        stats = {
            "max_chisq": max_chi2,
            "avg_chi": avg_chi,
        }
        pte = {
            "max_chisq": pte_max,
            "avg_chi": pte_avg
        }
        stats_sims = {
            "max_chisq": max_chi2_sims,
            "avg_chi": avg_chi_sims
        }
        pte_sims = {
            "max_chisq": pte_max_sims,
            "avg_chi": pte_avg_sims
        }

        return stats, pte, stats_sims, pte_sims

    def _plot_histogram(self, chi,
                        chi_sims,
                        pte,
                        title=None,
                        xlabel=None,
                        fname=None,
                        display=False,
                        is_pte=False):
        """
        """
        if is_pte:
            bins = np.linspace(0, 1, 2*int(np.sqrt(self.n_sims)))
        else:
            bins = 2*int(np.sqrt(self.n_sims))
        plt.figure(figsize=(6, 4))
        plt.hist(
            chi_sims,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Simulations"
        )
        plt.axvline(chi, color="r", linestyle="--", label="Data")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend()
        plt.text(0.95, 0.95, f"PTE = {pte:.3f}", transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right')
        if fname is not None:
            plt.savefig(fname)
        if display:
            plt.show()
        else:
            plt.close()

    def summary_stats(self, null_groups=None,
                      field_pairs=None,
                      ellmin=None,
                      ellmax=None,
                      diag_cov=True,
                      fname=None,
                      display=False):
        """
        """

        max_chi2_fp, max_chi2_sims_fp, pte_fp, pte_sims_fp = self._summary_stats_per_field_pair( # noqa
            null_groups=null_groups,
            field_pairs=field_pairs,
            ellmin=ellmin,
            ellmax=ellmax,
            diag_cov=diag_cov
        )
        max_chi2_group, max_chi2_sims_group, pte_group, pte_sims_group = self._summary_stats_per_test( # noqa
            null_groups=null_groups,
            field_pairs=field_pairs,
            ellmin=ellmin,
            ellmax=ellmax,
            diag_cov=diag_cov
        )
        stats_ell, pte_ell, stats_sims_ell, pte_sims_ell = self._summary_stats_per_ell( # noqa
            null_groups=null_groups,
            field_pairs=field_pairs,
            ellmin=ellmin,
            ellmax=ellmax,
            diag_cov=diag_cov
        )
        max_chi2_ell = stats_ell["max_chisq"]
        max_chi2_sims_ell = stats_sims_ell["max_chisq"]
        pte_max_ell = pte_ell["max_chisq"]
        pte_max_sims_ell = pte_sims_ell["max_chisq"]

        avg_chi_ell = stats_ell["avg_chi"]
        avg_chi_sims_ell = stats_sims_ell["avg_chi"]
        pte_avg_ell = pte_ell["avg_chi"]
        pte_avg_sims_ell = pte_sims_ell["avg_chi"]

        # Total chi2
        tot_chi, tot_chi2 = self.get_chi_chi2(
            null_groups=null_groups,
            field_pairs=field_pairs,
            ellmin=ellmin,
            ellmax=ellmax,
            diag_cov=diag_cov
        )
        tot_chi_sims, tot_chi2_sims = self.get_chi_chi2(
            null_groups=null_groups,
            field_pairs=field_pairs,
            ellmin=ellmin,
            ellmax=ellmax,
            diag_cov=diag_cov,
            on_sims=True
        )
        tot_pte = np.sum(tot_chi2_sims > tot_chi2) / self.n_sims
        tot_pte_sims = [
            np.sum(tot_chi2_sims > tot_chi2_sims[i]) / self.n_sims
            for i in range(self.n_sims)
        ]

        stacked_pte = np.array([
            pte_fp, pte_group, pte_max_ell, pte_avg_ell, tot_pte
        ])

        stacked_pte_sims = np.stack([
            pte_sims_fp,
            pte_sims_group,
            pte_max_sims_ell,
            pte_avg_sims_ell,
            tot_pte_sims
        ])

        max_pte = np.max(stacked_pte)
        min_pte = np.min(stacked_pte)
        max_pte_sims = np.max(stacked_pte_sims, axis=0)
        min_pte_sims = np.min(stacked_pte_sims, axis=0)

        meta_pte_max = np.sum(max_pte_sims > max_pte) / self.n_sims
        meta_pte_min = np.sum(min_pte_sims < min_pte) / self.n_sims

        # PLOTS
        self._plot_histogram(
            chi=tot_chi2,
            chi_sims=tot_chi2_sims,
            pte=tot_pte,
            title=r"Total $\chi^2$",
            xlabel=r"$\chi^2$",
            display=display,
            fname=f"{fname}_total_chi2.pdf" if fname is not None else None
        )
        self._plot_histogram(
            chi=max_chi2_ell,
            chi_sims=max_chi2_sims_ell,
            pte=pte_max_ell,
            title=r"Max $\chi^2$ across $\ell$ bin",
            xlabel=r"$\chi^2$",
            display=display,
            fname=f"{fname}_max_chi2_ell.pdf" if fname is not None else None
        )
        self._plot_histogram(
            chi=avg_chi_ell,
            chi_sims=avg_chi_sims_ell,
            pte=pte_avg_ell,
            title=r"Average $\chi$ across $\ell$ bin",
            xlabel=r"$\chi$",
            display=display,
            fname=f"{fname}_avg_chi_ell.pdf" if fname is not None else None
        )
        self._plot_histogram(
            chi=max_chi2_fp,
            chi_sims=max_chi2_sims_fp,
            pte=pte_fp,
            title=r"Max $\chi^2$ across field pairs",
            xlabel=r"$\chi^2$",
            display=display,
            fname=None if fname is None else f"{fname}_max_chi2_field_pairs.pdf" # noqa
        )
        self._plot_histogram(
            chi=max_chi2_group,
            chi_sims=max_chi2_sims_group,
            pte=pte_group,
            title=r"Max $\chi^2$ across null tests",
            xlabel=r"$\chi^2$",
            display=display,
            fname=None if fname is None else f"{fname}_max_chi2_null_groups.pdf" # noqa
        )
        self._plot_histogram(
            chi=max_pte,
            chi_sims=max_pte_sims,
            pte=meta_pte_max,
            title="Max PTE across all tests",
            xlabel="PTE",
            display=display,
            is_pte=True,
            fname=f"{fname}_max_pte.pdf" if fname is not None else None
        )
        self._plot_histogram(
            chi=min_pte,
            chi_sims=min_pte_sims,
            pte=meta_pte_min,
            title="Min PTE across all tests",
            xlabel="PTE",
            display=display,
            is_pte=True,
            fname=f"{fname}_min_pte.pdf" if fname is not None else None
        )
