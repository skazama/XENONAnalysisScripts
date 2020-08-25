import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import strax
import straxen
import numba
import datetime
import pandas as pd
import os
import warnings


def plots_area_vs_width(data, log=None, **kwargs):
    """basic wrapper to plot area vs width"""
    x, y = data['area'], data['range_50p_area']
    if log:
        x, y = np.log10(x), np.log10(y)
    plt.hist2d(x, y, norm=LogNorm(), **kwargs)
    plt.ylabel(f'Width [ns]')
    plt.xlabel(f'Area [PE]')
    if log:
        xt = np.arange(int(plt.xticks()[0].min()), int(plt.xticks()[0].max()))
        plt.xticks(xt, 10 ** xt)
        yt = np.arange(int(plt.yticks()[0].min()), int(plt.yticks()[0].max()))
        plt.yticks(yt, 10 ** yt)
    plt.colorbar(label='counts/bin')


def plots_area_vs_aft(data, log=None, **kwargs):
    """basic wrapper to plot area vs aft"""
    x, y = data['area'], data['area_fraction_top']
    if log:
        x, y = np.log10(x), np.log10(y)
    plt.hist2d(x, y, norm=LogNorm(), **kwargs)
    plt.ylabel(f'Aft')
    plt.xlabel(f'Area [PE]')
    if log:
        xt = np.arange(int(plt.xticks()[0].min()), int(plt.xticks()[0].max()))
        plt.xticks(xt, 10 ** xt)
        yt = np.arange(int(plt.yticks()[0].min()), int(plt.yticks()[0].max()))
        plt.yticks(yt, 10 ** yt)
    plt.colorbar(label='counts/bin')


def show_dat_over_time(dat, bins=100, **kwargs):
    """basic wrapper to plot counts for data per unit time (binned by n bins)"""
    # plot pulse rate
    ax = plt.gca()
    counts, bin_edges = np.histogram(dat['time'], bins=bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])
    counts, bin_centers = counts[counts > 0], bin_centers[counts > 0]
    ax.errorbar(
        [datetime.datetime.fromtimestamp(t / 1e9)
         for t in bin_centers],
        counts,
        yerr=np.sqrt(counts),
        **kwargs)
    # ax
    # Make legend
    ax.legend(loc='lower left')
    ax.set_ylabel('rate [Hz]')
    plt.xticks(rotation=30)


def save_canvas(name, save_dir='./figures', tight_layout=False):
    """Wrapper for saving current figure"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + '/.')
    if not os.path.exists(save_dir + '/pdf/.'):
        os.makedirs(save_dir + '/pdf/.')
    if not os.path.exists(save_dir + '/svg/.'):
        os.makedirs(save_dir + '/svg/.')
    if tight_layout:
        plt.tight_layout()
    if os.path.exists(save_dir) and os.path.exists(save_dir + '/pdf'):
        plt.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_dir}/pdf/{name}.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_dir}/svg/{name}.svg", dpi=300, bbox_inches="tight")
    else:
        raise FileExistsError(f'{save_dir} does not exist or does not have /pdf')


def plot_spectrum(data, log=None, **kwargs):
    """Simple function for making an spectrum of some data."""
    x = data['area']
    if log:
        x = np.log10(x)
    plt.hist(x, **kwargs)
    plt.ylabel(f'Counts/bin')
    plt.xlabel(f'Area [PE]')
    plt.yscale('log')
    if log:
        xt = np.arange(int(plt.xticks()[0].min()), int(plt.xticks()[0].max()))
        plt.xticks(xt, 10. ** xt)
    plt.show()


@straxen.mini_analysis(
    requires=('records',),
    default_time_selection='touching',
    warn_beyond_sec=1)
def plot_records_hitpatern(records, seconds_range, t_reference):
    records = strax.sort_by_time(records)
    res = np.zeros(straxen.n_tpc_pmts)
    res = smash_records(records, res)
    straxen.plot_pmts(res)


@numba.jit
def smash_records(records, buffer):
    """Sum records per channel"""
    for ch in range(len(buffer)):
        selection = records[records['channel'] == ch]
        buffer[ch] = np.sum(selection['area'])
    return buffer


def sample_peaks(st,
                 peaks, runs, max_peaks=3,
                 save_as='s1_peaks',
                 log=True,
                 **kwargs
                 ):
    i = 0
    if type(peaks) == pd.core.frame.DataFrame:
        _iter = peaks.iterrows()
    else:
        _iter = enumerate(peaks)

    def get_run_start(peak):
        this_run_id = runs['name'].values[0]
        this_start = st.estimate_run_start(this_run_id)
        for k, _run_id in enumerate(runs['name']):
            if k == len(runs) - 1:
                break
            next_run_id = runs['name'].values[k + 1]
            next_start = st.estimate_run_start(next_run_id)
            if next_start > peak['time']:
                break
            this_start = next_start
            this_run_id = next_run_id

        return this_run_id, this_start

    for _, p in _iter:
        print(i)
        if i > max_peaks:
            break
        run_id, run_start = get_run_start(p)
        plt.figure(figsize=(14, 11))
        ax0 = plt.subplot(212)
        plot_classified_peak(st, p, run_id=run_id, single_figure=False, **kwargs)
        ax1, ax2 = plt.subplot(221), plt.subplot(222)
        plot_pmts(p['area_per_channel'],
                  vmin=1 if log else None,
                  log_scale=log,
                  axes=[ax1, ax2],
                  label='Area per channel [PE]')

        save_canvas(f'peak_{i}_t{p["time"]}', save_dir=f'./figures/example_{save_as}/')
        plt.show()
        i += 1


# Copied from straxen.analysis
def plot_classified_peak(st, p, t_reference=None,
                         seconds_range=None,
                         run_id=None,
                         single_figure=True, figsize=(10, 4),
                         xaxis='since_run_start',
                         **kwargs):
    if not kwargs or 'color' not in kwargs:
        kwargs.update({'color': {0: 'gray', 1: 'b', 2: 'g'}[p['type']]})
    if seconds_range is None and run_id is not None:
        seconds_range = np.array([p['time'], strax.endtime(p)]) - st.estimate_run_start(run_id)
        seconds_range = seconds_range / int(1e9)
        t_reference = st.estimate_run_start(run_id)

    if single_figure:
        plt.figure(figsize=figsize)
    plt.axhline(0, c='k', alpha=0.2)

    plot_peak(p,
              t0=t_reference,
              color={0: 'gray', 1: 'b', 2: 'g'}[p['type']])

    if xaxis == 'since_peak':
        seconds_range_xaxis(seconds_range, t0=seconds_range[0])
    elif xaxis:
        seconds_range_xaxis(seconds_range)
    else:
        plt.xticks([])
    plt.xlim(*seconds_range)
    plt.ylabel("Intensity [PE/ns]")
    if single_figure:
        plt.tight_layout()


def plot_peak(p, t0=None, **kwargs):
    x, y = time_and_samples(p, t0=t0)
    kwargs.setdefault('linewidth', 1)

    # Plot waveform
    plt.plot(x, y,
             drawstyle='steps-pre',
             **kwargs)
    if 'linewidth' in kwargs:
        del kwargs['linewidth']
    kwargs['alpha'] = kwargs.get('alpha', 1) * 0.2
    plt.fill_between(x, 0, y, step='pre', linewidth=0, **kwargs)

    # Mark extent with thin black line
    plt.plot([x[0], x[-1]], [y.max(), y.max()],
             c='k', alpha=0.3, linewidth=1)


def seconds_range_xaxis(seconds_range, t0=None):
    """Make a pretty time axis given seconds_range"""
    plt.xlim(*seconds_range)
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    xticks = plt.xticks()[0]
    if not len(xticks):
        return

    #     xticks[0] = seconds_range[0]
    #     xticks[-1] = seconds_range[-1]

    # Format the labels
    # I am not very proud of this code...
    def chop(x):
        return np.floor(x).astype(np.int)

    if t0 is None:
        xticks_ns = np.round(xticks * int(1e9)).astype(np.int)
    else:
        xticks_ns = np.round((xticks - xticks[0]) * int(1e9)).astype(np.int)
    sec = chop(xticks_ns // int(1e9))
    ms = chop((xticks_ns % int(1e9)) // int(1e6))
    us = chop((xticks_ns % int(1e6)) // int(1e3))
    samples = chop((xticks_ns % int(1e3)) // 10)

    labels = [str(sec[i]) for i in range(len(xticks))]
    print_ns = np.any(samples != samples[0])
    print_us = print_ns | np.any(us != us[0])
    print_ms = print_us | np.any(ms != ms[0])
    if print_ms and t0 is None:
        labels = [l + f'.{ms[i]:03}' for i, l in enumerate(labels)]
        if print_us:
            labels = [l + r' $\bf{' + f'{us[i]:03}' + '}$'
                      for i, l in enumerate(labels)]
            if print_ns:
                labels = [l + f' {samples[i]:02}0' for i, l in enumerate(labels)]
        plt.xticks(ticks=xticks, labels=labels, rotation=90)
    else:
        labels = list(chop((xticks_ns // 10) * 10))
        labels[-1] = ""
        plt.xticks(ticks=xticks, labels=labels, rotation=0)
    if t0 is None:
        plt.xlabel("Time since run start [sec]")
    else:
        plt.xlabel("Time [ns]")


def time_and_samples(p, t0=None):
    """Return (x, y) numpy arrays for plotting the waveform data in p
    using 'steps-pre'.
    Where x is the time since t0 in seconds (or another time_scale),
      and y is intensity in PE / ns.
    :param p: Peak or other similar strax data type
    :param t0: Zero of time in ns since unix epoch
    """
    n = int(p['length'])
    if t0 is None:
        t0 = p['time']
    x = ((p['time'] - t0) + np.arange(n + 1) * p['dt']) / int(1e9)
    y = p['data'][:n] / p['dt']
    return x, np.concatenate([[y[0]], y])


def plot_pmts(
        c, label='',
        figsize=None,
        xenon1t=False,
        show_tpc=True,
        axes=None,
        extend='neither', vmin=None, vmax=None,
        **kwargs):
    """Plot the PMT arrays side-by-side, coloring the PMTS with c.
    :param c: Array of colors to use. Must have len() n_tpc_pmts
    :param label: Label for the color bar
    :param figsize: Figure size to use.
    :param extend: same as plt.colorbar(extend=...)
    :param vmin: Minimum of color scale
    :param vmax: maximum of color scale
    Other arguments are passed to plot_on_single_pmt_array.
    """
    if vmin is None:
        vmin = np.nanmin(c)
    if vmax is None:
        vmax = np.nanmax(c)
    if vmin == vmax:
        # Single-valued array passed
        vmax += 1
    if figsize is None:
        figsize = (11, 4) if xenon1t else (13, 5.5)

    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=figsize)

    for array_i, array_name in enumerate(['top', 'bottom']):
        ax = axes[array_i]
        plt.sca(ax)
        plt.title(array_name.capitalize())

        straxen.plot_on_single_pmt_array(
            c,
            xenon1t=xenon1t,
            array_name=array_name,
            show_tpc=show_tpc,
            vmin=vmin, vmax=vmax,
            **kwargs)

    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position('right')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.colorbar(ax=axes, extend=extend, label=label)


@straxen.mini_analysis(
    requires=('peaks', 'peak_basics'),
    default_time_selection='touching',
    warn_beyond_sec=60)
def plot_peaks(peaks, seconds_range, t_reference, show_largest=100,
               single_figure=True, figsize=(10, 4), xaxis=True):
    if single_figure:
        plt.figure(figsize=figsize)
    plt.axhline(0, c='k', alpha=0.2)

    peaks = peaks[np.argsort(-peaks['area'])[:show_largest]]
    peaks = strax.sort_by_time(peaks)

    for p in peaks:
        plot_peak(p,
                  t0=t_reference,
                  color={0: 'gray', 1: 'b', 2: 'g'}[p['type']])

    if xaxis == 'since_start':
        seconds_range_xaxis(seconds_range, t0=seconds_range[0])
    elif xaxis:
        seconds_range_xaxis(seconds_range)
    #     else:
    #         plt.xticks([])
    #     plt.xlim(*seconds_range)
    plt.ylabel("Intensity [PE/ns]")
    if single_figure:
        plt.tight_layout()


@straxen.mini_analysis(
    requires=('peaks', 'peak_basics'),
    default_time_selection='touching',
    warn_beyond_sec=60)
def plot_pattern(peaks, seconds_range, t_reference,
                 axes=None,
                 vmin=None,
                 log_scale=False,
                 label=None,
                 single_figure=False,
                 figsize=(10, 4), ):
    if single_figure:
        plt.figure(figsize=figsize)
    if len(peaks) > 1:
        print(f'warning showing total area of {len(peaks)} peaks')
    plot_pmts(np.sum(peaks['area_per_channel'], axis=0),
              axes=axes, vmin=vmin, log_scale=log_scale, label=label)


def sample_around_peaks(st, peaks, runs, max_peaks=3, save_as='s1_peaks',
                        log=True, plot_extension=5_000, **kwargs):
    if type(peaks) == pd.core.frame.DataFrame:
        _iter = peaks.iterrows()
    else:
        _iter = enumerate(peaks)

    for i, p in _iter:
        if i > max_peaks:
            break
        plot_wf(st, p, runs, log=log, plot_extension=plot_extension, **kwargs)
        save_canvas(f'peak_{i}_t{p["time"]}', save_dir=f'./figures/example_{save_as}/')
        plt.show()


def plot_wf(st, peaks, runs, log=True, plot_extension=5_000, hit_pattern=True,
            timestamp = True, time_fmt = "%d-%b-%Y (%H:%M:%S)",
            **kwargs):
    p = peaks

    if isinstance(runs, str):
        print(f'Assuming {runs} is a run id')
        runs = pd.DataFrame({'name':[runs]})
    def get_run_start(peak):
        this_run_id = runs['name'].values[0]
        this_start = st.estimate_run_start(this_run_id)
        for k, _run_id in enumerate(runs['name']):
            if k == len(runs) - 1:
                break
            next_run_id = runs['name'].values[k + 1]
            next_start = st.estimate_run_start(next_run_id)
            if next_start > peak['time'].max():
                break
            this_start = next_start
            this_run_id = next_run_id

        return this_run_id, this_start

    run_id, run_start = get_run_start(p)
    t_range = np.array([p['time'].min(), p['endtime'].max()])
    if not np.iterable(plot_extension):
        t_range += np.array([-plot_extension, plot_extension])
    elif len(plot_extension) == 2:
        t_range += plot_extension
    else:
        raise ValueError('Wrong dimensions for plot_extension. Use scalar or object of len( ) == 2')
    t_range -= run_start
    t_range = t_range / 10 ** 9
    t_range = np.clip(t_range, 0, np.inf)

    try:
        if hit_pattern:
            plt.figure(figsize=(14, 11))
            ax0 = plt.subplot(212)
        else:
            plt.figure(figsize=(14, 5))
        plot_peaks(st, run_id, seconds_range=t_range, single_figure=False,
                   **kwargs)
        if timestamp:
            _ax = plt.gca()
            t_stamp = datetime.datetime.fromtimestamp(peaks['time'].min()/10**9).strftime(time_fmt)
            _ax.text(0.975, 0.925, t_stamp, 
                    horizontalalignment='right',
                    verticalalignment='top', transform=_ax.transAxes)
        if hit_pattern:
            axes = plt.subplot(221), plt.subplot(222)
            plot_pattern(st, run_id, seconds_range=t_range,
                         axes=axes,
                         vmin=1 if log else None,
                         log_scale=log,
                         label='Area per channel [PE]')

    except (ValueError,
            ZeroDivisionError,
            strax.mailbox.MailboxKilled,
            RuntimeError) as e:
        if np.all(plot_extension == 0):
            warnings.warn(f'Failed despite 0 ns extension. Ran into {e}')
            plt.clf()
            return
        else:
            warnings.warn('Failed to deliver. Trying with no extension')
            return plot_wf(st, peaks, runs, log=log, plot_extension=0,
                           hit_pattern = hit_pattern, **kwargs)

