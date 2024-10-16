import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter, YearLocator
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter, LogFormatter, MultipleLocator, AutoMinorLocator
import matplotlib as mpl
import numpy as np


rf_psd_pkl = "Fig3_PSD.pkl"
pertb_pkl = "Fig3_pertb.pkl"



rf_psd_df = pd.read_pickle(rf_psd_pkl)
pertb_df = pd.read_pickle(pertb_pkl)


rf_psd_df["start"] = pd.to_datetime(rf_psd_df["start"]).dt.date
rf_psd_df["end"] = pd.to_datetime(rf_psd_df["end"]).dt.date
rf_psd_df["mid"] = pd.to_datetime(rf_psd_df["mid"]).dt.date

pertb_df["date-time"] = pd.to_datetime(pertb_df["date-time"]).dt.date

x = mdates.date2num(rf_psd_df["mid"].to_numpy())

y_true = np.unique(np.array(rf_psd_df["CP_PSD_rad"].to_list()))
extent_true = [np.min(x), np.max(x), np.min(np.unique(y_true)),  np.max(np.unique(y_true))]
zr_true = 10*np.log10(rf_psd_df["RM_PSD_rad"].to_numpy().tolist())
zt_true = 10*np.log10(rf_psd_df["RM_PSD_tr"].to_numpy().tolist())
z_true = np.append(zr_true, zt_true)

y_syn = np.unique(np.array(rf_psd_df["CP_PSD_rad_syn"].to_list()))
extent_syn = [np.min(x), np.max(x), np.min(np.unique(y_syn)),  np.max(np.unique(y_syn))]
zr_syn = 10*np.log10(rf_psd_df["RM_PSD_rad_syn"].to_numpy().tolist())
zt_syn = 10*np.log10(rf_psd_df["RM_PSD_tr_syn"].to_numpy().tolist())
z_syn = np.append(zr_syn, zt_syn)

z_true_sin = np.append(z_true, z_syn)

fig, axes = plt.subplots(nrows=5,ncols=2, figsize=(9,8), sharex=False, \
     gridspec_kw={"height_ratios":[0.6,0.4,0.6,0.4, 0.05], \
        "hspace":0.5, "wspace":0.35, "width_ratios":[1, 1]})

cmap = mpl.colormaps["plasma"]
bounds =np.linspace(np.min(z_true_sin), np.max(z_true_sin), 10)
norm  = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)


y_ticks = np.array([0.1,0.2, 0.5, 1, 2, 5, 10, 15, 20])
y_ticks = y_ticks[(y_ticks>=np.min(y_true))*(y_ticks<=np.max(y_true))]

cr_true = axes[0,0].pcolormesh(x,y_true,zr_true.T, cmap=cmap, norm=norm, shading='gouraud')
ct_true = axes[0,1].pcolormesh(x,y_true,zt_true.T, cmap=cmap, norm=norm, shading='gouraud')
axes[0,0].set_ylabel("Period (s)", fontsize=8)
axes[0,0].tick_params(axis='both', which='major', labelsize=8)
axes[0,1].set_ylabel("Period (s)", fontsize=8)
axes[0,1].tick_params(axis='both', which='major', labelsize=8)

cr_syn = axes[2,0].pcolormesh(x,y_syn,zr_syn.T, cmap=cmap, norm=norm, shading='gouraud')
ct_syn = axes[2,1].pcolormesh(x,y_syn,zt_syn.T, cmap=cmap, norm=norm, shading='gouraud')
axes[2,0].set_ylabel("Period (s)", fontsize=8)
axes[2,0].tick_params(axis='both', which='major', labelsize=8)
axes[2,1].set_ylabel("Period (s)", fontsize=8)
axes[2,1].tick_params(axis='both', which='major', labelsize=8)

axes[1,0].plot(pertb_df["date-time"], pertb_df["pertb"], color='red', label="Normalised GPS displacement (East)")
axes[1,0].set_ylabel("Displacement", fontsize=8)
axes[1,0].set_xlim(np.min(rf_psd_df["mid"]), np.max(rf_psd_df["mid"]))
axes[1,0].set_ylim(-1.1,1.1)
axes[1,0].tick_params(axis='both', which='major', labelsize=8)
axes[1,0].xaxis.set_major_locator(YearLocator(base=2, month=1, day=1))

axes10 = axes[1,0].twinx()
axes10.plot(rf_psd_df["mid"], rf_psd_df["Spower_rad"], color='k', label="Total spectral power")
axes10.set_ylabel("Power (W)", fontsize=8)
axes10.yaxis.offsetText.set_fontsize(8)
axes10.set_xlim(np.min(rf_psd_df["mid"]), np.max(rf_psd_df["mid"]))
axes10.tick_params(axis='both', which='major', labelsize=8)

axes[1,1].plot(pertb_df["date-time"], pertb_df["pertb"], color='red', label="Normalised GPS displacement (East)")
axes[1,1].set_ylabel("Displacement", fontsize=8)
axes[1,1].set_xlim(np.min(rf_psd_df["mid"]), np.max(rf_psd_df["mid"]))
axes[1,1].set_ylim(-1.1,1.1)
axes[1,1].tick_params(axis='both', which='major', labelsize=8)
axes[1,1].xaxis.set_major_locator(YearLocator(base=2, month=1, day=1))
axes11 = axes[1,1].twinx()
axes11.plot(rf_psd_df["mid"], rf_psd_df["Spower_tr"], color='k', label="Total spectral power")

axes11.set_ylabel("Power (W)", fontsize=8)
axes11.yaxis.offsetText.set_fontsize(8)
axes11.set_xlim(np.min(rf_psd_df["mid"]), np.max(rf_psd_df["mid"]))
axes11.tick_params(axis='both', which='major', labelsize=8)

axes[3,0].plot(pertb_df["date-time"], (pertb_df["vp_syn"].apply(lambda x: x[3])/pertb_df["vs_perturb_syn"].apply(lambda x: x[3])), color='g', label="$V_P/V_S$ base of UC", alpha=1)
axes[3,0].plot(pertb_df["date-time"], (pertb_df["vp_syn"].apply(lambda x: x[4])/pertb_df["vs_perturb_syn"].apply(lambda x: x[4])), color='blue', label="$V_P/V_S$ LVL", alpha=1)
axes[3,0].set_ylabel("$V_P/V_S$", fontsize=8)
axes[3,0].set_yticks(np.arange(1.4,2.4,0.2))
axes[3,0].yaxis.set_major_locator(MultipleLocator(0.2))
axes[3,0].yaxis.set_minor_locator(MultipleLocator(0.1))

axes[3,0].tick_params(axis='both', which='major', labelsize=8)
axes[3,0].xaxis.set_major_locator(YearLocator(base=2, month=1, day=1))
axes[3,0].set_xlim(np.min(rf_psd_df["mid"]), np.max(rf_psd_df["mid"]))
axes[3,0].yaxis.offsetText.set_fontsize(8)

axes30 = axes[3,0].twinx()
axes30.plot(rf_psd_df["mid"], rf_psd_df["Spower_rad_syn"], color='k', label="Total spectral power")
axes30.set_ylabel("Power (W)", fontsize=8)
axes30.yaxis.offsetText.set_fontsize(8)
axes30.set_xlim(np.min(rf_psd_df["mid"]), np.max(rf_psd_df["mid"]))
axes30.tick_params(axis='both', which='major', labelsize=8)

axes[3,1].plot(pertb_df["date-time"], (pertb_df["vp_syn"].apply(lambda x: x[3])/pertb_df["vs_perturb_syn"].apply(lambda x: x[3])), color='g', label="$V_P/V_S$ base of UC", alpha=1)
axes[3,1].plot(pertb_df["date-time"], (pertb_df["vp_syn"].apply(lambda x: x[4])/pertb_df["vs_perturb_syn"].apply(lambda x: x[4])), color='blue', label="$V_P/V_S$ LVL", alpha=1)
axes[3,1].set_yticks(np.arange(1.4,2.4,0.2))
axes[3,1].yaxis.set_major_locator(MultipleLocator(0.2))
axes[3,1].yaxis.set_minor_locator(MultipleLocator(0.1))
axes[3,1].tick_params(axis='both', which='major', labelsize=8)
axes[3,1].xaxis.set_major_locator(YearLocator(base=2, month=1, day=1))
axes[3,1].set_xlim(np.min(rf_psd_df["mid"]), np.max(rf_psd_df["mid"]))
axes[3,1].yaxis.offsetText.set_fontsize(8)

axes31 = axes[3,1].twinx()
axes31.plot(rf_psd_df["mid"], rf_psd_df["Spower_tr_syn"], color='k', label="Total spectral power")
axes31.set_ylabel("Power (W)", fontsize=8)
axes31.yaxis.offsetText.set_fontsize(8)
axes31.set_xlim(np.min(rf_psd_df["mid"]), np.max(rf_psd_df["mid"]))
axes31.tick_params(axis='both', which='major', labelsize=8)

axes[0,0].set_yscale('log')
axes[0,0].set_yticks(y_ticks)
axes[0,0].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
axes[0,0].set_xlim(np.min(rf_psd_df["mid"]), np.max(rf_psd_df["mid"]))
axes[0,0].xaxis_date()
axes[0,0].xaxis.set_major_locator(YearLocator(base=2, month=1, day=1))
axes[0,0].set_title("(a) Observed radial RF PSDs", fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.7), loc='left', x=0.01, y=0.91, verticalalignment='top')

axes[0,1].set_yscale('log')
axes[0,1].set_yticks(y_ticks)
axes[0,1].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
axes[0,1].xaxis_date()
axes[0,1].xaxis.set_major_locator(YearLocator(base=2, month=1, day=1))
axes[0,1].set_title("(b) Observed transverse RF PSDs", fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.7), loc='left', x=0.01, y=0.91, verticalalignment='top')

axes[2,0].set_yscale('log')
axes[2,0].set_yticks(y_ticks)
axes[2,0].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
axes[2,0].xaxis_date()
axes[2,0].xaxis.set_major_locator(YearLocator(base=2, month=1, day=1))
axes[2,0].set_title("(c) Predicted radial RF PSDs", fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.7), loc='left', x=0.01, y=0.91, verticalalignment='top')

axes[2,1].set_yscale('log')
axes[2,1].set_yticks(y_ticks)
axes[2,1].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
axes[2,1].xaxis_date()
axes[2,1].xaxis.set_major_locator(YearLocator(base=2, month=1, day=1))
axes[2,1].set_title("(d) Predicted transverse RF PSDs", fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.7), loc='left', x=0.01, y=0.91, verticalalignment='top')


gs = axes[4,0].get_gridspec()
axes[4,1].remove()
axes[4,0].remove()

cmap_axes = fig.add_subplot(gs[4, :])
cbar = fig.colorbar(cr_true, cax=cmap_axes, label="PSD (dB)", format='%d', pad=-0.0, orientation="horizontal")
cbar.ax.tick_params(labelsize=8) 
cbar.set_label( label="PSD (dB)", fontsize=8, weight='bold')

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

unique_labels = set(labels)
legend_dict = dict(zip(labels, lines))
unique_lines = [legend_dict[x] for x in unique_labels]


plt.savefig("True_Syn_PSD.png", dpi=500, bbox_inches='tight')