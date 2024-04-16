import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter, LogFormatter
import matplotlib as mpl
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance


wfolder = "/home/pasan/Desktop/Repositories/RF_PSD_Timelapse/"
station = "RATZ"  
network = "NZ"

outname = "Test1" ## output file name prefix for pkl files with psds and rfs




## make folder to output files
psd_folder = "{:s}/PSD/{:s}/".format(station, outname)

rf_psd_pkl = "{:s}/{:s}_PSD.pkl".format(psd_folder, outname)

rf_psd_df = pd.read_pickle(rf_psd_pkl)

# pertb_pkl = "{:s}/{:s}_pertb.pkl".format(psd_folder, outname)
# pertb_df = pd.read_pickle(pertb_pkl)


rf_psd_df["start"] = pd.to_datetime(rf_psd_df["start"]).dt.date
rf_psd_df["end"] = pd.to_datetime(rf_psd_df["end"]).dt.date
rf_psd_df["mid"] = pd.to_datetime(rf_psd_df["mid"]).dt.date


x = mdates.date2num(rf_psd_df["mid"].to_numpy())

y_true = np.unique(np.array(rf_psd_df["CP_PSD_rad"].to_list()))
extent_true = [np.min(x), np.max(x), np.min(np.unique(y_true)),  np.max(np.unique(y_true))]
zr_true = 10*np.log10(rf_psd_df["RM_PSD_rad"].to_numpy().tolist())
zt_true = 10*np.log10(rf_psd_df["RM_PSD_tr"].to_numpy().tolist())
z_true = np.append(zr_true, zt_true)


fig, axes = plt.subplots(nrows=3,ncols=1, figsize=(6,6), sharex=False, \
     gridspec_kw={"hspace":0.30, "height_ratios":[1, 1, 0.05], })
    #  gridspec_kw={"height_ratios": [0.3,0.1, 0.3, 0.1]}, constrained_layout=True)

cmap = mpl.colormaps["viridis"]
bounds =np.linspace(np.min(z_true), np.max(z_true), 10)
norm  = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)


y_ticks = np.array([0.1,0.2, 0.5, 1])
y_ticks = y_ticks[(y_ticks>=np.min(y_true))*(y_ticks<=np.max(y_true))]

cr_true = axes[0].pcolormesh(x,y_true,zr_true.T, cmap=cmap, norm=norm, shading='gouraud')
ct_true = axes[1].pcolormesh(x,y_true,zt_true.T, cmap=cmap, norm=norm, shading='gouraud')
# s se es 


axes[0].set_ylabel("Period (s)", fontsize=10, fontweight="bold")
axes[0].tick_params(axis='both', which='major', labelsize=10)
axes[1].set_ylabel("Period (s)", fontsize=10, fontweight="bold")
axes[1].tick_params(axis='both', which='both', labelsize=10)
# ax.tick_params(axis="x",direction="in", pad=-15)


axes[0].set_yscale('log')
axes[0].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
axes[0].set_yticks(y_ticks)
# axes[0].set_ylim(y_true[0], 1)
axes[0].set_xlim(np.min(x), np.max(x))
# axes[0].xaxis_date()
axes[0].xaxis.set_major_locator(YearLocator(base=2, month=1, day=1))
axes[0].xaxis.set_minor_locator(MonthLocator(interval=6))
axes[0].xaxis.set_major_formatter(DateFormatter("%Y"))
axes[0].set_title("(a) Observed radial RF PSDs", fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.7), loc='left', x=0.02, y=0.91, verticalalignment='top')

axes[1].set_yscale('log')
axes[1].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
# axes[1].set_ylim(y_true[0], 1)
axes[1].set_yticks(y_ticks)
axes[1].set_xlim(np.min(x), np.max(x))
# axes[1].xaxis_date()
axes[1].xaxis.set_major_locator(YearLocator(base=2, month=1, day=1))
axes[1].xaxis.set_minor_locator(MonthLocator(interval=6))
axes[1].xaxis.set_major_formatter(DateFormatter("%Y"))
axes[1].set_title("(b) Observed transverse RF PSDs", fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.7), loc='left', x=0.02, y=0.91, verticalalignment='top')

   

# fig.autofmt_xdate(rotation=45)

# gs = axes[4,0].get_gridspec()
# axes[4,1].remove()
# # cmap_axes = fig.add_subplot(gs[3, :])
cbar = fig.colorbar(cr_true, cax=axes[2], format='%d', pad=-0.0, orientation="horizontal")
cbar.ax.tick_params(labelsize=8) 
# cbar.set_label( label="PSD (dB)", fontsize=8, weight='bold', loc="left")

axes[2].set_title("PSD\n(dB)", fontsize=8, fontweight="bold", x=-0.01, y=-2, ha="right")
# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# # grab unique labels
# unique_labels = set(labels)

# # assign labels and legends in dict
# legend_dict = dict(zip(labels, lines))

# # query dict based on unique labels
# unique_lines = [legend_dict[x] for x in unique_labels]

# fig.legend(unique_lines, unique_labels, ncol=2, bbox_to_anchor=(0.95,0.14), fontsize=8, frameon=False)

# fig.autofmt_xdate()
plt.savefig(psd_folder+"True_PSD_no_gps_tp.png", dpi=500, bbox_inches='tight')
# plt.show()