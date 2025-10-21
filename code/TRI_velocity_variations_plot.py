#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mo Oct 20 09:25:31 2025

@author: Armin Dachauer
"""


#%%
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

#load functions
import functions as fnc


#define colors
golden_palm = "#AA8805"
scarlet_smile = "#FF2400"
aventurine = "#008080"


################  define other parameters
#  choose year
# year = 2023
year = 2024

current_folder = Path(__file__).parent
###########################################




#%%  Figure 2 and 3

#load velocity centerline data
if year == 2024: 
    cl = pd.read_csv(current_folder.parent / "data/velocity/EKaS24_veloc_timeseries_centerline.csv").T
    start_time_all = pd.Timestamp(str(datetime.datetime(2024, 7, 12, 0, 30)), tz='UTC')  
    end_time_all = pd.Timestamp(str(datetime.datetime(2024, 7, 25, 18, 37)), tz='UTC')   
if year == 2023:
    # load veloc
    cl = pd.read_csv(current_folder.parent / "data/velocity/EKaS23_veloc_timeseries_centerline.csv").T
    start_time_all = pd.Timestamp(str(datetime.datetime(2023, 8, 3, 15, 47)), tz='UTC')  
    end_time_all = pd.Timestamp(str(datetime.datetime(2023, 8, 15, 21, 47)), tz='UTC')    
#process velocity centerline data
cl.columns = cl.iloc[0] # Set the first row as the header
cl = cl.drop(cl.index[0]) # Drop the first row as it's now the header
#filter by time
cl.columns = pd.to_datetime(cl.columns)# Convert column names to datetime objects
cl_filtered_columns = cl.columns[(cl.columns >= start_time_all) & (cl.columns <= end_time_all)]
cl = cl[cl_filtered_columns]
#filter by part of centerline
filter_start=7
filter_end = 30
cl = cl.iloc[filter_start:filter_end]
#calc mean of centerline
cl_mean = cl.mean() * (-1)
cl_mean = pd.DataFrame(cl_mean, columns=["cl_mean"])
cl_mean["date"] = cl_mean.index
#change the centerline velocs sign
cl_new = cl*-1
# Compute the mean across the specified axis 
cl_mean['mean_smooth'] = fnc.butter_lowpass_filter(cl_new.T, axis=1, cutoff_hours=12, fs=2.0)
#fill time gaps with nans
data, time = fnc.interpolate_time_and_fill_gaps(cl_mean.cl_mean, cl_mean.date, freq='0.5H')
mean_smooth_gaps, time_smooth = fnc.interpolate_time_and_fill_gaps(cl_mean.mean_smooth, cl_mean.date, freq='0.5H')


#load weather station data
if year == 2024: 
    #load temperature data hill
    ws_hill = pd.read_csv(current_folder.parent / "data/meteo/meteo_hill15-25Jul24.csv", header=2, sep=",")
    ws_hill['date'] = pd.to_datetime(ws_hill['Timestamp'], utc=True)
    ws_hill=ws_hill[(ws_hill.date >= start_time_all) & (ws_hill.date <= end_time_all)]
    ws_hill=ws_hill[ws_hill['RH Relative Humidity']!=0] #filter air temperature after disconnection -> here RH = 0
    ws_hill['DailyMeanTemp'] = ws_hill.groupby(ws_hill['date'].dt.date)['°C Air Temperature'].transform('mean')  #create daily mean temperature    
    #create hourly precip sum
    ws_hill.set_index('date', inplace=True)# Resample the data to hourly intervals and sum the precipitation values for each hour
    ws_hill['HourlyPrecip'] = ws_hill['mm Precipitation'].resample('H').sum()
    ws_hill.reset_index(inplace=True) # Reset index if you want to work with 'date' as a column again
if year==2023:
    #load meteo data
    df_meteo = pd.read_csv(current_folder.parent / "data/meteo/EM26996_28Aug23-1458_all_data_cleaned.csv", header=2)  #fjord
    df_meteo['date_dt_UTC'] = pd.to_datetime(df_meteo['Measurement Time'], format="%d.%m.%y %H:%M")   #, format="%d/%m/%Y %H:%M"
    df_meteo=df_meteo.sort_values(by='date_dt_UTC')
    df_meteo=df_meteo[df_meteo.date_dt_UTC<end_time_all.to_datetime64()]
    df_meteo=df_meteo[df_meteo.date_dt_UTC>start_time_all.to_datetime64()].reset_index(drop=True)
    
#change starte/endtime to next/last day with data for entire day
start_time_all2 =  (start_time_all + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
end_time_all2 =  end_time_all.replace(hour=0, minute=0, second=0, microsecond=0)

#load Narsarsuaq weather station data for wind 
fn= print("data provided by Caroline Drost Jensen: https://www.dmi.dk/publikationer/ Weather Observations from Greenland 1958-2024, Narsarsuaq ID: 4270" )
df_AWS = pd.read_csv(fn, sep=";", dtype="float64")
df_AWS.columns=["station", "year", "month", "day", "hour", "Temp", "Tmax_1h", "T_max_12h","Tmin_1h", "T_min_12h", "RH", "veloc", "vmax_3s", "wd", "wd_1h", "p", "sun_dur", "in_globrad", "prec_1h", "prec_12h", "prec_24h", "cloud_cov" ]
df_AWS["date"]=pd.to_datetime(dict(year=df_AWS.year, month=df_AWS.month, day=df_AWS.day, hour=df_AWS.hour, minute=0))
# filter for days
df_AWS = df_AWS[(df_AWS.date >= start_time_all2.to_datetime64()) & (df_AWS.date <= end_time_all2.to_datetime64())]
#process wind data
df_AWS['radians'] = np.radians(df_AWS['wd'])
# Calculate U and V components
df_AWS['u'] = -df_AWS['veloc'] * np.sin(df_AWS['radians'])  # U component (east-west)
df_AWS['v'] = -df_AWS['veloc'] * np.cos(df_AWS['radians'])  # V component (north-south)
# Group by day and calculate daily mean U and V components
daily_avg = df_AWS.groupby(df_AWS['date'].dt.date).agg({
    'u': 'mean',
    'v': 'mean',
    'veloc': 'mean',
})
# Calculate daily mean wind direction
daily_avg['mean_direction'] = np.arctan2(daily_avg['v'], daily_avg['u'])  # In radians

    
#load plume/melange and calving
if year == 2023:
    df_plume = pd.read_csv(current_folder.parent / "data/other/plume_investigation_fieldwork_2023.csv", sep=",")
    df_calv = pd.read_csv(current_folder.parent / "data/other/all_events_V_coord_sorted_max15_2023.csv")
if year == 2024:
    df_plume = pd.read_csv(current_folder.parent / "data/other/plume_investigation_fieldwork_2024.csv", sep=",")
    df_calv = pd.read_csv(current_folder.parent / "data/other/all_events_V_coord_sorted_max15_2024.csv")

#add time to date
df_plume['date']=pd.to_datetime(df_plume['date'])

# sorted calving events
df_calv['date_dt_UTC'] = pd.to_datetime(df_calv.time, utc=True)
df_calv=df_calv.sort_values(by='date_dt_UTC')
#filter by date
df_calv=df_calv[df_calv.date_dt_UTC<end_time_all]
df_calv=df_calv[df_calv.date_dt_UTC>start_time_all].reset_index(drop=True)


#load tidedata 
df_tidep = pd.read_csv(current_folder.parent / "data/other/TidesEKAS_time_ppc_p1_23to25_30minres_v2.csv")  #bar  -> 1m ~ dbar
df_tidep['date_dt_UTC'] = pd.to_datetime(df_tidep.date_dt_UTC, utc=True)
df_tidep=df_tidep.sort_values(by='date_dt_UTC')
#filter by date
df_tidep=df_tidep[df_tidep.date_dt_UTC<end_time_all]
df_tidep=df_tidep[df_tidep.date_dt_UTC>start_time_all].reset_index(drop=True)



# Create a figure
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 0.24, 0.76, 1, 1, 1]})

#plot 1
ax1.plot(time, data, label='LOS velocity', color=golden_palm)
ax1.plot(time_smooth, mean_smooth_gaps, label='smoothed LOS velocity', color=aventurine, lw=4)
ax1.set_xlim([start_time_all, end_time_all])
if year == 2023: ax1.set_ylim([4.5, 6.8])
if year == 2024: ax1.set_ylim([5.8, 9.5])

# # add acc/dec map times
if year == 2024:
    acc_dec_starttimes = ['2024-07-22 22:00:00', '2024-07-19 20:30:00', '2024-07-20 08:30:00', '2024-07-24 08:30:00']
    acc_dec_endtimes = ['2024-07-22 23:30:59', '2024-07-19 22:00:59', '2024-07-20 10:00:00', '2024-07-24 10:00:00']
    ax1.axvspan(pd.Timestamp(acc_dec_starttimes[0], tz='UTC'), pd.Timestamp(acc_dec_endtimes[0], tz='UTC'), color='blue', alpha=0.3)
    ax1.axvspan(pd.Timestamp(acc_dec_starttimes[1], tz='UTC'), pd.Timestamp(acc_dec_endtimes[1], tz='UTC'), color='blue', label='deceleration transition', alpha=0.3)
    ax1.axvspan(pd.Timestamp(acc_dec_starttimes[2], tz='UTC'), pd.Timestamp(acc_dec_endtimes[2], tz='UTC'), color='red', alpha=0.3)
    ax1.axvspan(pd.Timestamp(acc_dec_starttimes[3], tz='UTC'), pd.Timestamp(acc_dec_endtimes[3], tz='UTC'), color='red', label='acceleration transition', alpha=0.3)

# add lake discharge timing
if year == 2024:
    acc_dec_starttimes = '2024-07-04 23:59:00'
    acc_dec_endtimes = '2024-07-15 23:59:00'
    ax1.axvspan(pd.Timestamp(acc_dec_starttimes, tz='UTC'), pd.Timestamp(acc_dec_endtimes, tz='UTC'), color='tab:blue', label='lake L1 emptying period', alpha=0.2)
    
# add lake discharge timing
if year == 2023:
    acc_dec_starttimes = ['2023-08-07 14:37:49', '2023-08-10 05:00:00']
    acc_dec_endtimes = ['2023-08-09 14:27:51', '2023-08-10 09:00:00']
    ax1.axvspan(pd.Timestamp(acc_dec_starttimes[0], tz='UTC'), pd.Timestamp(acc_dec_endtimes[0], tz='UTC'), color='tab:blue', label='lake L2 emptying period', alpha=0.2)
    ax1.axvspan(pd.Timestamp(acc_dec_starttimes[1], tz='UTC'), pd.Timestamp(acc_dec_endtimes[1], tz='UTC'), color='red', label='plume development', alpha=0.2)

ax1.set_xticklabels([])
ax1.set_ylabel('LOS velocity (m/d)')

#plot 2
x = np.arange(0, len(daily_avg), 1)
y = np.ones(len(daily_avg)) # Center position
width = 0.08  # Width of the arrow
head_width = 0.2 # Width of the arrow head
head_length = 0.4  # Length of the arrow head
length = np.ones(len(x)) *0.2  # Length of the arrow
x2, y2, dx, dy = fnc.calc_arrow(x, y, length, daily_avg['mean_direction'],head_length)
for (start_x, start_y, dx, dy) in zip(x2, y2, dx, dy):
    ax2.arrow(start_x, start_y, dx, dy, width=width, head_width=head_width, head_length=head_length, fc='k', ec='k')
if year==2024: ax2.set_xlim([-1.5, 12.4])
if year==2023: ax2.set_xlim([-0.8, 11.4])
ax2.set_ylim(ax2.get_ylim())  
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_yticks([])
ax2.spines['bottom'].set_visible(False)

#plot 3
if year==2024:
    ax3.plot(ws_hill['date'], ws_hill["RH Relative Humidity"]*100, color="grey", label='RH', zorder=2)
if year == 2023:
    ax3.plot(df_meteo.date_dt_UTC, df_meteo['RH']*100, color="grey", label='RH', zorder=2)
# #text wind speed below barb
for i, (date, group) in enumerate(list(daily_avg.iterrows())[0:]):
    if year == 2024: ax3.text(pd.to_datetime(daily_avg.index)[i] + pd.Timedelta(hours=12), 110, str(int(round(daily_avg['veloc'][i]*3.6))), ha='center', va='center', color='black', fontsize=10)
    if year ==2023: ax3.text(pd.to_datetime(daily_avg.index)[i] + pd.Timedelta(hours=12), 100, str(int(round(daily_avg['veloc'][i]*3.6))), ha='center', va='center', color='black', fontsize=10)
ax3.set_ylabel("RH (%)")
if year == 2024: ax3.set_ylim([24, 115])
if year == 2023: ax3.set_ylim([50, 105])
ax3.set_xlim([start_time_all, end_time_all])
ax3.set_ylim(ax3.get_ylim())  
ax3.spines['top'].set_visible(False)
ax3.set_xticklabels([])
handles1, labels1 = ax3.get_legend_handles_labels()
windarrow = Line2D([0], [0], color='k', lw=2, marker='>', markersize=10, markerfacecolor='k')
if year == 2024: ax3.legend(handles1+[windarrow], labels1+ ['wind (km/h)'], loc="lower right")
if year == 2023: ax3.legend(handles1+[windarrow], labels1+ ['wind (km/h)'], loc="lower right", ncol=2)

#plot 4  
if year==2024: 
    ax4_bar = ax4.twinx()
    ax4_bar.bar(ws_hill['date'], ws_hill['HourlyPrecip'], label='precipitation', color='#1B73A6', alpha=1, width=pd.Timedelta('1 hour'), zorder=1)
    ax4.plot(ws_hill['date'], ws_hill['°C Air Temperature'], color=scarlet_smile, label='air temperature', zorder=2)
    ax4_bar.set_ylabel("precipitation (mm/h)")
    handles1, labels1 = ax4.get_legend_handles_labels()
    handles2, labels2 = ax4_bar.get_legend_handles_labels()
    ax4.legend(handles1 + handles2 , labels1 + labels2 , ncol=1, loc='upper right')
if year==2023: 
    ax4.plot(df_meteo.date_dt_UTC, df_meteo['C Temp'], color=scarlet_smile, label="air temperature")
    handles1, labels1 = ax4.get_legend_handles_labels()
    ax4.legend(handles1, labels1, ncol=1, loc='lower left')
ax4.set_xticklabels([])
ax4.set_ylabel("air temperature (°C)")
ax4.set_xlim([start_time_all, end_time_all])
ax4.set_ylim(ax4.get_ylim())  

#plot 5
[ax5.axvline(x=row['date_dt_UTC'], color='#ADD8E6', linestyle='-') for _, row in df_calv.iterrows()] 
ax5.axvline(x=df_calv['date_dt_UTC'][0], color='#ADD8E6', linestyle='-', label='calving event') 
ax5.plot(np.repeat(df_plume['date'],2)[:-1], np.repeat(df_plume['plume'],2)[1:], label="plume extent", color='#C57644', linewidth=6)
ax5.plot(np.repeat(df_plume['date'],2)[:-1], np.repeat(df_plume['melange'],2)[1:], label="mélange", color='#668B8B', linewidth=4)
ax5.set_yticks([1, 2, 3, 4])
ylabels=["none", "weak/small", "medium", "strong/large"]
ax5.set_yticklabels(ylabels)
ax5.set_ylim([.5, 4.5])
ax5.set_xlim([start_time_all, end_time_all])
ax5.set_xticklabels([])
if year==2024: ax5.legend(ncol=3, loc='lower right')
if year==2023: ax5.legend(ncol=3, loc='lower left')


#plot 6 
ax6.plot(df_tidep['date_dt_UTC'], df_tidep['Tides'], label="tide", color='blue')
ax6.set_ylabel('tide (m)')
ax6.set_ylim(ax6.get_ylim())  
ax6.set_xlim([start_time_all, end_time_all])
ax6.set_xlabel('date')

    
# Add vertical lines at each day start
for day in df_AWS.date.dt.floor('D').unique():
    ax1.axvline(x=day, color='grey', linestyle='-', linewidth=0.2)
    ax2.axvline(x=day, color='grey', linestyle='-', linewidth=0.2)
    ax3.axvline(x=day, color='grey', linestyle='-', linewidth=0.2)
    ax5.axvline(x=day, color='grey', linestyle='-', linewidth=0.2)
    ax6.axvline(x=day, color='grey', linestyle='-', linewidth=0.2)

#add no data stripes
if year ==2023: 
    ax1.fill_betweenx(ax1.get_ylim(), pd.Timestamp(datetime.datetime(2023, 8, 5, 15, 47)), pd.Timestamp(datetime.datetime(2023, 8, 7, 21, 47)), color='lightgrey', label='no data', alpha=0.5)
    ax1.fill_betweenx(ax1.get_ylim(), pd.Timestamp(datetime.datetime(2023, 8, 9, 16, 47)), pd.Timestamp(datetime.datetime(2023, 8, 9, 19, 17)), color='lightgrey', alpha=0.5)
    ax5.fill_betweenx(ax5.get_ylim(), pd.Timestamp(datetime.datetime(2023, 8, 5, 15, 47)), pd.Timestamp(datetime.datetime(2023, 8, 7, 21, 47)), color='lightgrey', alpha=0.5)
    ax5.fill_betweenx(ax5.get_ylim(), pd.Timestamp(datetime.datetime(2023, 8, 9, 16, 47)), pd.Timestamp(datetime.datetime(2023, 8, 9, 19, 17)), color='lightgrey', alpha=0.5)
ax1.legend(loc='upper right')

#add a) b) c) labels
ax1.text(0.01, 0.975, 'a)', transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top')
ax2.text(0.01, 0.8, 'b)', transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top')
ax4.text(0.01, 0.95, 'c)', transform=ax4.transAxes, fontsize=20, fontweight='bold', va='top')
ax5.text(0.01, 0.95, 'd)', transform=ax5.transAxes, fontsize=20, fontweight='bold', va='top')
ax6.text(0.01, 0.95, 'e)', transform=ax6.transAxes, fontsize=20, fontweight='bold', va='top')


plt.subplots_adjust(hspace=0, wspace=0) #-0.4
plt.show()





#%% Figure 3 and 4

#########  filter centerline
if year == 2023: 
    filter_start= 4   #cut lowest part of centerline
    # filter_start= 10   #cut lowest part of centerline
    filter_end =  90   #cut upper part of centerline
    # filter_end =  50   #cut upper part of centerline
if year == 2024: 
    filter_start= 4 #8 #4  #cut lowest part of centerline
    # filter_start= 8     #cut lowest part of centerline
    filter_end =  90# 90   #cut upper part of centerline
########


#load smoothed centerline velocity (rollmean) and acceleration (rollmeangradient) data
if year==2023: cl_roll = pd.read_csv(current_folder.parent / 'data/velocity/EKaS23_veloc_rollmean_new_timeseries_centerline.csv').T
if year==2023: cl_acc = pd.read_csv(current_folder.parent / 'data/velocity/EKaS23_veloc_rollmeangradient_new_timeseries_centerline.csv').T 
if year==2024: cl_roll = pd.read_csv(current_folder.parent / 'data/velocity/EKaS24_veloc_rollmean_new_timeseries_centerline.csv').T    
if year==2024: cl_acc = pd.read_csv(current_folder.parent / 'data/velocity/EKaS24_veloc_rollmeangradient_new_timeseries_centerline.csv').T   
 
cl_roll.columns = cl_roll.iloc[0] # Set the first row as the header
cl_roll = cl_roll.drop(cl_roll.index[0]) # Drop the first row as it's now the header
cl_acc.columns = cl_acc.iloc[0] # Set the first row as the header
cl_acc = cl_acc.drop(cl_acc.index[0]) # Drop the first row as it's now the header

#filter by time
cl_roll.columns = pd.to_datetime(cl_roll.columns)# Convert column names to datetime objects
cl_roll_filtered_columns = cl_roll.columns[(cl_roll.columns >= start_time_all) & (cl_roll.columns <= end_time_all)]
cl_roll = cl_roll[cl_roll_filtered_columns]
cl_acc.columns = pd.to_datetime(cl_acc.columns)# Convert column names to datetime objects
cl_acc_filtered_columns = cl_acc.columns[(cl_acc.columns >= start_time_all) & (cl_acc.columns <= end_time_all)]
cl_acc = cl_acc[cl_acc_filtered_columns]

#filter by part of centerline
cl_roll = cl_roll.iloc[filter_start:filter_end]
cl_acc = cl_acc.iloc[filter_start:filter_end]

#get length of centerline
centerline=gpd.read_file(current_folder.parent / 'data/shapefiles/EKaS_radar_centerline2.shp')
interpolated_points = [centerline.geometry.iloc[0].interpolate(distance) for distance in range(0, int(centerline.geometry.iloc[0].length), int(centerline.geometry.iloc[0].length / 100))]  #get centerine in 100 parts
centerline_crop= LineString(interpolated_points[filter_start:filter_end]) #cut uppermost part
centerline_crop_len= centerline_crop.length / 1000  #get length in km

#calc mean of centerline
cl_roll_mean = cl_roll.mean() * (-1)
cl_roll_mean = pd.DataFrame(cl_roll_mean, columns=["cl_roll_mean"])
cl_roll_mean["date"] = cl_roll_mean.index
cl_acc_mean = cl_roll.mean() * (-1)
cl_acc_mean = pd.DataFrame(cl_acc_mean, columns=["cl_roll_mean"])
cl_acc_mean["date"] = cl_acc_mean.index

#change the centerline velocs sign
cl_roll_new = cl_roll*-1
cl_acc_new = cl_acc*-1

# Calculate deviation from mean at each time step
cl_roll_dev = cl_roll_new.subtract(cl_roll_new.mean(axis=1), axis=0)

# Convert all DataFrame values to numeric, coercing errors to NaN
cl_roll_new = cl_roll_new.apply(pd.to_numeric, errors='coerce')  # Convert to numeric values, setting non-numeric values to NaN
cl_roll_dev = cl_roll_dev.apply(pd.to_numeric, errors='coerce')  # Convert to numeric values, setting non-numeric values to NaN
cl_acc_new = cl_acc_new.apply(pd.to_numeric, errors='coerce')  # Convert to numeric values, setting non-numeric values to NaN

# Transpose the DataFrame to swap x and y axes
cl_roll_dev_transposed = cl_roll_dev.T
cl_roll_new_transposed = cl_roll_new.T
cl_acc_new_transposed = cl_acc_new.T

# Extract the dates from the columns and convert to datetime
time_values = mdates.date2num(cl_roll_dev_transposed.index)
time_values2 = mdates.date2num(cl_acc_new_transposed.index)

# Create a meshgrid for the time and the y-values (corresponding to distances)
X, Y = np.meshgrid(np.arange(cl_roll_dev_transposed.shape[1]), time_values)
X1, Y1 = np.meshgrid(np.arange(cl_roll_new_transposed.shape[1]), time_values)
X2, Y2 = np.meshgrid(np.arange(cl_acc_new_transposed.shape[1]), time_values2)

#add gap in plot in 2023
start_gap = mdates.date2num(datetime.datetime(2023, 8, 5, 15, 47))
end_gap = mdates.date2num(datetime.datetime(2023, 8, 7, 21, 47))

# Find the index corresponding to the start and end of the gap
x0 = cl_roll_dev_transposed.index[0]
for i in range(cl_roll_dev_transposed.shape[0]):
    if not cl_roll_dev_transposed.iloc[i].isna().all():
        first_non_nan_row = i
        break
last_non_nan_row = None
for i in range(cl_roll_dev_transposed.shape[0] - 1, -1, -1):
    if not cl_roll_dev_transposed.iloc[i].isna().all():
        last_non_nan_row = i
        break



# Plot 
fig = plt.figure(figsize=(10, 7))
outer_gs = GridSpec(2  , 1, height_ratios=[0.07, 2], hspace=0)  # No space between ax3 and ax1
inner_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[1], hspace=0.03)
ax1 = fig.add_subplot(inner_gs[0])
ax2 = fig.add_subplot(inner_gs[1])
ax4 = fig.add_subplot(inner_gs[2])
ax3 = ax4.twinx()  # Create a twin axis for ax4

#centerline veloc
if year == 2023: vmin=3.9; vmax=7; vmin_dev=-1.2; vmax_dev=1.2
if year == 2024: vmin=4.6; vmax=10; vmin_dev=-2; vmax_dev=2
#centerline acceleration
if year == 2023: vmin2=-3; vmax2=3
if year == 2024: vmin2=-3; vmax2=3


#plot 1 
im1 = ax1.pcolormesh(Y1, X1, cl_roll_new_transposed.values, vmin=vmin, vmax=vmax, cmap='magma_r', shading='auto')  
if year==2023: ax1.axvspan(start_gap, end_gap, color='lightgray', alpha=1)
ax1.set_xlim([time_values[int(first_non_nan_row)], time_values[int(last_non_nan_row)]])
#format y-axis by setting the centerline "centerline_crop" length for y axis
segment_length = centerline_crop_len / 86    # Calculate the segment length
y_ticks = np.arange(0, centerline_crop_len + segment_length, 1)  # Determine y-tick positions for intervals of 1000
y_tick_positions = y_ticks / segment_length
ax1.set_yticks(y_tick_positions) # Set the y-ticks and labels
ax1.set_yticklabels([f'{int(tick)}' for tick in y_ticks])
ax1.set_xticklabels([])
#colorbar
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="2%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.set_label("velocity (m/d)")


#plot 2
im2 = ax2.pcolormesh(Y, X, cl_roll_dev_transposed.values, vmin=vmin_dev, vmax=vmax_dev, cmap='bwr', shading='auto')  #bwr 
if year==2023: ax2.axvspan(start_gap, end_gap, color='lightgray', alpha=1)
ax2.set_xlim([time_values[int(first_non_nan_row)], time_values[int(last_non_nan_row)]])
# Format the y-axis
ax2.set_yticks(y_tick_positions) # Set the y-ticks and labels
ax2.set_yticklabels([f'{int(tick)}' for tick in y_ticks])
ax2.set_xticklabels([])
ax2.set_ylabel('centerline distance from front (km)')
#colorbar
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="2%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.set_label("mean velocity deviation (m/d)")


#plot 3
if year == 2023:   # Define colors for morning and evening spans based on year
    cols_evening_updown = [ 'w', 'w', 'r', 'r', '#008080', 'w', '#FF6347', 'w', '#FF6347', '#FF6347', '#FF6347', '#008080', 'w']  # Color for the rectangles
    cols_morning_updown = [ 'w', '#FF6347', '#FF6347', 'r', 'r', '#008080', 'w', 'w', '#FF6347', '#FF6347', 'w', '#FF6347', 'w']  # Color for the rectangles
    cols_morning_edgecolor = ['k', 'k', 'k', 'r', 'k', 'k', 'k', 'grey', 'k', 'k', 'k', 'k', 'k']  # Color for the rectangles
    cols_evening_edgecolor = ['r', 'k', 'r', 'k', 'k', 'k', 'k', 'grey', 'k',  'k', 'k', 'k', 'w']  # Color for the rectangles
if year == 2024: 
    cols_evening_updown = ['#008080', '#FF6347', '#FF6347', '#FF6347', '#FF6347', '#FF6347', '#FF6347', '#008080', '#008080', 'w', '#FF6347', 'w', 'w', 'w']  # Color for the rectangles
    cols_morning_updown = ['w', 'w', '#FF6347', '#008080', 'w', 'w', '#FF6347', '#FF6347', '#008080', 'w', 'w', '#FF6347', '#FF6347', 'w']  # Color for the rectangles
    cols_morning_edgecolor = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'] # Color for the rectangles
    cols_evening_edgecolor = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'w']   # Color for the rectangles
# Extract unique days as integers
unique_days = np.unique(np.floor(time_values).astype(int))
if year==2023: unique_days = np.append(unique_days, 19575); unique_days = np.sort(unique_days); #unique_days = unique_days +1
# Loop over unique days
for i, day in enumerate(unique_days):
    evening_start = day + 19 / 24  # 7 PM
    evening_end = evening_start + 5 / 24  # 5 hours span
    morning_start = day + 7 / 24  # 7 AM
    morning_end = morning_start + 5 / 24  # 5 hours span
    if day == 19575: continue
    if day == 19572: continue
    # Plot evening and morning spans 
    if cols_morning_edgecolor[i] == 'k' and cols_morning_updown[i] == 'w':
        fnc.draw_diagonal_bicolor_fill(ax3, morning_start, morning_end,  ymin=.9, ymax=1)
    if day == 19584: continue
    if cols_evening_edgecolor[i] == 'k' and cols_evening_updown[i] == 'w':
        fnc.draw_diagonal_bicolor_fill(ax3, evening_start, evening_end,  ymin=.9, ymax=1)
    if day != 19574: ax3.axvspan(evening_start, evening_end, ymin=.9, ymax=1, edgecolor=cols_evening_edgecolor[i % len(cols_evening_edgecolor)], facecolor=cols_evening_updown[i % len(cols_evening_updown)], linewidth=.5)
    if day != 19576: ax3.axvspan(morning_start, morning_end, ymin=.9, ymax=1, edgecolor=cols_morning_edgecolor[i % len(cols_morning_edgecolor)], facecolor=cols_morning_updown[i % len(cols_morning_updown)], linewidth=.5)  
ax3.set_xlim([time_values[int(first_non_nan_row)], time_values[int(last_non_nan_row)]])
#colorbar
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="2%", pad=0.05)
cax3.axis('off')  # Hide the dummy colorbar axis
#remove box and ylabel
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.set_yticks([])  # Remove y-ticks


#plot 4
im2 = ax4.pcolormesh(Y2, X2, cl_acc_new_transposed.values, vmin=vmin2, vmax=vmax2, cmap='bwr', shading='auto') 
if year==2023: ax4.axvspan(start_gap, end_gap, color='lightgray', alpha=1)
divider4 = make_axes_locatable(ax4)
cax4 = divider4.append_axes("right", size="2%", pad=0.05)
cbar4 = fig.colorbar(im2, cax=cax4, extend='both')
cbar4.set_label("acceleration (md$^{-2}$)")
# Format the x-axis to show date/times properly
ax4.xaxis_date()
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # %H:%M
ax4.set_xlim([time_values[int(first_non_nan_row)], time_values[int(last_non_nan_row)]])
#format y-axis by setting the centerline "centerline_crop" length for y axis
segment_length = centerline_crop_len / 86    # Calculate the segment length
y_ticks = np.arange(0, centerline_crop_len + segment_length, 1)  # Determine y-tick positions for intervals of 1000
y_tick_positions = y_ticks / segment_length
ax4.set_yticks(y_tick_positions) # Set the y-ticks and labels
ax4.set_yticklabels([f'{int(tick)}' for tick in y_ticks])
ax4.set_xlabel('date')


# Add vertical lines at each day start
start_of_day_values = np.unique(np.floor(time_values))
for day in start_of_day_values:
    ax1.axvline(x=day, color='grey', linestyle='-', linewidth=0.5)
    ax2.axvline(x=day, color='grey', linestyle='-', linewidth=0.5)
    ax3.axvline(x=day, color='grey', linestyle='-', linewidth=0.5)
    ax4.axvline(x=day, color='grey', linestyle='-', linewidth=0.5)
if year ==2023: 
    ax1.axvline(x=19575., color='grey', linestyle='-', linewidth=0.5) #add line on no data day
    ax2.axvline(x=19575., color='grey', linestyle='-', linewidth=0.5) #add line on no data day
    ax4.axvline(x=19575., color='grey', linestyle='-', linewidth=0.5) #add line on no data day

#add text no data
if year ==2023:
    ax1.text(19575.5, 50, 'no data',  verticalalignment='center', horizontalalignment='center', fontsize=10)
    ax2.text(19575.5, 50, 'no data',  verticalalignment='center', horizontalalignment='center', fontsize=10)
    ax4.text(19575.5, 50, 'no data',  verticalalignment='center', horizontalalignment='center', fontsize=10)

# add a) and b) in plot
ax1.text(0.01, 0.95, 'a)', transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top')
ax2.text(0.01, 0.95, 'b)', transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top')
ax3.text(0.01, 0.95, 'c)', transform=ax4.transAxes, fontsize=20, fontweight='bold', va='top')

# Add an overall ylabel
fig.text(.995, 0.5, 'smoothed LOS ... along centerline', va='center', rotation='vertical')


plt.tight_layout()
plt.subplots_adjust(hspace=0) 
plt.show()






#%% Figure 8

if year == 2024: 
    if (filter_start!=4) or (filter_end!=90): 
        raise ValueError("Error: adjust filter_start and filter_end in code section above")

    # Transpose the DataFrame to swap x and y axes
    cl_roll_new_transposed = cl_roll_new.T
    cl_acc_new_transposed = cl_acc_new.T
    cl_roll_new_transposed.index.name = 'time'  # or whatever makes sense
    cl_roll_new_transposed.reset_index(inplace=True)
    cl_acc_new_transposed.index.name = 'time'  # or whatever makes sense
    cl_acc_new_transposed.reset_index(inplace=True)

    #plot
    fig, (ax3, ax1) = plt.subplots(2, 1, figsize=(10, 7))

    #load shapefiles
    centerline=gpd.read_file(current_folder.parent / 'data/shapefiles/EKaS_radar_centerline2.shp')
    EKaS_outline = gpd.read_file(current_folder.parent / 'data/shapefiles/EKaS_terminus_outline.shp')

    #get centerline in 100 parts
    interpolated_points = [centerline.geometry.iloc[0].interpolate(distance) for distance in range(0, int(centerline.geometry.iloc[0].length), int(centerline.geometry.iloc[0].length / 100))]
    centerline_all_xcoord, centerline_all_ycoord = LineString(interpolated_points[filter_start:filter_end]).xy  #cut uppermost part
    selected_points = interpolated_points#[filter_start:filter_end]
    centerline_part_x_coords, centerline_part_y_coords = LineString(selected_points).xy

    #remove outliers/values in shadowed area: 
    time_col = cl_roll_new_transposed["time"] # Keep time separately
    numeric_cols = cl_roll_new_transposed.drop(columns=["time"]) # Work only on numeric columns
    filtered_numeric = numeric_cols.loc[:, numeric_cols.min() >= 4] # Keep only columns where the minimum value is >= 4
    cl_roll_new_transposed = pd.concat([time_col, filtered_numeric], axis=1) 
    cl_roll_new_transposed = cl_roll_new_transposed.drop(columns=['47','51', '73', '76'])  

    time_col_acc = cl_acc_new_transposed["time"] # Keep time separately
    numeric_cols_acc = cl_acc_new_transposed.drop(columns=["time"]) # Work only on numeric columns
    filtered_numeric_acc = numeric_cols_acc.loc[:, numeric_cols.min() >= 4] # Keep only columns where the minimum value is >= 4
    cl_acc_new_transposed = pd.concat([time_col_acc, filtered_numeric_acc], axis=1) 
    cl_acc_new_transposed = cl_acc_new_transposed.drop(columns=['47','51', '73', '76' ]) 


    # subplot 1: Time series of roomean
    for i, col in enumerate(cl_roll_new_transposed.columns[1:]):
        cmap = plt.cm.viridis_r((i) / (filter_end - filter_start))  # Normalize i to the range [0, 1] for colormap
        ax3.plot(cl_roll_new_transposed['time'], cl_roll_new_transposed[col], label=f'Channel {col}', color=cmap)
    ax3.set_xlim(cl_roll_new_transposed.time.iloc[0], cl_roll_new_transposed.time.iloc[-1])
    ax3.set_xticklabels([])
    ax3.set_xlabel('')
    ax3.set_ylabel('LOS velocity (md$^{-1}$)')


    # subplot 2: Time series of steepness
    for i, col in enumerate(cl_acc_new_transposed.columns[1:]):
        cmap = plt.cm.viridis_r((i) / (filter_end - filter_start))  # Normalize i to the range [0, 1] for colormap
        ax1.plot(cl_acc_new_transposed['time'], cl_acc_new_transposed[col], label=f'Channel {col}', color=cmap, zorder=2)
    ax1.hlines(0, cl_acc_new_transposed.time.iloc[0], cl_acc_new_transposed.time.iloc[-1], color='k', linestyle='--')
    ax1.set_xlim(cl_acc_new_transposed.time.iloc[0], cl_acc_new_transposed.time.iloc[-1])
    ax1.set_ylim(-16, 9.9)
    ax1.set_xlabel('date')
    ax1.set_ylabel('acceleration (md$^{-2}$)')


    #add inset map
    axins = fig.add_axes([0.26, 0.68, 0.2, 0.2])  # [left, bottom, width, height]
    EKaS_outline.plot(ax=axins, facecolor='None', edgecolor='k', linewidth=0.1)
    for i, col in enumerate(cl_acc_new_transposed.columns[1:]):
        cmap = plt.cm.viridis_r((int(col)-filter_start) / (filter_end - filter_start))  # Normalize i to the range [0, 1] for colormap
        axins.scatter(centerline_part_x_coords[int(col)], centerline_part_y_coords[int(col)], color=cmap, s=10)
    axins.set_xlim(-46500, -39900)
    axins.set_ylim(-3172000, -3165000)
    axins.set_xticklabels([])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_xlabel('')
    axins.set_ylabel('')
    axins.set_title('')
    

    # Add vertical lines at each day start
    for day in cl_acc_new_transposed.time.dt.floor('D').unique():
        ax3.axvline(x=day, color='grey', linestyle='-', linewidth=0.2)
        ax1.axvline(x=day, color='grey', linestyle='-', linewidth=0.2)
        
plt.subplots_adjust(hspace=0) 
plt.show()




 #%%  Figure 7: compare GPRI vs ITslive 

# Define start and end times for the analysis
start_time_all = pd.Timestamp("2023-03-01 15:47:00", tz='UTC')  # Start time
end_time_all = pd.Timestamp("2024-12-31 02:47:00", tz='UTC')    # End time

#load itslive data
fn = current_folder.parent / 'data/velocity/itslive_mean_centerline_timeseries_2023_2024.csv'
df_meas = pd.read_csv(fn)
df_meas['date'] = pd.to_datetime(df_meas['date'])

#load GPRI data
cl23 = pd.read_csv(current_folder.parent / 'data/velocity/EKaS23_veloc_timeseries_centerline.csv').T
cl24 = pd.read_csv(current_folder.parent / 'data/velocity/EKaS24_veloc_timeseries_centerline.csv').T
start_time_all24 = pd.Timestamp("2024-07-12 00:30:00", tz='UTC')  # Start time for 2024
end_time_all24 = pd.Timestamp("2024-07-25 18:37:00", tz='UTC')    # End time for 2024
start_time_all23 = pd.Timestamp("2023-08-03 15:47:00", tz='UTC')  # Start time for 2023
end_time_all23 = pd.Timestamp("2023-08-15 21:47:00", tz='UTC')    # End time for 2023

cl_mean23 = fnc.process_centerline_data(cl23, start_time_all23, end_time_all23)
cl_mean24 = fnc.process_centerline_data(cl24, start_time_all24, end_time_all24)


#plot
fig, ax = plt.subplots(1, figsize=(8, 4)) # create figure
ax.plot(df_meas.date, df_meas['mean_v'], label="MEaSUREs ITS_LIVE", color = "purple")
ax.plot(cl_mean23.date, cl_mean23.cl_mean, label="TRI LOS velocity", color="#008080")
ax.plot(cl_mean24.date, cl_mean24.cl_mean, color="#008080")
ax.set_xlabel("date (yyyy-mm-dd)")
ax.set_ylabel("ice velocity (m/d)")
ax.legend(ncol=3, loc="lower left")
ax.set_ylim((3.5, 12.5))

plt.tight_layout()
plt.show()

