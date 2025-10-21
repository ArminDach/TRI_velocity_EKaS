#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 10:25:52 2025

@author: Armin Dachauer
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

def butter_lowpass_filter(data, axis=1, cutoff_hours=3, fs=1.0, order=4):
    """
    Applies a Butterworth low-pass filter to a 1D signal.
    Parameters:
    data : array-like or Series
        Input signal.
    cutoff_hours : float
        The cutoff period in hours (frequencies with periods shorter than this will be removed).
    fs : float
        Sampling frequency in Hz (e.g., 1 per hour = 1.0).
    order : int
        Order of the filter.
    Returns:
    y : array-like or Series
        Filtered signal.
    """
    # Compute the mean across the specified axis and multiply by -1
    mean = data.mean(axis=axis) 
    
    # Apply the Butterworth low-pass filter
    nyq = 0.5 * fs
    cutoff_freq = 1.0 / cutoff_hours
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, mean)
    
    #add cl.index to the filtered signal
    y = pd.Series(y, index=data.index)
    return y



def interpolate_time_and_fill_gaps(data, time, freq):
    """
    Interpolate data over evenly spaced time intervals and remove gaps.

    Parameters
    ----------
    data : array-like
        Input data values (may contain NaNs).
    time : array-like
        Corresponding timestamps for the data values.
    freq : str
        Desired output frequency (e.g., '1H', '30min').
    Returns
    -------
    data_interp : array-like
        Interpolated data values at evenly spaced time intervals.
    new_time : array-like
        Evenly spaced timestamps corresponding to the interpolated data.
    """
    # Convert time and data to pandas Series
    time = pd.to_datetime(pd.Series(time))  # Ensuring the time array is in datetime format
    data = pd.Series(data)
    
    # Remove leading NaNs from both data and time

    valid_mask = ~data.isna()  # Boolean mask to remove NaNs
    data = data[valid_mask]
    time = time[valid_mask]
    

    # Create a new time index with evenly spaced intervals
    new_time = pd.date_range(start=time.min(), end=time.max(), freq=freq)

    # Reindex the data based on the new evenly spaced time and interpolate NaN values
    data_interp = data.reindex(new_time)

    return data_interp.to_numpy(), new_time.to_numpy()



def calc_arrow(x, y, length, angle, head_length):
    """
    Calculate the coordinates for drawing centered arrows based on direction and length.

    Parameters
    ----------
    x : array-like
        X coordinates of the arrow centers.
    y : array-like
        Y coordinates of the arrow centers.
    length : float or array-like
        Length of the arrow shaft.
    angle : float or array-like
        Angle of the arrow (in radians).
    head_length : float
        Length of the arrowhead.
        
    Returns
    -------
    x2, y2, dx, dy : tuple of arrays
        Starting coordinates (x2, y2) and deltas (dx, dy) for the arrows.
    """
    # Calculate the half of the total arrow length (for centering)
    half_total_length = (length+head_length) / 2
    # Calculate the starting point of the tail (x2, y2)
    x2 = x - half_total_length * np.cos(angle)
    y2 = y - half_total_length * np.sin(angle)
    # Calculate the length the arrow should extend forward (shaft + arrowhead)
    dx = (length)  * np.cos(angle)
    dy = (length)  * np.sin(angle)
    return x2, y2, dx, dy


def draw_diagonal_bicolor_fill(ax, start_time, end_time, ymin=0, ymax=0.9,
                                stripe_width=0.03, color1='#FF6347', color2='#008080',
                                angle=0):
    """
    Draw diagonal stripes between start_time and end_time on a given axis.
    
    Parameters:
    - ax: matplotlib axis to draw on
    - start_time, end_time: x-limits in data coordinates
    - ymin, ymax: y-range in axis fraction (0–1)
    - stripe_width: width of each stripe in x-axis units
    - color1, color2: alternating fill colors
    - angle: angle of the diagonal stripes in degrees
    
    Returns:
    None
    """
    trans = ax.get_xaxis_transform()
    height = ymax - ymin
    num_stripes = int((end_time - start_time) / stripe_width) + 1

    for i in range(num_stripes):
        x = start_time + i * stripe_width
        color = color1 if i % 2 == 0 else color2
        
        # Build the transformation: rotate about the center of the stripe
        rot = Affine2D().rotate_deg_around(x + stripe_width / 2, ymin + height / 2, angle)
        t = rot + trans

        rect = Rectangle(
            (x, ymin),
            stripe_width,
            height,
            transform=t,
            facecolor=color,
            edgecolor='none',
            linewidth=0,
            zorder=2
        )
        ax.add_patch(rect)
        
def process_centerline_data(cl, start_time_all, end_time_all, filter_start=7, filter_end=30):
    """     
    Parameters
    ----------
    cl : pd.DataFrame
        Raw centerline data.
    start_time_all : datetime-like
        Start of the time range for filtering.
    end_time_all : datetime-like
        End of the time range for filtering.
    filter_start : int, optional
        Starting index of the centerline portion to include (default=7).
    filter_end : int, optional
        Ending index of the centerline portion to include (default=30).
        
    Returns
    -------
    cl_mean : pd.DataFrame
        Processed mean centerline data with columns 'cl_mean' and 'date'.
    """
    cl.columns = cl.iloc[0]  # Set the first row as the header
    cl = cl.drop(cl.index[0])  # Drop the first row as it's now the header
    # Filter by time
    cl.columns = pd.to_datetime(cl.columns)  # Convert column names to datetime objects
    cl_filtered_columns = cl.columns[(cl.columns >= start_time_all) & (cl.columns <= end_time_all)]
    cl = cl[cl_filtered_columns]
    # Filter by part of centerline
    cl = cl.iloc[filter_start:filter_end]
    # Calculate mean of centerline
    cl_mean = cl.mean() * (-1)
    cl_mean = pd.DataFrame(cl_mean, columns=["cl_mean"])
    cl_mean["date"] = cl_mean.index
    return cl_mean

