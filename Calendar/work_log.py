#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:31:32 2021

@author: ena
"""

# Imports
import functools
import warnings
import os
import datetime
import dateparser
import pandas as pd

# Options
pd.options.display.max_columns = 100
pd.options.display.max_rows = 200

# Set Up
# directory = pathlib.Path(__file__).parent.absolute()
# os.system('cd')
# os.system('cd {directory}')
cat_codes = {'r': 'Research', 'rq': 'Qualifying exams research', 's': 'Superpower',
             'c': 'MMC DLS Consult Research', 'a': 'Administrative', 'm': 'Meeting', 'w': 'work'}
prompts = functools.reduce(lambda i, j: i + '\n' + j, [f'{i}: {cat_codes[i]}' for i in cat_codes])
work_categories = {}  # initialize empty dictionary of entries for today
WORKING = True  # start off still working
now = datetime.datetime.now()  # datetime of right now
year, month, day = now.year, now.month, now.day
print(f'\n\n{"=" * 80}{" " * 40}\nWork Log: {month}-{day}-{year}\n{"=" * 80}\n\n')
file = input('Where to save (<input>.csv or blank for work_<date>.csv): ')
file = f'work_{year}_{month}_{day}.csv' if file.strip() == '' else file + '.csv'
CONTINUING, count = False, 0  # signal that are just starting out until file 1st written


def dt_hm(date_time):
    """Convert datetime variable to HH:MM format"""
    return date_time.strftime('%H:%M')


def dt_now(fmt=False):
    """Return datetime object for right now"""
    out = datetime.datetime.now()  # datetime variable
    if fmt:
        out = out.strftime('%H:%M')  # HH:MM format
    return out


def end_or_pause(prompts='', cat_codes=None, times=None, resume=False, event=None):
    """Recursive function to track times when pausing as needed"""
    now = dt_now()
    if cat_codes is None:
        cat_codes = {}
    if times is None:
        times = []
    if event is None:
        category = input(f'What are you working on?\n{prompts}\nor custom event name\n\n')  # event
        event = cat_codes[category] if category in cat_codes.keys() else category  # convert
    term = ['', 'done', 'pause']
    vrb, ger = ['RESUME' if resume else 'start', 'Resuming' if resume else 'Starting']
    start_sig = input(f'Press <enter> to {vrb} now or custom start time (with am/pm or 24-hour): ')
    st_v = now if start_sig.strip() == '' else dateparser.parse(start_sig)  # start
    print(f'***** {ger} {event} at {dt_hm(st_v)}...')
    opts = '"pause" to toll clock, custom end time, or done to end day'
    end_sig = input(f'Press <enter> to end now, {opts}: ')
    end_v = dt_now() if end_sig.strip() in term else dateparser.parse(end_sig)  # end
    if end_v is None:
        ValueError('Either something went wrong internally, or dateparser failed.')
    times = times + [(dt_now(st_v), dt_now(end_v))]  # formatted start & end
    while end_sig.strip().lower() == 'pause':  # while pausing
        print(f'Pausing at {dt_hm(end_v)}...')
        times, event, end_sig = end_or_pause(times=times, resume=True, event=event)
    print(f'***** Ending {event} at {dt_hm(end_v)}...')
    return times, event, end_sig


# Re-Load File (if exists)
if os.path.exists(file):  # if file exists
    pre_exist = pd.read_csv(file, index_col=['Start', 'End'])  # retrive day's entries in file
    if CONTINUING is False:
        write_opt = input('Overwrite file (o), append (a), or preview then decide (p)? ')
        if write_opt.strip().lower() == 'p':  # to preview pre-existing data 1st
            print(pre_exist)  # preview...
            write_opt = input('\nOverwrite file (o) or append (a)? ')  # then prompt again...
        if write_opt.strip().lower() == 'o':  # to overwrite pre-existing
            print(f'***** Will overwrite {file}')  # over-writing...
            so_far = pd.DataFrame(columns=['Start', 'End', 'Event'])  # ...so start empty
        elif write_opt.strip().lower() == 'a':  # to append to pre-existing
            so_far = pre_exist
            print(f'***** Will to append to {file}')
        else:
            raise ValueError('Must choose o or a.')
    else:
        so_far = pre_exist  # use from reading .csv if continuing loop
elif CONTINUING:  # if continuing, but didn't find file, throw error
    raise FileNotFoundError('File not found even though \'CONTINUING\' signal is True.')
# Entries Input
cal, end_sig = {}, ''
while end_sig.strip().lower() != 'done':
    times, event, end_sig = end_or_pause(prompts=prompts, cat_codes=cat_codes)
    cal.update({event: times})
events_all = functools.reduce(lambda i, j: i + j, [[k] * len(cal[k]) for k in cal])
times_all = functools.reduce(lambda i, j: i + j, [cal[k] for k in cal])
cal_df = pd.Series(events_all, index=times_all).to_frame(name='Event')
cal_df = cal_df.rename_axis(['Start', 'End']).rename_axis(['Start', 'End'])

# Read & Print File
if os.path.exists(file):  # if file exists
    df_final = pd.read_csv(file, index_col=['Start', 'End'])
    print(df_final)
else:
    warnings.warn(f'{file} not written.')
