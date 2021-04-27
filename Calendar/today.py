#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 07:58:20 2021

@author: ena
"""

# Imports
import functools
import os
import os.path
# from pathlib import Path
import shutil
import pickle
from datetime import datetime
import calendar
import sys
# import psutil
# import re
import numpy as np
import pandas as pd
from calendar_functions import Calendar  # internal module with function & Calendar class
from calendar_functions import datetime_to_list  # internal fx to turn datetime variable to YMD list
from calendar_functions import now  # internal fx to get current time
from calendar_functions import wh  # internal fx to search lists
from project_setup import make_save_creds  # get service & credentials, write today.sh here & to home
from project_setup import locate_pickles, direct  # get credential pickled file names, cleaned directory path
# import googleapiclient


# Interpret Arguments & Retrieve Calendar
# print(psutil.Process().parent().cmdline())
today = datetime_to_list(now())  # [<year>, <month>, <day>] of today
pkls = locate_pickles()  # get file names for pickled credentials written in project_setup.py
directory = direct()
if not os.path.exists(pkls[0]):
    print('***** Credentials not found...Creating using project_setup.py function setup()...')
    make_save_creds()  # run setup if can't find pickled credentials
print('\n')
if len(sys.argv) == 1:  # default arguments if only 1 fed by system (because 1st = this file's name)
    begin, kws = datetime_to_list(now()), None
    os.system('echo "Running with default arguments..."')
else:  # pull command line arguments (from running bash file, today.sh)
    # today (day to retrieve, unless modified below because of argument 2 == 'tomorrow')
    begin = today
    if sys.argv[1].lower() == 'tomorrow':
        begin = [begin[0], begin[1], begin[2] + 1]
    elif sys.argv[1].lower() == 'yesterday':
        begin = [begin[0], begin[1], begin[2] - 1]
    kws = None if len(sys.argv) == 2 else sys.argv[2].strip().split(
        ' ')  # keywords for event titles (or all if blank)
    os.system('echo "Running with argument(s): %s..."' %
              (functools.reduce(lambda i, j: '%s %s' % (i, j), sys.argv[1:])))
end = [begin[0], begin[1], begin[2] + 1]
os.system('echo "Creating calendar..."')
cal = Calendar(**dict(start=[today[0], today[1], 1],
                      stop=[today[0], today[1], calendar.monthrange(*begin[:2])[1]],
                      credentials_file=pkls[0], keyword_list=kws))  # class (calendar_functions.py)

# Display Today's Events
os.system('echo "Printing calendar..."')
df = cal.to_frame(print_df=False)  # return events as dataframe & print
day = 'today' if len(sys.argv) == 1 else sys.argv[1].lower()  # day header
ixs = [functools.reduce(lambda i, j: [i + [j]], [begin[:(x + 1)]]) if x > 0 else begin[x] for x in range(len(begin))]
i_na = [ixs[x] not in [list(i[:(x + 1)]) if x > 0 else i[0]
                       for i in cal.data.index.values] for x in range(len(begin))]
if any(i_na):  # if line above detected no events for year, year-month, or year-month-day...
    print('No events for this %s' % str(wh(i_na, 1, ['year', 'month', 'day'])))
else:
    data = cal.data.loc[begin[0]].loc[begin[1]].loc[begin[2]].reset_index()
    df = data.assign(Start=data.apply(lambda x: '%s:%s' %
                                      (str(x.hour), '{:02d}'.format(x.minute)), axis=1))  # HH:MM
    df = df.assign(End=df.apply(lambda x: '%s:%s' %
                                (str(x.end_list[-2]), '{:02d}'.format(x.end_list[-1])),
                                axis=1))  # HH:MM end
    df = df.set_index(['Start', 'End'])[cal.data.index.names[-1]
                                        ].to_string()  # subset & to string for printing
    col_n = shutil.get_terminal_size().columns
    center = int(col_n / 2)  # ~ center of terminal
    header = '\n%s\n%s%s\n%s\n\n' % (
        '=' * col_n, ' ' * (center - int(len(day.upper()) / 2)), day.upper(), '=' * col_n)  # header
    print(header, df, '\n')  # header & df
    data.to_csv('%s.csv' % day.lower())  # save as CSV

# Warn if Event Soon
if (day.lower() == 'today') and (any(i_na) is False):  # if asking about today
    until = data.start_list.apply(
        lambda i: [(x.seconds / 3600, x.seconds / 60) if x.seconds <= 0 else
                   np.nan for x in [datetime(*i) - datetime.now()]])
    if all(until.apply(lambda x: all(pd.isnull(x)))):  # if no more events today...
        print('\n\nNo more events today! (I think. Check to be sure.)')
    else:
        print()
