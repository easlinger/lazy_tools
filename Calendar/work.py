#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:11:19 2021

@author: ena
"""

# Imports
import os
import os.path
import dateparser
from datetime import timedelta as timedelta
import sys
from calendar_functions import Calendar  # internal module with function & Calendar class
from calendar_functions import datetime_to_list  # internal fx to turn datetime variable to YMD list
from calendar_functions import now  # internal fx to get current time
from project_setup import make_save_creds  # get service & credentials, write today.sh here & to home
from project_setup import locate_pickles, direct  # get credential pickled file names, cleaned directory path

# Set Up
print('\n')
os.system('echo "***** Change 1st line after imports & print/echo commands in work.py to change work keywords."')
kws = ['research', 'work', 'administrative', 'meeting', 'superpower']

# Get Credentials
pkls = locate_pickles()  # get file names for pickled credentials written in project_setup.py
directory = direct()
if not os.path.exists(pkls[0]):
    os.system('echo ***** Credentials not found...Creating using project_setup.py function setup()...')
    make_save_creds()  # run setup if can't find pickled credentials

# Retrieve Arguments
today = now()  # [<year>, <month>, <day>] of today
monday = dateparser.parse('monday')  # Monday of this week
last_sunday = dateparser.parse('sunday')  # Sunday of last week
sunday = monday + timedelta(days=7)  # upcoming Sunday
last_monday = monday + timedelta(days=-7)  # last Monday
if len(sys.argv) == 1:  # default arguments if only 1 fed by system (because 1st = this file's name)
    begin, end = [datetime_to_list(x) for x in [monday, today]]
    os.system('echo "Running with default arguments (this week until today)..."')
else:
    begin = datetime_to_list(last_monday) if sys.argv[1].lower() == 'last' else monday
    end = datetime_to_list(last_sunday) if sys.argv[1].lower() == 'last' else today

# Retrieve & Analyze Calendar
cal = Calendar(**dict(start=begin, stop=end, credentials_file=pkls[0]))
cal.to_frame(keywords=kws, print_df=False)
hours = cal.data.hours.sum()  # number of hours worked this/last week
brkdwn = dict([(k, cal.data.reset_index().apply(lambda x: x.hours if k.lower() in x.Event.lower() else 0, axis=1).sum())
               for k in kws])  # hours in each category
print('\n')

os.system('echo ' + '=' * 80)
os.system('echo ' + f'{hours} hours worked')
os.system('echo ' + '=' * 80)
for k in brkdwn:
    os.system(f'echo {k}: {brkdwn[k]}')
print('\n')
