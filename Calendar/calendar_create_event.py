#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Created on Sat Apr  3 12:22:53 2021

@author: ena
"""

# Imports
import argparse
import re
# from pathlib import Path
from datetime import timedelta, datetime
import dateparser
from googleapiclient.discovery import build

# Internal Module Function Imports
from calendar_functions import create_event, wh, now, datetime_to_list
from project_setup import creds_json

# # Parse Natural Language Time
# def parse_time(time):
#     time = str(time) # ensure string
#     hm = re.sub('a?p?m?', '', time) # hm = am/pm-agnostic time
#     out = hm.strip() if ':' in hm else hm.strip() + ':00'
#     if 'pm' in time:  # convert to 24-hour clock if needed
#         str(int(out.split(':')[0]) + 12) + ':%s'%(out.split(':')[1])
#     out = dateparser.parse(out) # parse time
#     return out

# Example command line arguments
# $ python calendar_create_event.py "Qualifying exams resesarch scripted", 3:30, -d 4-15 -t "2.5 hours"
# arguments = ['Research scripted', '3:30', '-d 4-15', '-t 2.5 hours']
# arguments = ['Research scripted', '3:30', '8', '-d 4-15']
# arguments = ['Research scripted', '-t 0.75']


# Load Credentials (pickled in project_setup.py) & Service
# SCOPE = 'https://www.googleapis.com/auth/calendar'
SCOPE = 'https://www.googleapis.com/auth/calendar.events'
SCOPES = ['https://www.googleapis.com/auth/calendar', 'https://www.googleapis.com/auth/calendar.events']
creds = creds_json()
# http_obj = httplib2.Http()
# http_obj = creds.authorize(http_obj)
service = build('calendar', 'v3', credentials=creds)  # calendar API service
# service = build('calendar', 'v3', http=http_obj)  # calendar API service


# Function
def insert_event(arg_list=None):
    """
    Insert event into calendar

    Parameters
    ----------
    arg_list : TYPE, optional
        List of strings (representing command line arguments) to pass to parser.parse_args()
        to call function directly. The default is None.

    Returns
    -------
    event_s : dict
        Event.

    """
    parser = argparse.ArgumentParser()  # parse provided string if not calling from command line
    short_opts = ['d', 't', 'c', 'e']  # short tags (-tag)
    long_opts = ['date', 'duration', 'color', 'email']  # long tags (--tag)
    defaults = [now().strftime('%m-%d-%y'), '1 hour', None, False]
    helps = ['Event date', 'Event duration (instead of 2nd positional argument end time)',
             'Event color', 'Email notification?']
    parser.add_argument('name', type=str)
    parser.add_argument('begin', nargs='?', type=str)
    parser.add_argument('end', nargs='?', type=str)
    for o in zip(short_opts, long_opts, helps, defaults):
        parser.add_argument('-%s' % o[0], '--%s' % o[1], help=o[2], default=o[3],
                            action=['store', 'store_true'][o[0] == 'e'])  # + optional arguments
    if arg_list is not None:
        args = parser.parse_args(arg_list)  # parse list of arguments
    else:
        args = parser.parse_args()  # parse command line arguments
    print('Arguments: ', args)
    # Manipulate Arguments
    color_kws = {'superpower': '9', 'meet': '11', 'research': '1'}
    if args.color is None:
        match_c = [c.lower() in args.name.lower() for c in color_kws]  # find if/where title has keyword
        if any(match_c):
            COLOR = color_kws[wh(match_c, 1, list(color_kws.keys()))]
        else:
            COLOR = '5'
    else:
        COLOR = args.color  # use specified color
    if args.begin is None:
        start = datetime.now().strftime('%T')  # default start is now
    else:
        start = args.begin if any([':' in args.begin, 'm' in args.begin]) else args.begin + ':00'
    if args.end is None:
        durtn = args.duration.strip()
        if any([x in durtn] for x in ['hours', 'minutes']):  # if duration in natural language...
            hrs = float(re.sub(r'(\d*\.?\d*) *hour.*', '\\1', durtn if 'hour' in durtn else 0))  # extract # hours
            mns = re.sub(r' *.*hour?s?', '', '1 hours 30 minutes').strip()  # remove any hour component
            mns = float(re.sub(r'(\d*\.?\d*) *minute.*', '\\1', mns)) if 'min' in args.duration else 0
            hours = hrs + mns / 60
        else:  # if specified as numerical, assume as hours...
            hours = float(durtn)
        stop = dateparser.parse(start) + timedelta(hours=hours)  # stop = start + duration
        stop = str(stop.hour) + ':' + '{:02d}'.format(stop.minute)  # make HH:MM format
    else:
        stop = args.end if any([':' in args.end, 'm' in args.end]) else args.end + ':00'
    START, STOP = [re.sub('\'*\"*', '', t) for t in [start, stop]]  # remove extraneous quotation marks
    # Create Event
    event = create_event(name=args.name, times=[START, STOP], time_zone='EST',
                         start_date=datetime_to_list(dateparser.parse(args.date)), end_date=None, color=COLOR,
                         reminder_times=['1h', '30m', '15m'] + [[], '1d'][args.email],
                         reminder_types=['popup'] * 3 + [[], 'email'][args.email],
                         recur_frequency=None, recur_interval=None,
                         recur_byday=None, recur_until=None)  # create event list entry (dictionary)
    [print(k + ': ' + str(event[k])) for k in event]
    create_ans = input('\n\nConfirm event creation (yes/y)? ')
    if create_ans.strip().lower() in ['yes', 'y']:  # if confirmed, create
        print('\nCreating event\n\n')
        event_s = service.events().insert(calendarId='primary', body=event).execute()
        # event_s = service.calendars().insert(event).execute()
    else:
        print('\n\nEvent not created.')
    return event_s


# Run Function
insert_event()
