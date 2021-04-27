#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:55:11 2021

@author: ena
"""

# %% Imports

# Basics
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import functools
import numpy as np
import pandas as pd
import scipy as sp
import re
import os
import math
import warnings
import pickle
from pathlib import Path
from google.oauth2.credentials import Credentials


# Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sb

# Calendars, Dates, & Times
from dateutil import tz
from datetime import datetime, timedelta
import calendar
import dateparser


def week_of_month(datetime=None, year=None, month=None, day=None):
    """Get week of month: Modified https://stackoverflow.com/questions/3806473/python-week-number-of-the-month"""
    if datetime is not None:  # if using datetime object instead of ymd (my modification)
        year, month, day = datetime.year, datetime.month, datetime.day
    x = np.array(calendar.monthcalendar(year, month))
    week_of_month = np.where(x == day)[0][0] + 1
    return week_of_month


# Google
# import com.google.api.services.calendar.Calendar


# %% Basic Custom Functions

def square_grid(num_subplots):
    """Calculate number of rows & columns needed for subplots to be close to square"""
    rows = int(np.sqrt(num_subplots))  # number of rows (try to get close to square grid)
    cols = math.ceil(num_subplots/rows)  # number of columns
    return rows, cols


def wh(x, y, z=0, q='all', search=False):
    """Search x for match with y (return corresponding z element(s) (if z!=0))"""
    if type(z) == pd.core.indexes.base.Index:
        z = list(z)
    if search == True:
        out = np.where([y in i for i in x])[0]
    else:
        out = np.where([i == y for i in x])[0]
    if len(out) == 0:
        out = np.nan
    else:
        out = [z[int(f)] for f in out] if z != 0 else out
        out = out[0] if len(out) == 1 else out
    return out if (q.lower() == 'all') or (type(out) is not list) else out[0]


def now():
    """Current date, time (timezone-aware)"""
    return datetime.now(tz=tz.tzlocal())


def datetime_to_list(datetime_obj):
    """Turn a datetime object into a list [yyyy, m(m), d(d)]"""
    return [datetime_obj.year, datetime_obj.month, datetime_obj.day]


def get_calendar(cal_id='primary', credentials=None, time_zone='EST',
                 start=None, stop=None, max_events=100000,
                 only_with_summary=True, keyword=None,
                 credentials_pkl_file='~/token.pkl'):
    """
    Retrieve Google calendar (list of dictionaries ~ event)

    Parameters
    ----------
    cal_id : str, optional
        ID of calendar with which to work. The default is 'primary'.
    credentials : google.oauth2.credentials.Credentials, optional
        Google credentials. The default is None.
    time_zone : str, optional
        Time zone. The default is 'EST'.
    start : list, optional
        Retrieve calendar starting at [<year>, <month>, <day>]. The default is None.
    stop : list, optional
        Retrieve calendar ending at [<year>, <month>, <day>]. The default is None.
    max_events : int, optional
        Maximum events to retrieve from calendar. The default is 10000.
    only_with_summary : bool, optional
        Only retrieve event summaries? The default is True.
    keyword : str, optional
        Keyword by which to subset events by title ("summary"). The default is None.
    credentials_pkl_file : str, optional
        Path to file with pickled credentials. The default is '~/token.pkl'.

    Returns
    -------
    cal : TYPE
        DESCRIPTION.

    """
    if start is None:
        if start is None:
            begin = datetime_to_list(now())
        if stop is None:
            end = datetime_to_list(now())
    else:
        end = now().isoformat() if stop is None else datetime(*stop, tzinfo=tz.gettz(time_zone)).isoformat()
        begin = datetime(*start, tzinfo=tz.gettz(time_zone)).isoformat()
    if credentials is None:
        try:
            credentials = pickle.load(open(credentials_pkl_file, 'rb'))
        except:
            try:
                credentials = pickle.load(open(credentials_pkl_file, 'rb'))
            except:
                try:
                    credentials = pickle.load(open('././%s' % credentials_pkl_file, 'rb'))
                except:
                    try:
                        credentials = pickle.load(open('./././%s' % credentials_pkl_file, 'rb'))
                    except Exception as err:
                        print(err, '\n\nCould not load pkl file for Gmail credentials')
                        credentials = None
    service = build('calendar', 'v3', credentials=credentials)
    print(f'\n\nRetrieving (up to {max_events}) events between {start} & {stop}.')
    cal = service.events().list(calendarId=cal_id, maxResults=max_events,
                                timeMin=begin, timeMax=end).execute()
    cal = cal['items']
    if only_with_summary == True:
        cal = wh(['summary' in c.keys() for c in cal], True, cal)  # only events with titles
    if keyword is not None:
        cal = wh([keyword in c['summary'] if 'summary' in c.keys() else False for c in cal], True, cal)
    if type(cal is not list):
        cal = [cal]  # make list if not already
    return cal


# %% Calendar Custom Functions

def create_event(name, times, start_date=None, end_date=None, year=datetime.now().year, time_zone='-05:00',
                 color='3', reminder_times=None, reminder_types=None,
                 recur_frequency=None, recur_n=0, recur_interval=1, recur_byday=None, recur_until=None,
                 me_dict={'email': 'elizabeth.aslinger@aya.yale.edu',
                          'displayName': 'Elizabeth Aslinger', 'self': True}):
    """

    Create a Google Calendar Event

    Parameters
    ----------
    times : list of length 2, str
        ['start time', 'end time'] in format '(h)h:mm'
    start_date : str, list of int [yyyy, (m)m, (d)d], or other format parseable by dateparser.parse
        Start date for event.
    end_date : str, list of int [yyyy, m(m), d(d)], or other format parseable by dateparser.parse
        End date for event.
    time_zone : str, optional
        Time zone in the hours offset or string (e.g. 'EST') format. The default is '-05:00'.
    year : int, optional
        Year in which event occurs. The default is datetime.now().year.
    me_dict: dict, optional
        Dictionary with possible keys 'email', 'displayName', and 'self' (items would be str, str, bool, respectively)

    Returns
    -------
    None.

    Examples
    ________
    name = 'Lab Meeting' # name of event
    times = ['2:45', '05:00']
    start_date = [2020, 12, 2] # also could do '12/2/2020', 'December 2, 2020', etc.
        Can also use natural language (e.g. 'yesterday', '2 hours ago')
    end_date = [2020, 12, 3] # also could do '12/3/2020', 'December 3, 2020', etc.
    time_zone = 'EST' # also could do '-05:00'
    recur_frequency = 'MONTHLY'
    recur_interval = 2 # every 2 <recur_frequency>s (with above, every 2 months)
    recur_byday = 'TU' # (with above, every 2 months) on Tuesday
    # recur_n = 5 # recurrence happens 5 times (don't specify at same time as "recur_until")
    recur_until = datetime(2020, 12, 25) # recur until Christmas 2020
    reminder_times = ['1w', '1d', '1h', '15m']
    reminder_types = ['email', 'popup', 'popup', 'popup']

    """

    if start_date is None:
        start_date = datetime_to_list(now())  # assume starts today if not otherwise specified
    if end_date is None:
        end_date = start_date  # assume ends on same day if not otherwise specified
    dates = ['%s-%s-%s'%tuple([str(s) for s in t]) if type(t) is list else t for t in [start_date, end_date]]
    se_times = [dateparser.parse(str(t[0]) + ' ' + str(t[1]) + ' ' + time_zone) for t in zip(dates, times)]
    se_times = [t.strftime('%y-%m-%dT%T%z') for t in se_times]
    print(se_times)
    if reminder_times is None:
        remind = {'useDefault': True}
    else:
        if reminder_types is None:
            reminder_types = ['popup']*len(reminder_times)  # default to pop-up reminders
        # time scales to convert to minutes
        scales = dict(zip(['w', 'd', 'h', 'm'], [7*24*60, 24*60, 60, 1]))
        time_nums = [float(re.sub('(\d+).*', r'\1', t))
                     for t in reminder_times]  # numerical (unit-less) reminder times
        time_units = [str(re.sub('\d+([w, d, h, m])', r'\1', t))
                      for t in reminder_times]  # weeks, days, hours, or minutes
        remind_minutes = [t[0]*scales[t[1]]
                          for t in zip(time_nums, time_units)]  # convert reminder times to minutes
        # start dictionary with signalling not to use default reminders
        remind = {'useDefault': False}
        overrides = []  # empty list to hold dictionaries with reminder
        for r in range(len(remind_minutes)):
            # add reminder time & type
            overrides = overrides + [{'method': reminder_types[r], 'minutes': remind_minutes[r]}]
        remind.update({'overrides': overrides})  # add overrides list to reminders dictionary
    now_strf = datetime.now().strftime('%y-%m-%dT%T%z') + '.000Z'
    event = {'kind': 'calendar#event', 'status': 'confirmed', 'creator': me_dict, 'organizer': me_dict,
             'created': now_strf, 'updated': now_strf,
             'summary': name, 'colorId': str(color),
             **dict(zip(['start', 'end'], [{'dateTime': t} for t in se_times])),
             'reminders': remind}
    if recur_frequency is not None:
        rec_str = []
        if recur_frequency not in [0, None]:
            rec_str = rec_str + ['RRULE:FREQ=%s' % recur_frequency]
        if recur_byday not in [0, None]:
            rec_str = rec_str + ['BYDAY=']
        if recur_n not in [0, None]:
            rec_str = rec_str + ['COUNT=%s' % recur_n]  # if limited # of occurrences...
        if recur_until not in [0, None]:
            rec_str = rec_str + ['UNTIL=%s' % recur_until]  # if end date for occurrences...
        event.update({'recurrence': functools.reduce(lambda i, j: i + ';' + j, rec_str)})
    event.update({'accessRole': 'owner'})
    return event


# %% Time Sheet Functions

def ymd_from_datetime(dt=datetime.now()):
    """Retrieve [<year>, <month>, <day>] list from datetime object"""
    return [dt.date().year, dt.date().month, dt.now().date().day]


def event_key_subset(cal, key, cols):
    """Subset event keys"""
    if type(cols) is not list:
        cols = [cols]
    out = [dict(zip(e.keys(), [e[k] if type(e[k]) is not dict
                               else e[k][key] if (k in cols) & (key in e[k].keys())
                               else e[k] for k in e.keys()])) for e in cal]
    return out


# find events whose titles contain a keyword/string
def events_by_keyword(keyword, cal=None, cal_kws=None):
    if cal is None:
        cal = get_calendar(**cal_kws)  # get calendar if not provided
    is_there = [keyword.lower() in c['summary'].lower(
    ) if 'summary' in c.keys() else False for c in cal]
    # return event (or None if none fits keyword)
    return wh(is_there, True, cal) if any(is_there) else None


def start_to_end(keyword, cal=None, cal_kws=None):
    """List of lists - start & end for events with keyword in title"""
    if cal is None:
        cal = get_calendar(**cal_kws)  # get calendar if not provided
    events = events_by_keyword(keyword, cal)  # subset by events with keyword in title
    if events is None:
        df = None
    else:
        s = [pd.DataFrame([e['start']['dateTime'], e['end']['dateTime']]).T for e in events]  # start/end date-time
        s = pd.concat(s).rename({0: 'Start', 1: 'End'}, axis=1).set_index('Start').sort_index().reset_index()
        s = s.assign(Start_Date=s.apply(lambda x: dateparser.parse(x['Start']).date(), axis=1))  # parse start date
        s = s.assign(End_Date=s.apply(lambda x: dateparser.parse(x['End']).date(), axis=1))  # parse end date
        s = s.assign(Start_Time=s.apply(lambda x: dateparser.parse(x['Start']).timetz(), axis=1))  # parse start time
        s = s.assign(End_Time=s.apply(lambda x: dateparser.parse(x['End']).timetz(), axis=1))  # parse end time
        s = s.assign(Duration=s.apply(lambda x: dateparser.parse(x.End) - dateparser.parse(x.Start), axis=1))
        s = s.assign(Hours=s.Duration.apply(
            lambda x: x.components.days * 24 + x.components.hours + x.components.minutes / 60))
        summ = [pd.DataFrame([e['start']['dateTime'], e['summary']]).T for e in events]  # event name
        summ = pd.concat(summ).rename({0: 'Start', 1: 'Description'}, axis=1).set_index('Start').sort_index()
        df = s.join(summ, on='Start')  # join summary & start/end df by start date-time
    return df


def duration_by_keyword(keyword, cal=None, total=False, **kwargs):
    """Get total time spent on events with given keyword"""
    if cal is None:
        cal = get_calendar(**kwargs)  # get calendar if not provided
    events = events_by_keyword(keyword, cal)  # subset by events with keyword in title
    diff = [dateparser.parse(e['end']['dateTime']) - dateparser.parse(e['start']['dateTime']) for e in events]
    out = sum(diff) if total else diff  # return sum of all or duration for each
    return out


def week_numbers(sheet):
    """Get week numbers in a sheet"""
    if 'Week' not in sheet.columns:
        sheet = sheet.assign(Week=np.nan)  # allocate empty column for weeks #s (if needed)
    sheet = sheet.assign(Month_Week=sheet.apply(
        lambda x: week_of_month(x.Date), axis=1))  # week of month
    # 1st month-week = 1 (even if not 1st week of month)
    sheet.loc[sheet.Month_Number == 1, 'Month_Week'] = 1
    sheet = sheet.assign(Month_Number=sheet.apply(lambda x: x.Date.month - sheet.iloc[0, :]['Date'].month + 1,
                                                  axis=1))
    m_shift = sheet.loc[sheet.Date.apply(
        lambda x: x.year != sheet.iloc[0, :]['Date'].year), 'Month_Number'] + 12
    # which rows are year after 1st row
    next_yr = sheet.Date.apply(lambda x: x.year != sheet.iloc[0, :]['Date'].year)
    if sum(next_yr) > 1:
        sheet.loc[next_yr, 'Month_Number'] = m_shift  # shift month # forward by 12 if next year
    # assign week (month-weeks*months)
    sheet = sheet.assign(Week=(sheet.Month_Week)*sheet.Month_Number)
    return sheet


def create_time_sheet(cal=None, keyword='', file=None,
                      weekday_names=['Monday', 'Tuesday', 'Wednesday',
                                     'Thursday', 'Friday', 'Saturday', 'Sunday'],
                      hours_owed_weekly=None, holiday_hours=0,
                      start_date=None, end_date=None, max_events=100000, **kwargs):
    """
    Make a time sheet

    Parameters
    ----------
    cal : list, optional
        List of dictionary entries retrieved from the Google calendar. The default is None (will retrieve calendar).
    keyword : str, optional
        Keyword to subset calendar by event tile ("summary"). The default is ''.
    file : str, optional
        Path to file to be written by the function (or None to eschew writing). The default is None.
    weekday_names : list, optional
        Names of days. The default is ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].
    hours_owed_weekly : float, optional
        Number of hours to which you've committed for the activity/ies. The default is None.
    holiday_hours : float, optional
        Number of hours to subtract from total hours owed for the start to end date period. The default is 0.
    start_date : list, optional
        Starting at [<year>, <month>, <day>]. The default is None.
    end_date : list, optional
        Ending at [<year>, <month>, <day>]. The default is None.
    max_events : int, optional
        Maximum events to retrieve from calendar. The default is 100000.
    **kwargs : dict, optional
        Keyword arguments to pass to get_calendar.

    Returns
    -------
    sheet : pandas DataFrame
        Dataframe with start & end time, durations, etc. for time sheets.

    """
    if start_date is None:
        start_date = [datetime.now().date().year, 1, 1]  # default start at beginning of year
    if end_date is None:
        end_date = ymd_from_datetime()  # default end time = now
    if cal is None:  # get calendar if not provided
        cal = get_calendar(start=start_date, stop=end_date,
                           max_events=max_events, **kwargs)  # get calendar
    events = events_by_keyword(keyword, cal=cal)
    s = start_to_end(keyword, cal=cal)
    s = s.assign(Day=pd.DataFrame(s).apply(
        lambda x: weekday_names[x['Start_Date'].weekday()], axis=1))  # weekday
    s = s.assign(Year_Week=s.apply(lambda x: dateparser.parse(
        x['Start']).isocalendar()[1], axis=1))  # week of year
    s = s.assign(Week=s.Year_Week - min(s['Year_Week']))  # week of time sheet
    wk_hrs = pd.DataFrame(s.groupby('Year_Week').apply(lambda x: sum(x['Hours']))).rename({0: 'Weekly_Hours'},
                                                                                          axis=1)
    if hours_owed_weekly is not None:
        wk_hrs = wk_hrs.assign(Banked_Weekly=wk_hrs.apply(lambda x: x - hours_owed_weekly))
        s = s.join(wk_hrs, on='Year_Week')  # hours & hours banked
        s = s.groupby('Week').apply(lambda x: x.assign(
            Banked=[np.nan]*(x.shape[0] - 1) + [x.Banked_Weekly.iloc[0]]))
    else:
        s = s.join(wk_hrs, on='Year_Week')  # hours & hours banked
    s = s.reset_index(drop=True)
    s = s.groupby('Year_Week').apply(
        lambda x: x.assign(Week_Hours=[np.nan]*(x.shape[0] - 1) + [x.Weekly_Hours.iloc[0]]))
    sheet = s.set_index(['Year_Week', 'Day'])  # week # & day as indices
    if file is not None:
        sheet.to_csv(file, index=['Week', 'Day'])  # save (if file name specified)
    print('Total Hours = %d' % sum(sheet.Hours.dropna()))
    return sheet


# %% Classes

class Calendar():

    # Initalize
    def __init__(self, cal_id='primary', credentials=None, time_zone='EST',
                 start=None, stop=None, max_events=10000,
                 only_with_summary=True, keyword_list=None, SCOPES=None,
                 reauthenticate=False, credentials_file=None, json_file=None, secrets_file=None, port=0):
        """
        Initialize Calendar object with methods to retrieve, manipulate, display, & summarize Google calendar events.

        Parameters
        ----------
        cal_id : str, optional
            ID of calendar with which to work. The default is 'primary'.
        credentials : google.oauth2.credentials.Credentials, optional
            Google credentials. The default is None.
        time_zone : str, optional
            Time zone. The default is 'EST'.
        start : list, optional
            Retrieve calendar starting at [<year>, <month>, <day>]. The default is None.
        stop : list, optional
            Retrieve calendar ending at [<year>, <month>, <day>]. The default is None.
        max_events : int, optional
            Maximum events to retrieve from calendar. The default is 10000.
        only_with_summary : bool, optional
            Only retrieve event summaries? The default is True.
        keyword_list : list, optional
            Keywords to subset events by title ("summary"). The default is None.
        reauthenticate : bool, optional
            Re-authenticate Google? The default is False.
        credentials_file : str, optional
            Path to file with pickled credentials. The default is None (-> 'token.pkl' in the home directory).
        secrets_file : str, optional
            Path to file with secrets for Google credentials. The default is None.
        port : int, optional
            Port for local server (if need to re-run Google API authentication). The default is 0.

        Returns
        -------
        None.

        """
        if SCOPES is None:
            SCOPES = ['https://www.googleapis.com/auth/calendar', 'https://www.googleapis.com/auth/calendar.events']
        if (credentials_file is None) and (reauthenticate is False):
            if '.pkl' in credentials_file:
                credentials_file = os.path.join(Path.home(), 'token.pkl')  # assume token.pkl in home directory
            elif '.json' in credentials_file:
                credentials = Credentials.from_authorized_user_file(json_file, SCOPES)
        if reauthenticate:  # if want to re-authenticate Google
            if secrets_file is not None:  # create credentials file from secrets file if specified
                scopes = ['https://www.googleapis.com/auth/calendar']
                flow = InstalledAppFlow.from_client_secrets_file(secrets_file, scopes=scopes)
                credentials = flow.run_local_server(port=port)
                if credentials_file is not None:  # pickle credentials file if specified
                    pickle.dump(credentials, open(credentials_file, 'wb'))
        elif credentials_file is not None:  # ...or load credentials file instead of re-authenticating
            credentials = pickle.load(open(credentials_file, 'rb'))
        cal_kws = dict(cal_id=cal_id, credentials=credentials, time_zone=time_zone,
                       start=start, stop=stop, max_events=max_events,
                       only_with_summary=only_with_summary)
        if keyword_list is not None:  # extract based on keywords
            if isinstance(keyword_list, str):
                ca = get_calendar(keyword=keyword_list, **cal_kws)
            else:
                cl = [get_calendar(keyword=k, **cal_kws)
                      for k in keyword_list]  # list of calendar lists for keywords
                ca = functools.reduce(lambda x, y: [x, y][x is None] if any(
                    [x is None, y is None]) else x + y, cl)
        else:  # otherwise, whole calendar
            ca = get_calendar(**cal_kws)
        if isinstance(ca[0], list):
            ca = functools.reduce(lambda x, y: [x, y][x is None]
                                  if any([x is None, y is None]) else x + y, ca)
        ca = [None if pd.isnull(c) else c for c in ca]
        ca = [ca[i] for i in np.where([c is not None for c in ca])[0]]
        self.calendar = ca  # store calendar list of event dictionaries as attribute

    def to_frame(self, calendar=None, slot_key_dict={'dateTime': ['start', 'end'], 'date': ['start', 'end']},
                 cols=None, print_df=True, clock=24, file_out=None, keywords=None):
        """Calendar entry dictionaries to dataframe"""
        cal = self.calendar if calendar is None else calendar
        if slot_key_dict is not None:
            for k in slot_key_dict:  # extract keys from dictionaries nested in event slots
                cal = event_key_subset(cal, k, slot_key_dict[k])
        df = pd.concat(
            [pd.DataFrame([e[c] for c in e], index=e.keys()).T for e in cal]).set_index('id')
        df = df.join(df[['start', 'end']].applymap(lambda x: dateparser.parse(x)), lsuffix='_string')  # parse dates
        df = df.assign(duration=df.end - df.start).reset_index()  # event duration
        df = df.assign(hours=df.duration.apply(lambda x: x.total_seconds() / 3600))  # duration in hours
        ymdhm = df[['start', 'end']].applymap(lambda x: [x.year, x.month, x.day, x.hour, x.minute])  # ymd & hm
        df = df.rename({'summary': 'Event'}, axis=1).join(
            ymdhm, rsuffix='_list')  # date & time component lists
        st = ymdhm['start'].apply(pd.Series).rename(dict(zip(np.arange(5),
                                                             ['year', 'month', 'day', 'hour', 'minute'])), axis=1)
        df = df.join(st).set_index(list(st.columns)).sort_index()  # join event titles with times
        df = df.assign(Date=list(df.reset_index().apply(lambda x: f'{x.month}-{x.day}-{x.year}', axis=1)))  # mm-dd-yy
        if keywords is not None:  # if desired, subset by keywords
            if isinstance(keywords, str):  # if just 1 keyword provided...
                keywords = [keywords]  # ...ensure list
            df = df.reset_index()[df.reset_index().apply(
                lambda x: any([k.lower() in x.Event.lower()] for k in keywords), axis=1)].set_index(df.index.names)
        self.data = df.reset_index().set_index(list(st.columns) + ['Event'])  # times -> index
        if len(pd.unique(self.data.id)) != self.data.shape[0]:
            warnings.warn('May have duplicate event ids!')  # warn of potential duplicates
        if print_df:
            df_2_print = self.data
            if cols is not None:  # subset (if desired)
                if (type(cols) is not list) or any([type(c) is not str for c in cols]):
                    raise TypeError('cols must be a list of strings (column names)')
                df_2_print = df_2_print.reset_index()[cols].set_index(
                    wh([i in cols for i in df_2_print.index.names], 1, df_2_print.index.names))
                if all([c in df_2_print.index.names for c in cols]):  # if all columns by which to subset are indices
                    df_2_print = df_2_print.reset_index(-1)  # last index -> column
            hr12 = self.data.reset_index().apply(lambda x: x.hour - 12 if x.hour > 12 else x.hour, axis=1)
            df_2_print = self.data if clock == 24 else self.data.reset_index().assign(hour=hr12)
            if cols is not None:
                df_2_print = df_2_print.reset_index()[cols]
            if all(df_2_print.iloc[:, 0] == np.arange(df_2_print.shape[0])):
                df_2_print = df_2_print.drop(df_2_print.columns[0], axis=1)  # make sure no row #s column
            print(df_2_print)
        if file_out is not None:  # save .csv file (if desired)
            if '.csv' not in file_out:
                ValueError('File out must have a .csv extension')
            self.data.to_csv(file_out) if print_df is False else df_2_print.to_csv(file_out, index=False)
        if print_df:
            return df_2_print
        else:
            return self.data

    def find_events(self, keywords, exact=False):
        """Extract Events by Keyword (returns list of dictionaries, dataframe)"""
        if 'data' not in dir(self):
            self.to_frame()  # make data attribute if not already present
        if type(keywords) in [str, int, float]:
            keywords = [keywords]  # ensure keywords iterable (even if just 1)
        cdf = self.data.reset_index()
        fx = [lambda x, i: x.lower() in i.lower(), lambda x, i: x.lower() ==
              i.lower()][exact]  # exact or partial match fx
        try:
            cl = pd.Series(self.calendar).apply(
                lambda x: x if any([fx(k.lower(), x['summary'].lower()) for k in keywords]) else np.nan)
            cl = list(cl.dropna())
        except:
            cl = None
        try:
            df = self.to_frame(cl)
            if cl is None:
                warnings.warn('Could not extract events. Returning dataframe, None.')
        except:
            df = None
            if cl is not None:
                warnings.warn('Could not create dataframe. Returning None, list of dictionaries.')
            else:
                warnings.warn('Could not extract events. Returning None, None.')
        return df, cl

    def detect_duplicates(self):
        """Detect Invited Events with Duplicates"""
        warnings.warn('FEATURE UNDER DEVELOPMENT')
        if len(pd.unique(self.data.id)) != self.data.shape[0]:
            warnings.warn('May have duplicate event ids!')  # warn of potential duplicates
        private_copies = self.data[self.data.privateCopy == True]  # events with copies of invites
        cops = self.data.reset_index(-1).loc[wh([self.data.loc[i[:-1]].shape[0] > 1
                                                 for i in private_copies.index.values],
                                                1, [i[:-1] for i in private_copies.index.values])]
