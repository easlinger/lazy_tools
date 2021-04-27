#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 08:42:41 2021

@author: ena
"""

import os
import os.path
import pickle
from pathlib import Path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
# from oauth2client import tools, client, file


def direct(directory=None):
    """
    Convert directory specified as None to home or ensure provided directory path ends in /

    Parameters
    ----------
    directory : str, optional
        Directory path. The default is None. LEAVE DEFAULT AS None -> Home (assumed by project_setup.py).

    Returns
    -------
    directory : str
        Cleaned &/or converted directory path.

    """
    if directory is None:
        directory = str(Path.home()) + '/'  # home directory if unspecified
    elif directory[-1] != '/':
        directory = directory + '/'  # make sure ends with slash
    return directory


def locate_pickles(directory=None):
    """Retrieve file names of pickled credentials (with & without URI redirect)"""
    directory = direct(directory)
    pkls = ['%stoken%s.pkl' % (directory, ['', '_re'][i]) for i in range(2)]  # pkl file names
    return pkls


def write_bash():
    """
    Write bash script that calls today.py to get today or tomorrow\'s calendar & set up & pickle credentials

    Parameters
    ----------
    directory : str, optional
        Directory path, ensuring ends in slash & specifying as home if None. The default is None.

    Returns
    -------
    None.

    """
    directory = direct(None)  # get default home directory or make sure provided one ends in slash
    bash_file_names = ['today.sh', f'{directory}/today.sh']  # target file names for bash script (current WD & home)
    bash_script = '#!/bin/bash\n\ncd\ncd %s\n\npython %s/today.py $1 $2 2>&1' % (os.getcwd(), os.getcwd())
    bash_file_loc = open(bash_file_names[0], 'w')  # open write connection to this module's bash file
    bash_file_home = open(bash_file_names[1], 'w')  # open write connection to home directory bash file
    bash_file_loc.writelines(bash_script)  # write script to current directory
    bash_file_loc.close()  # close file connection (current directory)
    bash_file_home.writelines(bash_script)  # write script to home
    bash_file_home.close()  # close file connection (home)
    for i in bash_file_names:  # iterate through file in current directory & home
        os.system('chmod +x %s' % i)  # make script executable


# Don't want to have to call in today.py when it's being called by the bash script
write_bash()


def make_save_creds(directory=None, secrets_stem='client_id', redirect_suffix='_redirect', port=0):
    """
    Make & store credentials

    Parameters
    ----------
    directory : str, optional
        Directory where secrets are located & pickled credentials should be stored. The default is None (i.e. home).
        \nLEAVE as DEFAULT None unless copying this function to use elsewhere (assumed by project_setup.py)
    secrets_stem : str, optional
        Stem (not including .json or directory) for file holding secrets. The default is 'client_id'.
    redirect_suffix : str, optional
        Suffix on secrets stem for file with secrets with URI redirect. The default is '_redirect'.
    port : int, optional
        Port for local server (if need to re-run Google API authentication). The default is 0.

    Returns
    -------
    credentials : list
        List containing credentials with & without redirect.
    service : googleapiclient.discovery.Resource
        Google API service.

    """
    directory = direct(directory)  # get default home directory or make sure provided one ends in slash
    SCOPES = ['https://www.googleapis.com/auth/calendar', 'https://www.googleapis.com/auth/calendar.events']
    secrets = [f'{directory}{secrets_stem}{s}.json' for s in ['', redirect_suffix]]  # from Google developers' console
    flow = InstalledAppFlow.from_client_secrets_file(secrets[0], scopes=SCOPES)  # flow
    creds = flow.run_local_server(port=port)  # credentials
    input('Press enter when ready. ')
    flow_re = InstalledAppFlow.from_client_secrets_file(secrets[1], scopes=SCOPES)  # flow with redirect
    creds_re = flow_re.run_local_server(port=0)  # credentials with redirect URIs
    pkls = locate_pickles()
    [pickle.dump(p[0], open(p[1], 'wb')) for p in zip([creds, creds_re], pkls)]  # store pickled credentials
    credentials = [pickle.load(open(p, 'rb')) for p in pkls]  # load from pickles
    service = build('calendar', 'v3', credentials=credentials[1])
    return credentials, service


def creds_json(directory=None, secrets_stem='client_id.json', json_file_relative='credentials.json',
               port=0, recreate=False):
    """
    Make & store credentials (JSON) or retrieve if present

    Parameters
    ----------
    directory : str, optional
        Directory where secrets are located & credentials should be stored. The default is None (i.e. home).
        \nLEAVE as DEFAULT None unless copying this function to use elsewhere (assumed by project_setup.py)
    secrets_stem : str, optional
        Stem (not including directory) for file holding secrets. The default is 'client_id'.
    json_file_relative : str, optional
        Relative path (not including directory) for file holding or to hold JSON. The default is 'credentials.json'.
    port : int, optional
        Port for local server (if need to re-run Google API authentication). The default is 0.
    recreate : bool, optional
        Recreate even if JSON already exists? The default is False.

    Returns
    -------
    creds : google.oauth2.credentials.Credentials
        Google-authorized credentials

    """
    SCOPES = ['https://www.googleapis.com/auth/calendar', 'https://www.googleapis.com/auth/calendar.events']
    directory = direct(directory)
    json_file = directory + json_file_relative
    if (not os.path.exists(json_file)) or (recreate):  # if want or need to recreate, do so
        print('Recreating JSON file...')
        flow = InstalledAppFlow.from_client_secrets_file(directory + secrets_stem, SCOPES)
        creds = flow.run_local_server(port=port)
        with open(json_file, 'w') as token:
            token.write(creds.to_json())
    else:
        creds = Credentials.from_authorized_user_file(json_file, SCOPES)
    return creds
