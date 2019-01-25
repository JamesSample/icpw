#-------------------------------------------------------------------------------
# Name:        mvm_python.py
# Purpose:     Download data from the MVM web API.
#
# Author:      James Sample
#
# Created:     19/10/2018
# Copyright:   (c) James Sample and NIVA, 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
""" Basic Python functions for downloading data from the Swedish water quality 
    database:
    
        https://miljodata.slu.se/mvm/    
"""

def convert_dates(time_string): 
    """ Modifed from here:
            
            https://stackoverflow.com/a/28507530/505698
    """
    import re
    from datetime import datetime, timedelta, timezone

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    ticks, offset = re.match(r'/Date\((\d+)([+-]\d{4})?\)/$', time_string).groups()
    utc_dt = epoch + timedelta(milliseconds=int(ticks))

    if offset:
        offset = int(offset)
        hours, minutes = divmod(abs(offset), 100)
        if offset < 0:
            hours, minutes = -hours, -minutes
        dt = utc_dt.astimezone(timezone(timedelta(hours=hours, minutes=minutes)))

        return datetime(dt.year, dt.month, dt.day)
    
    else:
        return datetime(utc_dt.year, utc_dt.month, utc_date.day)
    
def get_mvm_token(token_path):
    """ Read valid MVM access token from .xlsx
    """
    import pandas as pd
    import datetime as dt
    
    # Read Excel
    df = pd.read_excel(token_path, sheet_name='Sheet1')
    
    # Get most recent token
    df.sort_values(by=['Expiry_Date'],
                   inplace=True,
                   ascending=False)
    token = df['Token'][0]
    exp_date = df['Expiry_Date'][0]
    
    # Check valid
    if exp_date < dt.datetime.now():
        raise ValueError('The token file has no valid tokens.\n'
                         'Please log-in to MVM and update the tokens in the file:\n\n'
                         '    https://miljodata.slu.se/mvm/')
    
    return token

def query_mvm_station_data(site_id, st_yr, end_yr, token_path):
    """ Download data for a specific site using the Miljödata MVM API.
    
        Based on documentation here:
        
            http://miljodata.slu.se/mvm/OpenAPI
        
        Args:
            site_id:      Int. MD-MVM ID for site of interest
            st_yr:        Int. Start year of interest
            end_yr:       Int. End year of interest 
            public_token: Raw str. Path to .xlsx containing tokens. See
                          example .xlsx for details
        
        Returns:
            Dataframe
    """
    import requests
    import pandas as pd
    import numpy as np
    
    # Get access token
    token = get_mvm_token(token_path)

    # Build url and get data
    url = (r'http://miljodata.slu.se/mvm/ws/ObservationsService.svc/rest'
           r'/GetSamplesBySite?token=%s&siteid=%s&fromYear=%s&toYear=%s' % (token, 
                                                                            site_id, 
                                                                            st_yr, 
                                                                            end_yr))
    response = requests.get(url)
    data = response.json()

    # Dict for data of interest
    data_dict = {'mvm_id':[],
                 'station_name':[],
                 'station_type':[],
                 'sample_id':[],
                 'sample_date':[],
                 'depth1':[],
                 'depth2':[],
                 'par_name':[],
                 'par_desc':[],
                 'unit':[],
                 'value':[],
                 'flag':[]}

    # Loop over samples
    for samp in data:
        # Get sample data
        stn_id = int(samp['SiteId'])
        stn_name = samp['SiteName']
        stn_type = samp['SiteType']
        samp_id = int(samp['SampleId'])
        date = convert_dates(samp['SampleDate'])
        
        depth1 = samp['MinDepth']
        if depth1:
            depth1 = float(samp['MinDepth'].replace(',', '.'))
        else:
            depth1 = np.nan
            
        depth2 = samp['MaxDepth']
        if depth2:
            depth2 = float(samp['MaxDepth'].replace(',', '.'))
        else:
            depth2 = np.nan

        # Loop over pars
        for par in samp['ObservationValues']:
            # Get par data
            par_name = par['PropertyAbbrevName']
            par_desc = par['PropertyName']
            par_unit = par['UnitOfMeasureName']
            
            # Deal with LOD flags
            par_value = par['ReportedValue']
            if par_value:
                par_value = par_value.replace(',', '.')
                if par_value[0] in ('<', '>'):
                    flag = par_value[0]
                    par_value = par_value[1:]
                else:
                    flag = np.nan
                par_value = float(par_value)
            else:
                par_value = np.nan
                flag = np.nan            

            # Add to dict
            data_dict['mvm_id'].append(stn_id)
            data_dict['station_name'].append(stn_name)
            data_dict['station_type'].append(stn_type)
            data_dict['sample_id'].append(samp_id)
            data_dict['sample_date'].append(date)
            data_dict['depth1'].append(depth1)
            data_dict['depth2'].append(depth2)
            data_dict['par_name'].append(par_name)
            data_dict['par_desc'].append(par_desc)
            data_dict['unit'].append(par_unit)
            data_dict['value'].append(par_value)
            data_dict['flag'].append(flag)

    df = pd.DataFrame(data_dict)
    df.sort_values(by=['sample_date', 'par_name'],
                   inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    return df
