#------------------------------------------------------------------------------
# Name:        toc_trends_analysis.py
# Purpose:     Analyse RESA2 data for trends.
#
# Author:      James Sample
#
# Created:     Fri Jul 15 11:35:12 2016
# Copyright:   (c) James Sample and NIVA
# Licence:     
#------------------------------------------------------------------------------
""" Tore has previously written code to perform trend analyses on the data in
    RESA2. I haven't been able to find the code, but it appears to shifts data 
    between RESA2, Excel and Access, which seems a bit messy.
    
    In the notebook updated_toc_trends_analysis.ipynb, I tested some code which
    refactors all the analysis into Python, interfacing directly with the 
    database and returning results as dataframes. This seems to have worked 
    well.
    
    The code below takes the main function from this notebook and tidies them
    up a bit. This file can then be imported into new notebooks, which should
    make it easy to re-run trend analyses on different datasets in the future.
"""

def mk_test(x, stn_id, par, alpha=0.05):
    """ Adapted from http://pydoc.net/Python/ambhas/0.4.0/ambhas.stats/
        by Sat Kumar Tomer.
    
        Perform the MK test for monotonic trends. Uses the "normal
        approximation" to determine significance and therefore should 
        only be used if the number of values is >= 10.
    
    Args:
        x:     1D array of data
        name:  Name for data series (string)
        alpha: Significance level
    
    Returns:
        var_s: Variance of test statistic
        s:     M-K test statistic
        z:     Normalised test statistic 
        p:     p-value of the significance test
        trend: Whether to reject the null hypothesis (no trend) at
               the specified significance level. One of: 
               'increasing', 'decreasing' or 'no trend'
    """
    import numpy as np
    from scipy.stats import norm
    
    n = len(x)
    
    if n < 10:
        print ('    Data series for %s at site %s has fewer than 10 non-null values. '
               'Significance estimates may be unreliable.' % (par, int(stn_id)))
    
    # calculate S 
    s = 0
    for k in xrange(n-1):
        for j in xrange(k+1,n):
            s += np.sign(x[j] - x[k])
    
    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)
    
    # calculate the var(s)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18.           
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in xrange(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        # Sat Kumar's code has "+ np.sum", which is incorrect
        var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18.
    
    if s>0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
            z = 0
    elif s<0:
        z = (s + 1)/np.sqrt(var_s)
    else:
        z = np.nan
        
    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z))) # two tail test
    h = abs(z) > norm.ppf(1-alpha/2.) 

    if (z<0) and h:
        trend = 'decreasing'
    elif (z>0) and h:
        trend = 'increasing'
    elif np.isnan(z):
        trend = np.nan
    else:
        trend = 'no trend'
        
    return var_s, s, z, p, trend

def wc_stats(raw_df, st_yr=None, end_yr=None, plot=False, fold=None):
    """ Calculate key statistics for the TOC trends analysis:
        
            'station_id'
            'par_id'
            'non_missing'
            'median'
            'mean'
            'std_dev'
            'period'
            'mk_std_dev'
            'mk_stat'
            'norm_mk_stat'
            'mk_p_val'
            'trend'
            'sen_slp'
    
    Args:
        raw_df:   Dataframe with annual data for a single station. Columns must 
                  be: [station_id, year, par1, par2, ... parn]
        st_yr:    First year to include in analysis. Pass None to start
                  at the beginning of the series
        end_year: Last year to include in analysis. Pass None to start
                  at the beginning of the series
        plot:     Whether to generate a PNG plot of the Sen's slope 
                  regression
        fold:     Folder in which to save PNGs if plot=True
        
    Returns:
        df of key statistics.
    """
    import numpy as np, pandas as pd
    import seaborn as sn, matplotlib.pyplot as plt, os
    from scipy.stats import theilslopes
    sn.set_context('poster')
    
    # Checking
    df = raw_df.copy()
    assert list(df.columns[:2]) == ['STATION_ID', 'YEAR'], 'Columns must be: [STATION_ID, YEAR, par1, par2, ... parn]'
    assert len(df['STATION_ID'].unique()) == 1, 'You can only process data for one site at a time'
    
    # Get just the period of interest
    if st_yr:
        df = df.query('YEAR >= @st_yr')
    if end_yr:
        df = df.query('YEAR <= @end_yr')
    
    # Get stn_id
    stn_id = df['STATION_ID'].iloc[0]
    
    # Tidy up df
    df.index = df['YEAR']
    df.sort_index(inplace=True)
    del df['STATION_ID'], df['YEAR']
    
    # Container for results
    data_dict = {'station_id':[],
                 'par_id':[],
                 'non_missing':[],
                 'median':[],
                 'mean':[],
                 'std_dev':[],
                 'period':[],
                 'mk_std_dev':[],
                 'mk_stat':[],
                 'norm_mk_stat':[],
                 'mk_p_val':[],
                 'trend':[],
                 'sen_slp':[]}
    
    # Loop over pars
    for col in df.columns:
        # 1. Station ID
        data_dict['station_id'].append(stn_id)
        
        # 2. Par ID
        data_dict['par_id'].append(col)
        
        # 3. Non-missing
        data_dict['non_missing'].append(pd.notnull(df[col]).sum())
        
        # 4. Median
        data_dict['median'].append(df[col].median())
        
        # 5. Mean
        data_dict['mean'].append(df[col].mean())
        
        # 6. Std dev
        data_dict['std_dev'].append(df[col].std())
        
        # 7. Period
        st_yr = df.index.min()
        end_yr = df.index.max()
        per = '%s-%s' % (st_yr, end_yr)
        data_dict['period'].append(per)

        # 8. M-K test
        # Drop missing values
        mk_df = df[[col]].dropna(how='any')

        # Only run stats if more than 1 valid value
        if len(mk_df) > 1:
            var_s, s, z, p, trend = mk_test(mk_df[col].values, stn_id, col)
            data_dict['mk_std_dev'].append(np.sqrt(var_s)) 
            data_dict['mk_stat'].append(s)
            data_dict['norm_mk_stat'].append(z)
            data_dict['mk_p_val'].append(p)
            data_dict['trend'].append(trend)      

            # 8. Sen's slope. Returns:
            # Median slope, median intercept, 95% CI lower bound, 
            # 95% CI upper bound
            sslp, icpt, lb, ub = theilslopes(mk_df[col].values, 
                                             mk_df.index, 0.95)
            data_dict['sen_slp'].append(sslp)
            
            # 9. Plot if desired
            if plot:
                fig = plt.figure()
                plt.plot(mk_df.index, mk_df[col].values, 'bo-')
                plt.plot(mk_df.index, mk_df.index*sslp + icpt, 'k-')
                if col in ('Al', 'TOC'):
                    plt.ylabel('%s (mg/l)' % col, fontsize=24)
                else:
                    plt.ylabel('%s (ueq/l)' % col, fontsize=24)
                plt.title('%s at station %s' % (col, int(stn_id)),
                          fontsize=32)
                plt.tight_layout()
                
                # Save fig
                out_path = os.path.join(fold,
                                        '%s_%s_%s-%s.png' % (int(stn_id), col, 
                                                             st_yr, end_yr))
                plt.savefig(out_path, dpi=150)
                plt.close()
            
        # Otherwise all NaN
        else:
            for par in ['mk_std_dev', 'mk_stat', 'norm_mk_stat', 
                        'mk_p_val', 'trend', 'sen_slp']:
                data_dict[par].append(np.nan)
    
    # Build to df
    res_df = pd.DataFrame(data_dict)
    res_df = res_df[['station_id', 'par_id', 'period', 'non_missing',
                     'mean', 'median', 'std_dev', 'mk_stat', 'norm_mk_stat',
                     'mk_p_val', 'mk_std_dev', 'trend', 'sen_slp']]    
    
    return res_df
    
def read_resa2(proj_list, engine):
    """ Reads raw data for the specified projects from RESA2. Extracts only
        the parameters required for the trends analysis and calculates 
        aggregated annual values by taking medians.
    
    Args:
        proj_list: List of RESA2 project names for which to extract data
        engine:    SQLAlchemy 'engine' object already connected to RESA2
        
    Returns:       
        [stn_df, wc_df, dup_df]. Dataframe of stations; Dataframe of annual 
        water chemistry values; dataframe of duplicates to check
    """
    import pandas as pd    

    # Get par IDs etc. for pars of interest
    par_list = ['SO4', 'Cl', 'Ca', 'Mg', 'NO3-N', 'TOC', 'Al']
    
    sql = ('SELECT * FROM resa2.parameter_definitions '
           'WHERE name in %s' % str(tuple(par_list)))
        
    par_df = pd.read_sql_query(sql, engine)
    
    # Get stations for a specified list of projects
    if len(proj_list) == 1:
        sql = ("SELECT station_id, station_code "
               "FROM resa2.stations "
               "WHERE station_id IN (SELECT UNIQUE(station_id) "
                                    "FROM resa2.projects_stations "
                                    "WHERE project_id IN (SELECT project_id "
                                                         "FROM resa2.projects "
                                                         "WHERE project_name = '%s'))"
                                                         % proj_list[0])
    else:
        sql = ('SELECT station_id, station_code '
               'FROM resa2.stations '
               'WHERE station_id IN (SELECT UNIQUE(station_id) '
                                    'FROM resa2.projects_stations '
                                    'WHERE project_id IN (SELECT project_id '
                                                         'FROM resa2.projects '
                                                         'WHERE project_name IN %s))'
                                                         % str(tuple(proj_list)))    
    stn_df = pd.read_sql(sql, engine)

    # Get results for ALL pars for these sites
    if len(stn_df)==1:
        sql = ("SELECT * FROM resa2.water_chemistry_values2 "
               "WHERE sample_id IN (SELECT water_sample_id FROM resa2.water_samples "
                                   "WHERE station_id = %s)"
                                   % stn_df['station_id'].iloc[0])    
    else:
        sql = ("SELECT * FROM resa2.water_chemistry_values2 "
               "WHERE sample_id IN (SELECT water_sample_id FROM resa2.water_samples "
                                   "WHERE station_id IN %s)"
                                   % str(tuple(stn_df['station_id'].values)))
        
    wc_df = pd.read_sql_query(sql, engine)
    
    # Get all sample dates for sites and period of interest
    # We're only interested in surface sample (< 1 m from surface)
    if len(stn_df)==1:
        sql = ("SELECT water_sample_id, station_id, sample_date "
               "FROM resa2.water_samples "
               "WHERE station_id = %s "
               "AND depth1 <= 1 "
               "AND depth2 <= 1" % stn_df['station_id'].iloc[0])    
    else:
        sql = ("SELECT water_sample_id, station_id, sample_date "
               "FROM resa2.water_samples "
               "WHERE station_id IN %s "
               "AND depth1 <= 1 "
               "AND depth2 <= 1" % str(tuple(stn_df['station_id'].values)))
        
    samp_df = pd.read_sql_query(sql, engine)
    
    # Join in par IDs based on method IDs
    sql = ('SELECT * FROM resa2.wc_parameters_methods')
    meth_par_df = pd.read_sql_query(sql, engine)
    
    wc_df = pd.merge(wc_df, meth_par_df, how='left',
                     left_on='method_id', right_on='wc_method_id')
    
    # Get just the parameters of interest
    wc_df = wc_df.query('wc_parameter_id in %s' 
                        % str(tuple(par_df['parameter_id'].values)))
    
    # Join in sample dates
    wc_df = pd.merge(wc_df, samp_df, how='left',
                     left_on='sample_id', right_on='water_sample_id')
    
    # Join in parameter units
    sql = ('SELECT * FROM resa2.parameter_definitions')
    all_par_df = pd.read_sql_query(sql, engine)
    
    wc_df = pd.merge(wc_df, all_par_df, how='left',
                     left_on='wc_parameter_id', right_on='parameter_id')
    
    # Join in station codes
    wc_df = pd.merge(wc_df, stn_df, how='left',
                     left_on='station_id', right_on='station_id')
    
    # Convert units
    wc_df['value'] = wc_df['value'] * wc_df['conversion_factor']
    
    # Extract columns of interest
    wc_df = wc_df[['station_id', 'sample_date', 'name', 'value']]
    
    # Check for duplicates
    dup_df = wc_df[wc_df.duplicated(subset=['station_id',
                                            'sample_date',
                                            'name'], 
                                            keep=False)].sort_values(by=['station_id', 
                                                                         'sample_date', 
                                                                         'name'])
    if len(dup_df) > 0:
        print ('    The database contains duplicate values for some station-'
               'date-parameter combinations.\n    These will be averaged, but '
               'you should check the repeated values are not errors.\n    The '
               'duplicated entries are returned in a separate dataframe.\n')
        
        # Groupby station_id, date and par
        grpd = wc_df.groupby(['station_id', 'sample_date', 'name'])
        
        # Calculate mean
        wc_df = grpd.agg('mean')
        
        wc_df.reset_index(inplace=True)
    
    # Unstack
    wc_df.set_index(['station_id', 'sample_date', 'name'], inplace=True)
    wc_df = wc_df.unstack(level='name')
    wc_df.columns = wc_df.columns.droplevel()
    wc_df.reset_index(inplace=True)
    wc_df.columns.name = None
    
    # Extract year from date column
    wc_df['year'] = wc_df['sample_date'].map(lambda x: x.year)
    del wc_df['sample_date']
    
    # Groupby station_id and year
    grpd = wc_df.groupby(['station_id', 'year'])
    
    # Calculate median
    wc_df = grpd.agg('median')
    
    return stn_df, wc_df, dup_df

def conv_units_and_correct(wc_df):
    """ Take a dataframe of aggregated annual values in the units specified by
        RESA2.PARAMETERS and performs unit conversions to ueq/l. Also applies
        sea-salt correction where necessary.
    
    Args:
        wc_df: Dataframe in original units
    
    Returns:
        Dataframe in converted units
    """
    import pandas as pd
    
    # Tabulate chemical properties
    chem_dict = {'molar_mass':[96, 35, 40, 24, 14],
                 'valency':[2, 1, 2, 2, 1],
                 'resa2_ref_ratio':[0.103, 1., 0.037, 0.196, 'N/A']}
    
    chem_df = pd.DataFrame(chem_dict, index=['SO4', 'Cl', 'Ca', 'Mg', 'NO3-N'])
    chem_df = chem_df[['molar_mass', 'valency', 'resa2_ref_ratio']]
    
    # 1. Convert to ueq/l
    for par in ['SO4', 'Cl', 'Mg', 'Ca', 'NO3-N']:
        val = chem_df.ix[par, 'valency']
        mm = chem_df.ix[par, 'molar_mass']
        
        if par == 'NO3-N':
            wc_df['ENO3'] = wc_df[par] * val / mm
        else:
            wc_df['E%s' % par] = wc_df[par] * val * 1000. / mm
    
    # 2. Apply sea-salt correction
    for par in ['ESO4', 'EMg', 'ECa']:
        ref = chem_df.ix[par[1:], 'resa2_ref_ratio']
        wc_df['%sX' % par] = wc_df[par] - (ref*wc_df['ECl'])
        
    # 3. Calculate combinations
    # 3.1. ESO4 + ECl
    wc_df['ESO4_ECl'] = wc_df['ESO4'] + wc_df['ECl']
    
    # 3.2. ECa + EMg
    wc_df['ECa_EMg'] = wc_df['ECa'] + wc_df['EMg']
    
    # 3.3. ECaX + EMgX
    wc_df['ECaX_EMgX'] = wc_df['ECaX'] + wc_df['EMgX']
    
    # 3.4. ESO4 + ECl + ENO3
    wc_df['ESO4_ECl_ENO3'] = wc_df['ESO4'] + wc_df['ECl'] + wc_df['ENO3']
    
    # 4. Delete unnecessary columns and tidy
    for col in ['SO4', 'Cl', 'Mg', 'Ca', 'NO3-N']:
        del wc_df[col]
    
    wc_df.reset_index(inplace=True)
        
    return wc_df

def run_trend_analysis(proj_list, engine, st_yr=None, end_yr=None,
                       plot=False, fold=None):
    """ Run the trend analysis for the specified projects and time period.
    
    Args:
        proj_list: List of RESA2 project names for which to extract data
        engine:    SQLAlchemy 'engine' object already connected to RESA2
        st_yr:     First year to include in analysis. Pass None to start
                   at the beginning of the series
        end_year:  Last year to include in analysis. Pass None to start
                   at the beginning of the series
        plot:      Whether to generate a PNG plot of the Sen's slope 
                   regression
        fold:      Folder in which to save PNGs if plot=True
    
    Returns:    
        [res_df, dup_df, no_data_df]. Dataframe of statistics; dataframe of 
        duplicated water chemistry values for investigation; dataframe of 
        stations with no relevant data in the period of interest
    """
    import pandas as pd, os
    
    # Check paths valid
    if plot:
        assert os.path.isdir(fold), 'The specified folder does not exist.'
    
    # Get raw data from db
    print 'Extracting data from RESA2...'
    stn_df, wc_df, dup_df = read_resa2(proj_list, engine)
    
    # Identify stations with no relevant records
    stns_no_data = (set(stn_df['station_id'].values) - 
                    set(wc_df.index.get_level_values('station_id')))
    
    if len(stns_no_data) > 0:
        print ('    Some stations have no relevant data in the period '
               'specified. Their IDs are returned in a separate dataframe.\n')
        no_data_df = pd.DataFrame({'station_id':list(stns_no_data)})
    else:
        no_data_df = None
    
    print '    Done.'
    
    # Convert units and apply sea-salt correction
    print '\nConverting units and applying sea-salt correction...'
    wc_df = conv_units_and_correct(wc_df)
    print '    Done.'
    
    # Calculate stats    
    # Container for output
    df_list = []

    # Loop over sites
    print '\nCalculating statistics...'
    for stn_id in wc_df['station_id'].unique():
        # Extract data for this site
        df = wc_df.query('station_id == @stn_id')

        # Modify col names
        names = list(df.columns)
        names[:2] = ['STATION_ID', 'YEAR']
        df.columns = names

        # Run analysis
        df_list.append(wc_stats(df, st_yr=st_yr, end_yr=end_yr,
                                plot=plot, fold=fold))
   
    res_df = pd.concat(df_list, axis=0)

    print '    Done.'    
    print '\nFinished.'
    
    return res_df, dup_df, no_data_df