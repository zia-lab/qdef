#!/usr/bin/env python

# +------------------------------------------------------------------+
# |                                                                  |
# |   This script parses the data from the radial averages (<r^2>,   |
# |   <r^4>, <r^6>) for free ions as found in Fraga's Handbook of    |
# |                           Atomic Data.                           |
# |                                                                  |
# |      Data is contained in the spreadsheet ./data/HF radial       |
# |                          averages.xlsx.                          |
# |                                                                  |
# |     It changes the unit of length for the quoted values from     |
# |   atomic units to Angstrom and it extrapolates values for the    |
# |                  fifth spectrum in most cases.                   |
# |                                                                  |
# |        Results are saved both to a pickle that contains a        |
# |    dictionary and to three csv files. The dictionary has keys    |
# |       ["<r^2>","<r^4>","<r^6>"] and values equal to pandas       |
# |        dataframes. The csv values are csv exports of the         |
# |                    corresponding dataframes.                     |
# |                                                                  |
# |                    Oct-19 2022-10-19 11:59:24                    |
# |                                                                  |
# +------------------------------------------------------------------+

import numpy as np
import pandas as pd
import pickle, os
import os

def sigrounder(num, sig_figs):
    '''
    Round a given number to the given number of significant figures.
    Not elegant but gets the job done.
    '''
    pat = '%.'+str(sig_figs)+'g'
    return float('%s' % float(pat % num))

conversion_factors = pickle.load(open(os.path.join('./','data','conversion_facts.pkl'),'rb'))
pickle_fname = './data/HF_radial_avgs.pkl'
save_to_disk = True

if __name__ == '__main__':
    radialavgs = pd.read_excel("./data/HF radial averages.xlsx",None)
    metadata = radialavgs['Preamble']
    radialavgs = {k:radialavgs[k][['Atom','II','III','IV']] for k in ['r^2','r^4','r^6']}
    # conversion from atomic units to Å^2, Å^4, and Å^6
    for k in ['r^2','r^4','r^6']:
        power = int(k.split('^')[-1])
        scaler = conversion_factors[('a_{0}','Å')]**power
        for ionstate in 'II III IV'.split(' '):
            radialavgs[k][ionstate] = radialavgs[k][ionstate]*scaler
            radialavgs[k][ionstate] = radialavgs[k][ionstate].apply(lambda x: sigrounder(x,5))
        radialavgs[k].set_index('Atom',inplace=True)
    HFradavg = {}
    HFradavg['<r^2>'] = radialavgs['r^2']
    HFradavg['<r^2>'].metadata = str(metadata.iloc[0]['preamble']) + '\n Units were changed from atomic units to A^2.' + '\nData for V is extrapolated.'
    HFradavg['<r^4>'] = radialavgs['r^4']
    HFradavg['<r^4>'].metadata = str(metadata.iloc[1]['preamble']) + '\n Units were changed from atomic units to A^4.' + '\nData for V is extrapolated.'
    HFradavg['<r^6>'] = radialavgs['r^6']
    HFradavg['<r^6>'].metadata = str(metadata.iloc[2]['preamble']) + '\n Units were changed from atomic units to A^6.' + '\nData for V is extrapolated.'

    # extrapolate for X^4+ using a 2nd order polynomial in the cases where 
    # the values for II III and IV are present
    print("Extrapolating values for the 5th spectrum.")
    for key in HFradavg:
        dframe = HFradavg[key]
        extras = []
        for row in dframe.iterrows():
            vals = list(row[1])
            isnan = list(map(np.isnan, vals))
            if any(isnan):
                extra = np.nan
            else:
                fitparams = np.polyfit([1,2,3],vals,2)
                extra = sigrounder(np.polyval(fitparams, 4),5)
            extras.append(extra)
        HFradavg[key]['V'] = extras
    
    if save_to_disk:
        print("Saving to pickle %s..." % pickle_fname)
        pickle.dump(HFradavg, open(pickle_fname,'wb'))
        for key, dframe in HFradavg.items():
            csv_fname = './data/HF_radial_avg_%s.csv' % key
            print("Savint to csv %s..." % csv_fname)
            dframe.to_csv(csv_fname)
