# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 22:07:02 2025

@author: Jakob
"""

import requests
import time
import json

# Adapted, all credit goes to the classyfire people.

# For Classyfire, I think we need to implement a delay between requests.
# They don't seem to like it if you query them too frequently.

# proxy_url =  "https://gnps-classyfire.ucsd.edu"
# chunk_size = 1000
# sleep_interval = 60

def classyfire_query(smiles, label='pyclassyfire'):
    url = "http://classyfire.wishartlab.com"
    r = requests.post(url + '/queries.json', data='{"label": "%s", '
                      '"query_input": "%s", "query_type": "STRUCTURE"}'
                                                  % (label, smiles),
                      headers={"Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()['id']

#id_no = classyfire_query('CC1CC2C3CCC(C3(CC(C2C4(C1=CC(=O)C=C4)C)O)C)(C(=O)CO)O')

def classyfire_fetch(query_id, return_format="json", blocking=False):
    url = "http://classyfire.wishartlab.com"   
    if blocking == False:
        r = requests.get('%s/queries/%s.%s' % (url, query_id, return_format),
                         headers={"Content-Type": "application/%s" % return_format})
        r.raise_for_status()
        return r.text
    else:
        while True:
            r = requests.get('%s/queries/%s.%s' % (url, query_id, return_format),
                             headers={"Content-Type": "application/%s" % return_format})
            r.raise_for_status()
            result_json = r.json()
            if result_json["classification_status"] != "In Queue":
                return r.text
            else:
                print("WAITING")
                time.sleep(10)
                
#fetched = classyfire_fetch(id_no)
    
def get_classyfire(smiles):
    query_id = classyfire_query(smiles)
    if query_id:
        results = classyfire_fetch(query_id)
        if results:
            try:
                results = json.loads(results)
                if results and results['entities']:
                    entity = results['entities'][0]
                    class_data = '; '.join([
                        #entity.get('kingdom', {}).get('name', ''), # Excessive
                        #entity.get('superclass', {}).get('name', ''), # Excessive
                        entity.get('class', {}).get('name', ''),
                        entity.get('subclass', {}).get('name', ''),
                        entity.get('direct_parent', {}).get('name', '')
                    ])
                    return class_data
            except json.JSONDecodeError:
                pass
    print(f'Classyfire query failed for {smiles}.')
    return None

#result = get_classyfire('CC(=O)OC1CN2CCC1CC2')
        
#get_classyfire('C1CN(CCN1)C2=NC3=CC=CC=C3C=C2')
#get_classyfire('C[N+]1(CCCC1)CC2=C(N3C(C(C3=O)NC(=O)C(=NOC)C4=CSC(=N4)N)SC2)C(=O)[O-]')