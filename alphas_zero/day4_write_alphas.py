import requests
import json
from os.path import expanduser
from requests.auth import HTTPBasicAuth
import tqdm
import requests
import json
from os.path import expanduser
from requests.auth import HTTPBasicAuth
import tqdm
import logging
import time
import os
import ast
from datetime import datetime
from pytz import timezone
import csv

def sign_in():
	# Load credentials
	with open(expanduser("brain_credentials.txt")) as f:
		credentials = json.load(f)
	# Extract username and password from the list
	username, password = credentials
	# Create a session object
	sess = requests.Session()
	# Set up basic authentication
	sess.auth = HTTPBasicAuth(username, password)
	# Send a POST request to the API for authentication
	response = sess.post("https://api.worldquantbrain.com/authentication")
	# Print response status and content for debugging
	print(response.status_code)
	print(response.json())
	return sess

sess = sign_in()

#特#Get Datafield 1ike Data Explorer 获取所有满足条件的数据字段及其工D
def get_datafields(
        s,
        searchScope,
        dataset_id: str = '',
        search: str = ''):
    import pandas as pd
    instrument_type = searchScope['instrumentType']
    region = searchScope[ 'region' ]
    delay = searchScope['delay']
    universe = searchScope ['universe']
    if len(search) == 0:
        url_template = "https://api.worldquantbrain.com/data-fields?"+\
            f"&instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" +\
            "&offset={x}"
        count = s.get(url_template.format(x=0)).json()['count']
    else:
        url_template = "https://api.worldquantbrain.com/data-fields?" +\
            f"&instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" +\
            f"&search={search}" +\
            "&offset={x}"
        count = 100

    datafields_list = []
    for x in range(0, count, 50):
        datafields = s.get(url_template.format(x=x))
        datafields_list.append(datafields.json()['results'])
    datafields_list_flat = [item for sublist in datafields_list for item in sublist]
    datafields_df = pd.DataFrame(datafields_list_flat)
    return datafields_df


searchScope = {'region': 'USA', 'delay': '1', 'universe': 'TOP3000', 'instrumentType': 'EQUITY'}
fundamental6 = get_datafields(sess, searchScope=searchScope, dataset_id='fundamental6')

fundamental6 = fundamental6[fundamental6['type']=="MATRIX"]
datafields_list_fundamental6 = fundamental6['id'].values


group_compare_op = ['group_rank', 'group_zscore', 'group_neutralize']
ts_compare_op = ['ts_rank', 'ts_zscore', 'ts_av_diff']
company_fundamentals = datafields_list_fundamental6  # Replace with actual list if needed
days = [66, 252]
group = ['market', 'industry', 'subindustry', 'sector', 'densify(pv13_h_f1_sector)']
alpha_expressions = []

for eco in group_compare_op:
    for tco in ts_compare_op:
        for cf in company_fundamentals:
            for d in days:
                for grp in group:
                    alpha_expression = f"{eco}({tco}({cf}, {d}), {grp})"
                    alpha_expressions.append(alpha_expression)

# Print or return the result_strings list
print(f'there are total {len(alpha_expressions)} alpha expressions')

alpha_list = []
for alpha_expression in alpha_expressions:
    # 将如下Alpha表达式与setting封装
    simulation_data = {
        "type": "REGULAR",
        "settings": {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP1000",
            "delay": 1,
            "decay": 0,
            "neutralization": "SUBINDUSTRY",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "language": "FASTEXPR",
            "visualization": False,
            },
        "regular": alpha_expression
    }
    
    alpha_list.append(simulation_data)

print(f'there are total {len(alpha_list)} alpha simulations')
print(alpha_list[0])

import csv

estern = timezone('US/Eastern')
fmt = '%Y-%m-%d %H:%M:%S %Z%z'
loc_dt = datetime.now(estern)
print("estern time:", loc_dt.strftime(fmt))

with open(f'alpha_pending_simulation_list_{loc_dt.strftime(fmt)}.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['type', 'settings', 'regular'])  # Write header
    for alpha in alpha_list[3044:]:  # Write from the 3045th alpha simulation onward
        writer.writerow([alpha['type'], json.dumps(alpha['settings']), alpha['regular']])  # Write each alpha simulation