import requests
import json
from os.path import expanduser
from requests.auth import HTTPBasicAuth
import pandas as pd
import tqdm

# Load credentials
with open(expanduser('brain_credentials.txt')) as f:
    credentials = json.load(f)

# Extract username and password from the list
username, password = credentials

# Create a session object
sess = requests.Session()

# Set up basic authentication
sess.auth = HTTPBasicAuth(username, password)

# Send a POST request to the API for authentication
response = sess.post('https://api.worldquantbrain.com/authentication')

# Print response status and content for debugging
print(response.status_code)
try:
    print(response.json())
except Exception:
    print(response.text)

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

alpha_list = []
for datafield in datafields_list_fundamental6:
    print("正在将如下Alpha表达式与setting封装")
    alpha_expression = f'group_rank({datafield}/cap, subindustry)'
    print(alpha_expression)
    simulation_data = {
        'type': 'REGULAR',
        'settings': {
            'instrumentType': 'EQUITY',
            'region': 'USA',
            'universe': 'TOP3000',
            'delay': 1,
            'decay': 0,
            'neutralization': 'SUBINDUSTRY',
            'truncation': 0.08,
            'pasteurization': 'ON',
            'unitHandling': 'VERIFY',
            'nanHandling': 'ON',
            'language': 'FASTEXPR',
            'visualization': False,
        },
        'regular': alpha_expression
    }
    alpha_list.append(simulation_data)
print(f'there are {len(alpha_list)} Alphas to simulate')


from time import sleep
for alpha in tqdm.tqdm(alpha_list[336:]):
    while True:
        try:
            sim_resp = sess.post('https://api.worldquantbrain.com/simulations', json=alpha)
            break
        except Exception as e:
            print(f"Exception during POST request: {e}. Retrying in 10 seconds...")
            sleep(10)

    try:
        sim_progress_url = sim_resp.headers['Location']
        while True:
            sim_progress_resp = sess.get(sim_progress_url)
            retry_after_sec = float(sim_progress_resp.headers.get("Retry-After", 0))
            if retry_after_sec == 0:  # simulation done!
                break
            sleep(retry_after_sec)
        alpha_id = sim_progress_resp.json()["alpha"]  # the final simulation result
        print(alpha_id)
    except:
        print("no location, sleep for 10 seconds and try next alpha")
        sleep(10)

