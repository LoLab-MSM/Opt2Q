from urllib.request import urlretrieve, urlopen
import json
import os

ZENODO_URL = 'https://zenodo.org/api/records/4768842'
# Get list of URLs/dict from Zenodo
with urlopen(ZENODO_URL) as response:
    html = response.read().decode('utf8')
j = json.loads(html)

results_dir = 'ordinal_data_calibration_results'
os.mkdir(results_dir)
script_dir = os.path.join(os.path.dirname(__file__), results_dir)

urls = {f['links']['self']: f['key'] for f in j['files']}

for url, target in urls.items():
    _ = urlretrieve(url, os.path.join(script_dir, target))
