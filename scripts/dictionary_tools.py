# Load and dump json files
import json

def load_json(json_fi):
	with open(json_fi) as jf:
		d = json.load(jf)
	return d

def dump_json(d, json_fi, indent=0):
	with open(json_fi, 'w') as jf_o:
		json.dump(d, jf_o, indent=indent)

