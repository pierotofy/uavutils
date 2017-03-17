# -*- coding: utf-8 -*-

import string
import json
import os

with open(os.path.join(os.path.dirname(__file__), "sensor_data.json"),'rb') as f:
    sensor_data = json.loads(f.read())

# Convert model types to lower cases for easier query
sensor_data = dict(zip(map(str.lower,sensor_data.keys()),sensor_data.values()))
