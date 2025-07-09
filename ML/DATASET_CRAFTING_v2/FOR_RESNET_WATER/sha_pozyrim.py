import pandas as pd
import numpy as np
import os
from pathlib import Path
dirs = [
        #"UP NEW/103 16.02 up new/water_flow","UP NEW/106 16.02 up new/water_flow",
        #"UP NEW/104 19.02 up new/water_flow","UP NEW/106 19.02 up new/water_flow",
        #"UP NEW/104 26.02 down new/water_flow","UP NEW/106 26.02 up new/water_flow",
        #"UP NEW/104 29.02 down new/water_flow","UP NEW/106 29.02 up new/water_flow",
        #"UP NEW/104 25.03 up new/water_flow","UP NEW/106 25.03 up new/water_flow",
        #"UP NEW/PC 25.03 up new/water_flow",
        #"UP NEW/PC 15.04 up new/water_flow", "UP NEW/104 15.04 up new/water_flow",
        #"UP NEW/PC 17.04 up new/water_flow", "UP NEW/104 17.04 up new/water_flow",
        #"UP NEW/105 17.04 up new/water_flow", "UP NEW/109 17.04 up new/water_flow",
        #"UP NEW/107 2.05 up new new/water_flow", "UP NEW/PC 2.05 up new new/water_flow",
        "UP NEW/104 2.05 up new new/water_flow",
        #"TEST/water_flow",
        ]
banlist = ["color1", "color2", "roughness", "brightness", "IOR", "domain_dimensions", "cache_folder", "baking_time", "smoke_cache_folder", "temperature",
           "blackbody_intensity", "smoke_color", "density"]
print(len(dirs))

DATA = pd.DataFrame(columns=['key','num'])

for dir in dirs:
    for vidos in os.listdir(dir):
        if vidos != 'images':
            source = os.path.join(dir, vidos)

            settings = os.path.join(source, 'settings.txt')
            js = eval(Path(settings).read_text().replace("Vector", ""))
            new_js = {}
            for key, val in zip(js.keys(), js.values()):
                if key not in banlist:
                    new_js[key] = [val]
            print(new_js)
            data = pd.DataFrame(new_js)
            print(data)
            exit()
            """for key, val in zip(js.keys(), js.values()):
                if key not in banlist:
                    try:
                        DATA[key+str(val)] += 1
                    except:
                        DATA[key+str(val)] = 1"""
print(DATA["flow_typeFlow_from_pipe"])