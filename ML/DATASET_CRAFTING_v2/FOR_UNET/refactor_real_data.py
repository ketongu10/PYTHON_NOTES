from pathlib import Path
import shutil
import os


source = Path("/vol2/WATER/RUNS/SEG/labeled_real_data_falses/images")
for vidos in source.iterdir():
    for sample in vidos.iterdir():
        s_name = sample.name
        v_name = vidos.name.replace("WATER_", "")
        v_parent =Path(vidos.parent)
        new_sample = Path(v_parent/f"{(v_name)}_{(sample.name)}")
        Path(sample).rename(new_sample)
        #new_sample.mkdir(parents=True)

        print(new_sample)