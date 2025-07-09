import matplotlib.pyplot as plt
import numpy as np
import os

# ===========SEEMS IT DOESN'T WORK


"""category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']
loh = ["loh"+str(i)for i in range(6)]
results = {
    'Question 1': [10, 15, 17, 32, 26],
    'Question 2': [26, 22, 29, 10, 13],
    'Question 3': [35, 37, 7, 2, 19],
    'Question 4': [32, 11, 9, 15, 33],
    'Question 5': [21, 29, 5, 5, 40],
    'Question 6': [8, 19, 5, 30, 38]
}


def survey(results, category_names, ax, key):
  
    labels = key #list(results.keys())
    data = np.array(results)#list(results.values()))
    #print(data)
    data_cum = data.cumsum(axis=0)
    #print(data_cum)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    #fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        print(widths)
        starts = data_cum[i] - widths
        print(starts)
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)

    ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
            loc='lower left', fontsize='small')

    return fig, ax




fig, axs = plt.subplots(nrows=6)
for ax, key in zip(axs, list(results.keys())):
    res = [[results[key][i] for j in range(5)] for i in range(5)]
    survey(res, category_names, ax, [key])
plt.show()"""

dirs = ["From 106 down/water_flow", "From 106 down3/water_flow", "from pc 6.12",
        "From 106 another one/water_flow", "From PC down new/water_flow", "From PC down new2/water_flow"]

dirs = ["From 106",
"From 106 mv light",
"From PC down new2",
"From pc 6.12",
"From PC mv light",
"From PC 2",
"From 106 last",
"From 106 down",
"From PC down",
"From 106 no water",
"From PC no water",
"From 106 mv light2",
"From 106 down3",
"From PC mv light2",
"From PC",
"From 106 another one",
"From PC down new",
"From 105"]


dirs = [

"From 104 ropes",
]

print(len(dirs))
print(dirs)
for grand_dir in dirs:
    VALUES = {"'should_water_flow'": {"True": 0,
                                      "False": 0},
              "'flow_type'": {"'Vjuh_from_pipe'": 0,
                              "'Flow_from_pipe'": 0,
                              "'Flow_from_spider'": 0,
                              "'Flow_from_flance'": 0,
                              "'Flow_from_half_flance'": 0,
                              "'Flow_down_from_spider'": 0,
                              "'Vjuh_down_from_pipe'": 0,
                              "'Flow_down_from_spider1'": 0,
                              "'Vjuh_down_from_pipe1'": 0,
                              "'Flow_down_from_spider2'": 0,
                              "'Flow_down_from_spider3'": 0,
                              "'Vjuh_down_from_pipe2'": 0,
                              "'Flow_down_with_jets": 0,
                              "'Flow_down_vertical_jets": 0,
                              }
              }
    UP_DOWN = {"up": 0,
               "down": 0}
    total = 0
    for dir in os.listdir(grand_dir+"/water_flow"):
        if dir != 'images':
            total+=1
            source = os.path.join(grand_dir+"/water_flow", dir)
            settings = os.path.join(source, 'settings.txt')
            with open(settings, 'r') as f:
                text = f.read()
                for key_type in VALUES.keys():
                    for answer in VALUES[key_type].keys():
                        string = key_type+": "+answer
                        if string in text:
                            VALUES[key_type][answer]+=1
    for key in VALUES["'flow_type'"].keys():
        if 'down' in key:
            UP_DOWN["down"] +=  VALUES["'flow_type'"][key]
        else:
            UP_DOWN["up"] += VALUES["'flow_type'"][key]
    print(grand_dir+";"+str(UP_DOWN["up"])+";"+str(UP_DOWN["down"])+";"+str(total)+";"+str(VALUES["'should_water_flow'"]["True"])+";"+str(VALUES["'should_water_flow'"]["False"]))

#print(VALUES)
#print(UP_DOWN)
up = ["From 106",
"From 106 mv light",
"From pc 6.12",
"From PC mv light",
"From PC 2",
"From 106 last",
"From 106 no water",
"From PC no water",
"From 106 mv light2",
"From PC mv light2",
"From 106 another one",
"From 106 6.12",
"From 106 7.12",
"From PC smoke",
"From 106 smoke"]

down = [
"From PC down new2",
"From 106 down",
"From 106 down3",
"From 106 another one",
"From PC down new",

"From 106 7.12",

"From 106 6.12",
"From pc 6.12",

"From PC 11.12",
"From 103 11.12",

"From 106 18.12",
"From PC 18.12",
"From 103 18.12",

"From PC 20.12",

"From PC smoke",
"From 106 smoke",
]

for a in up:
    print('"'+a+"/water_flow"+'",')


print("\n\n")
for a in down:
    print('"'+a+"/water_flow"+'",')
