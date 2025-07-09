#import pyautogui
import os
import shutil

import numpy as np
import pynput.keyboard
from pynput.keyboard import Key, Controller
from pynput.mouse import Button as mButton
from pynput.mouse import Controller as mCtrl
from subprocess import run
from multiprocessing import Process
from positions import *
import time












mouse = mCtrl()
keyboard = Controller()

W, H = 1920, 1080

def disable_mouse(a):
    run(['xinput', 'set-prop', '10', 'Device Enabled', str(a)])

def run_liquigen(anim_path):
    run(['/home/popovpe/Downloads/liquigen-latest/LiquiGen-0.3.0-alpha/liquigen',
        anim_path],
        shell=False
)

def sleep_while_not_enought(dir):
    timeout = 0
    max_time = 120
    while True:
        time.sleep(1)
        timeout+=1
        if os.path.exists(dir) and len(os.listdir(dir)) > 10:
            max_time = 360
        if os.path.exists(dir) and len(os.listdir(dir)) == 135:
            return 0, timeout

        if timeout > max_time:
            return 1, timeout

def double_clic(position, delay=GENERAL_DELAY):
    mouse.position = position
    time.sleep(delay)
    mouse.press(mButton.left)
    mouse.release(mButton.left)
    time.sleep(0.2)
    mouse.press(mButton.left)
    mouse.release(mButton.left)
    time.sleep(delay)

def keyboard_print(data: str, delay=GENERAL_DELAY):
    for char in data:
        keyboard.press(char)
        keyboard.release(char)
    time.sleep(delay)

def keyboard_print_win(data: str, delay=GENERAL_DELAY):
    for char in data:
        if char.isupper() or char == '_' or char == ':':
            keyboard.press(pynput.keyboard.Key.shift)
            keyboard.press(char)
            keyboard.release(char)
            keyboard.release(pynput.keyboard.Key.shift)
        else:
            keyboard.press(char)
            keyboard.release(char)
    time.sleep(delay)

def press_key(key: str | pynput.keyboard.Key, delay=GENERAL_DELAY):
    keyboard.press(key)
    keyboard.release(key)
    time.sleep(delay)

def lClic(position=(W/2, H/2), delay=GENERAL_DELAY):
    mouse.position = position
    time.sleep(delay)
    mouse.press(mButton.left)
    mouse.release(mButton.left)
    time.sleep(delay)

def rClic(position=(W/2, H/2), delay=GENERAL_DELAY):
    mouse.position = position
    time.sleep(delay)
    mouse.press(mButton.right)
    mouse.release(mButton.right)
    time.sleep(delay)

def connect(pos1, pos2, delay=GENERAL_DELAY):
    mouse.position = pos1
    time.sleep(delay)
    mouse.press(mButton.left)
    time.sleep(delay)
    mouse.position = pos2
    time.sleep(delay)
    mouse.release(mButton.left)
    time.sleep(delay)

def ctrl_s(delay=GENERAL_DELAY):
    keyboard.press(pynput.keyboard.Key.ctrl_l)
    time.sleep(0.1)
    keyboard.press(pynput.keyboard.Key.shift_l)
    time.sleep(0.1)
    keyboard.press('s')
    time.sleep(0.1)
    keyboard.release(pynput.keyboard.Key.ctrl)
    time.sleep(0.1)
    keyboard.release(pynput.keyboard.Key.shift)
    time.sleep(0.1)
    keyboard.release('s')
    time.sleep(0.1+delay)

def set_anim_point(position, value, delay=GENERAL_DELAY):

    lClic(position)
    double_clic(position, delay)

    lClic((pos["Anim.set_value.x"],position[1]) , delay)

    keyboard_print(str(value))
    press_key(pynput.keyboard.Key.enter)
    time.sleep(delay)

def gen_sequence_speed(pipe_choice):
    vs = pipes[pipe_choice].v
    arr = []
    N = np.random.randint(N_speed_min,N_speed_max)
    for i in range(N+2):
        if i == 0:
            x = 0
        elif i == N + 1:
            x = 1
        else:
            x = i / (N + 1) + 1 / (N + 1) * np.random.uniform(-0.25, 0.25)
        x = int(x * pos["Anim.x.end"] + (1 - x) * pos["Anim.x.start"])
        arr.append((x, np.random.uniform(vs[0], vs[1])))
    return arr

def gen_sequence_target(pipe_choice):
    vs = pipes[pipe_choice].target
    ret = {}
    for key, value in vs.items(): # MEANS FOR X, Y, Z
        N = np.random.randint(N_target_min, N_target_max)
        if isinstance(value, float) or isinstance(value, int):
            ret[key] = value
        if isinstance(value, tuple):
            arr = []
            for i in range(N+2):
                if i == 0:
                    x = 0
                elif i == N+1:
                    x = 1
                else:
                    x = i / (N + 1) + 1 / (N + 1) * np.random.uniform(-0.25, 0.25)

                x = int(x * pos["Anim.x.end"] + (1 - x) * pos["Anim.x.start"])
                arr.append((x, np.random.uniform(value[0], value[1])))
            ret[key] = arr

    return ret



def create_mesh(pipe_choice, file_h, anim, stdout=None, is_test=False):
    P1 = Process(target=run_liquigen, args=(f'{PATH2ANIMS}/{pipe_choice}/{anim}', ))
    P1.start()
    time.sleep(LARGE_DELAY)
    disable_mouse(0)
    # PREPARE WORKSPACE
    lClic(pos["StopAnim"])
    lClic(pos["Import"])  # change window to liquigen
    connect(pos["Shtorka.right"], (W / 2, H / 2))  # move shtorka left

    # SET UP COLLISION IMPORT SETTINGS
    lClic(pos["Import.pos"])
    connect(pos["Shtorka.down"], (W * 3 / 4, H / 2))  # move shtorka mid
    lClic(pos["Import.filepath"])
    keyboard_print(f'{PATH_W_PIPES}/{pipe_choice}/{file_h}')
    lClic(pos["Import.reimport"])
    time.sleep(LARGE_DELAY)
    connect((W * 3 / 4, H / 2), pos["Shtorka.up"])  # move shtorka up
    lClic(pos["Import.select_all"])
    for el in pipes[pipe_choice].get_hidden_in_collision():
        if el:
            lClic((pos["Import.elements_x"], el))
    lClic(pos["Import.transform_after_sh_up"])
    lClic(pos["Import.scale"])
    keyboard_print("500")
    connect(pos["Shtorka.up"], pos["Shtorka.down"])

    # COPY AND CONNECT IMPORT TO EMITTER
    lClic(pos["Import.cp_pos_1"])
    keyboard.press(pynput.keyboard.Key.ctrl)
    keyboard.press("c")
    time.sleep(GENERAL_DELAY)
    keyboard.release("c")
    time.sleep(GENERAL_DELAY)
    lClic(pos["Import.cp_pos_2"])
    keyboard.press("v")
    time.sleep(GENERAL_DELAY)
    keyboard.release("v")
    keyboard.release(pynput.keyboard.Key.ctrl)

    # SET UP EMITTER IMPORT SETTINGS
    lClic(pos["Import.cp_pos_2"])
    connect(pos["Shtorka.down"], pos["Shtorka.up"])  # move shtorka mid
    lClic(pos["Import.select_none"])
    lClic((pos["Import.elements_x"], pipes[pipe_choice].sphere_pos_y))
    connect(pos["Shtorka.up"], pos["Shtorka.down"])
    connect(pos["Import.geometry"], pos["Emitter.shapes"])

    lClic(pos["ExpMesh"])
    connect(pos["Shtorka.down"], pos["Shtorka.up"])
    lClic(pos["ExpMesh.filename"])
    keyboard_print("frame")
    lClic(pos["ExpMesh.format_slider"])
    lClic(pos["ExpMesh.fbx_pos"])
    lClic(pos["ExpMesh.dir"])
    export_dir = f'{EXPORT_DIR}/{pipe_choice}/{file_h.replace(".fbx", "")}/{anim.replace(".liquigen", "")}'
    print(export_dir, file=stdout)
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    keyboard_print(export_dir)
    press_key(pynput.keyboard.Key.esc)

    if not is_test:
        lClic(pos["ExpMesh.export_now"])
        exit_code, exp_time = sleep_while_not_enought(export_dir)
        print(exit_code, exp_time, file=stdout)
        if exit_code == 1:
            lClic(pos["ExpMesh.cancel"])

    time.sleep(LARGE_DELAY)
    connect(pos["Shtorka.up"], pos["Shtorka.down"])
    connect((W / 2, H / 2), pos["Shtorka.right"])
    lClic(pos["Exit"])
    lClic(pos["Exit.dont_save"])
    disable_mouse(1)
    try:
        if P1.is_alive():
            P1.terminate()
    finally:
        P1.join()

def run_mesh():
    with open("/vol1/WATER/DATASET/FOR_UNET/process.log", 'w+') as logfile:
        for pipe in os.listdir(PATH_W_PIPES):
            if pipe not in ("Flow_from_elbow", "Flow_from_elbow_side", "Flow_from_half_flance"):
                for file_h in os.listdir(f'{PATH_W_PIPES}/{pipe}'):
                    if file_h != "1.0.fbx":
                        for anim in os.listdir(f'{PATH2ANIMS}/{pipe}')[::]:
                            print(f"STARTED {f'{PATH_W_PIPES}/{pipe}/{file_h}'} with anim {anim}", file=logfile)
                            create_mesh(pipe, file_h, anim, is_test=False, stdout=logfile)
                            print(f"FINISHED {f'{PATH_W_PIPES}/{pipe}/{file_h}'} with anim {anim}", file=logfile)

def create_animation(pipe_choice, anim_i, stdout=None):

    disable_mouse(0)

    # LOAD TEMPLATE
    double_clic(pos["LiquiGen.icon"])
    time.sleep(LARGE_DELAY)
    lClic(pos["Project.open"])
    lClic(pos["Project.filename"])
    BASE_ANIM_PATH = BASE_ANIM_PATH_003 if pipes[pipe_choice].resolution == 0.03 else BASE_ANIM_PATH_006
    kuda = BASE_ANIM_PATH.replace('/', '\\')
    keyboard_print_win(f"Z:{kuda}")
    press_key(pynput.keyboard.Key.enter)
    time.sleep(LARGE_DELAY)
    press_key(pynput.keyboard.Key.space)


    # SET UP ANIMATION SEQUENCE
    connect(pos["Shtorka.left.down"], pos["Shtorka.left.mid"])
    seq = gen_sequence_speed(pipe_choice)
    for x, value in seq:
        set_anim_point((x, pos["Anim.y"]), value)

    # SET UP EMITTER DIRECTION
    connect(pos["Shtorka.right"], (W / 2, H / 4))
    lClic(pos["Emitter.shapes"])
    connect(pos["Shtorka.down"], pos["Shtorka.up.win"])

    target = gen_sequence_target(pipe_choice)
    # FOR FIXED PIDORS
    for xyz, value in target.items():
        if isinstance(value, float) or isinstance(value, int):
            lClic(pos["Emitter.TargetPos."+xyz])
            keyboard_print(str(value))
            press_key(pynput.keyboard.Key.enter)

    # FOR ANIMATED PIDORS
    lClic(pos["Emitter.TargetPos.Toggle"])
    connect(pos["Shtorka.up.win"], pos["Shtorka.down"])
    connect((W / 2, H / 4), pos["Shtorka.right"])
    for xyz, seq in target.items():
        if isinstance(seq, list):
            print(seq, file=stdout)
            for x, value in seq:
                set_anim_point((x, pos["Anim.TargetPos.y."+xyz]), value)
    connect(pos["Shtorka.left.mid"], pos["Shtorka.left.down"])

    # SAVE AS
    lClic(pos["Emitter.shapes"])
    ctrl_s()
    lClic(pos["Project.filename"])
    linux_path = f"{PATH2ANIMS}/{pipe_choice}/"
    os.makedirs(linux_path,exist_ok=True)
    kuda = PATH2ANIMS.replace('/', '\\')
    print(f"Z:{kuda}\\{pipe_choice}\\{anim_i}", file=stdout)
    keyboard_print_win(f"Z:{kuda}\\{pipe_choice}\\{anim_i}")
    press_key(pynput.keyboard.Key.enter)
    time.sleep(LARGE_DELAY)
    lClic(pos["Exit"])
    disable_mouse(1)
    time.sleep(LARGE_DELAY)

def run_animation():
    with open("/vol1/WATER/DATASET/FOR_UNET/animations.log", 'w+') as logfile:
        for i in range(ANIMATION_PER_PIPE):
            for pipe, info in pipes.items():
                print(f"STARTED {f'{pipe}/anim_{i}'}", file=logfile)
                create_animation(pipe, i, logfile)
                print(f"FINISHED {f'{pipe}/anim_{i}'}\n", file=logfile)

#run_animation()
time.sleep(60)
run_mesh()























































































