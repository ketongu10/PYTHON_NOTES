from pynput.mouse import Button as mButton
from pynput.mouse import Controller as mCtrl
from time import sleep

mouse = mCtrl()

while True:
    print(mouse.position, end="\r")
    sleep(1)