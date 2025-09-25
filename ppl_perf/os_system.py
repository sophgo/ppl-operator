import logging
import argparse
import subprocess
import os
import time
import shlex
import signal
from subprocess import Popen, PIPE

def _os_subprocess(cmd: list, time_out: int = 0):
    cmd_str = ""
    for s in cmd:
        cmd_str += str(s) + " "
    print("[Running]: {}".format(cmd_str))
    process = Popen(shlex.split(cmd_str), stdout=PIPE, stderr=subprocess.STDOUT)
    st = time.time()
    info = ""
    while True:
        output = process.stdout.readline().rstrip().decode('utf-8')
        if output == '' and process.poll() is not None:
            break
        if output:
            info += output.strip() + "\n"

        if time_out > 0 and time.time() - st > time_out:
            os.kill(process.pid, signal.SIGTERM)
            print("[!Warning:TimeOut]: {}".format(cmd_str))
            return 2 , "[!Warning:TimeOut]: {}".format(cmd_str)

    rc = process.poll()
    if rc == 0:
        print("[Success]: {}".format(cmd_str))
        info = " "
    else:
        print("[!Error]: {}".format(cmd_str))
        print(info)
        rc = 1
    return rc, info

def _os_system(cmd: list, time_out: int = 0):
    cmd_str = ""
    for s in cmd:
        cmd_str += str(s) + " "
    print("[Running]: {}".format(cmd_str))
    ret = os.system(cmd_str)
    if ret == 0:
        print("[Success]: {}".format(cmd_str))
    else:
        print("[!Error]: {}".format(cmd_str))
        return 1
    return 0
