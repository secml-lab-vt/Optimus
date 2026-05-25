from time import sleep
import time
import pandas as pd
import argparse
import os

def get_next_job(loop_file):
    with open(loop_file) as f:
        con = f.read()
    spl = con.split('\n')
    ind, done, command = -1, '1', ''
    remaining = 0
    final_command = "waiting"
    crashed = False
    for i, s in enumerate(spl):
        spl2 = s.split('---')
        if(len(spl2) != 2): continue
        done, command = spl2
        if(done == "1"):
            final_command = spl2[1]
            ind = i
        elif(done == "3"):
            crashed = True
            final_command = spl2[1]
            ind = i
        elif(done == "0"):
            remaining += 1
    return ind, final_command, remaining, crashed

#loop_name = args.loop_name
#loop_file = "./jobs/" + loop_name + '.txt'

def seconds2time(s):
    h = int((s - (s % 3600)) / 3600)
    s = s - (h * 3600)
    m = int((s - (s % 60)) / 60)
    s = s - (m * 60)
    s = int(s)
    return str(h).rjust(2, '0') + ":" + str(m).rjust(2, '0') + ":" + str(s).rjust(2, '0')

current_jobs = {}
start_times = {}
job_num = {}
jobs_rem = {}
loop_crash = {}

flasher = False
while(True):
    #flasher = not flasher
    loops_active = []
    for file in os.listdir("./jobs/"):
        #print(file)
        if file.endswith(".txt"):
            loop_name = file[:-4]
            if(loop_name in ['log', 'master_list']): continue
            loops_active.append(loop_name)
            #print(os.path.join("/mydir", file))
            file_name = os.path.join("./jobs", file)
            num, job, rem, crash = get_next_job(file_name)

            if((loop_name not in current_jobs) or 
                (current_jobs[loop_name] != job) or 
                (job_num[loop_name] != num)):
                current_jobs[loop_name] = job
                job_num[loop_name] = num
                start_times[loop_name] = time.perf_counter()
                loop_crash[loop_name] = False
            loop_crash[loop_name] = crash
            jobs_rem[loop_name] = str(rem)

    os.system("clear")
    print("loop_name".ljust(15, " ") + "remaining".ljust(15, " ") + "time".ljust(15, " ") + "command".ljust(15, " ") + "\n")
    for loop in loops_active:
        time_str = seconds2time(time.perf_counter() - start_times[loop])
        if((loop_crash[loop] == False) or flasher):
            remaining = jobs_rem[loop]
        else:
            remaining = "Error"
        print(loop.ljust(15, " ") + remaining.ljust(15, " ") + time_str.ljust(15, " ") + current_jobs[loop].ljust(15, " ") + "\n")
    sleep(1)
