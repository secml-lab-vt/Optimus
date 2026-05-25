from time import sleep
import time
import argparse
import os

def find_next_job():
    with open(loop_file) as f:
        con = f.read()
    spl = con.split('\n')
    ind, done, command = -1, '1', ''
    for i, s in enumerate(spl):
        spl2 = s.split('---')
        if(len(spl2) != 2): continue
        done, command = spl2
        ind = i
        if(done == '0'): break
        if(done == '3'): break
    return ind, command, done

def do_next_job(ind, command):
    with open(loop_file) as f:
        con = f.read()
    spl = con.split('\n')
    spl[ind] = '1---' + command
    f = open(loop_file, 'w+')
    f.write('\n'.join(spl))
    f.close()
    print("Doing: " + command)
    start_time = time.perf_counter()
    res = os.system(command)
    end_time = time.perf_counter()
    total_time = end_time - start_time

    if(res == 0): #Normal exit code
        return "2", total_time
    else: #Some other exit code
        print(f"\nThe job '{command}' has crashed!!!", end="")
        return "3", total_time

def save_to_queue(command, ind, state):
    with open(loop_file) as f:
        con = f.read()
    spl = con.split('\n')

    assert(command in spl[ind])
    spl[ind] = f'{state}---' + command
    f = open(loop_file, 'w+')
    f.write('\n'.join(spl))
    f.close()

def seconds2time(s):
    h = int((s - (s % 3600)) / 3600)
    s = s - (h * 3600)
    m = int((s - (s % 60)) / 60)
    s = s - (m * 60)
    s = int(s)
    return str(h).rjust(2, '0') + ":" + str(m).rjust(2, '0') + ":" + str(s).rjust(2, '0')

def save_to_log(command, total_time, res):
    with open(log_file, "a+") as f:
        f.write(f"{loop_name}---{res_to_string[res]}---{seconds2time(total_time)}---{command}\n")

parser = argparse.ArgumentParser(description='Run a loop')
parser.add_argument('loop_name', help='loop name')

args = parser.parse_args()

res_to_string = {"2":"Success", "3":"Failure"}
loop_name = args.loop_name
jobs_path = "./jobs/"
loop_file = jobs_path + loop_name + '.txt'
log_file = jobs_path + "log.txt"

if not os.path.exists(jobs_path):
   os.makedirs(jobs_path)
if(os.path.exists(loop_file) == False):
    open(loop_file, 'w+')

print("Starting loop: " + loop_name, end="")
waiting = False
while(True):
    ind, command, done = find_next_job()
    if(done == "0"): #Job found
        save_to_queue(command, ind, 1)
        res, total_time = do_next_job(ind, command)
        save_to_queue(command, ind, res)
        save_to_log(command, total_time, res)

        waiting = False
    else:
        if(waiting == False):
            print(f"\n\n{loop_name} is waiting for job!")
            print(f"Enter job into '{jobs_path}{loop_name}.txt'")
            print("\tExamples: '0---echo hello', '0---python helloWorld.py'")
            waiting = True
        else:
            sleep(5)
    
