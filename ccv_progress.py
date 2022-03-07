#!/usr/bin/env python3

from alive_progress import alive_bar
import time, os

mesh_count = 111
total_milestones = 9 * mesh_count**2
milestone_foler = '/users/jlizaraz/scratch/out/qdef/tsk/'

with alive_bar(total_milestones, manual=True) as bar:
    bar(0)
    while True:
        prog = len(os.listdir(milestone_foler))/total_milestones
        bar(prog)
        time.sleep(1)
        if prog == 1:
            break
