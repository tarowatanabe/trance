#!/usr/bin/env python

import sys
import os
import os.path

import re

groups=[]
for arg in sys.argv[1:]:
    span = map(int, arg.split('-'))
    if len(span) != 2:
        raise ValueError, "invlaid arg: " + arg
    
    groups.append(tuple(span))

if not groups:
    raise ValueError, "no spans"


dir='orig/'
pattern=re.compile(r"chtb_([0-9]*).*")


for root,dirs,files in os.walk(dir):
    
    for file in files:
        result = pattern.match(file)
        if not result: continue
        id = int(result.group(1))
        
        for group in groups:
            if group[0] <= id and id <= group[1]:
                print os.path.join(root, file)
