#!/usr/bin/env python

import sys

for line in sys.stdin:
    sent = []
    for word in line.split():
        sent.append(word.split('_')[-1])

    sys.stdout.write(' '.join(sent)+'\n')
        
    
