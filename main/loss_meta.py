print('loss_meta...')
import sys
import os
saved_stdout = sys.stdout
TEMP = 'temp.txt'
with open(TEMP, 'w') as f:
    sys.stdout = f
    from torchWork import *
    sys.stdout = saved_stdout
try:
    os.remove(TEMP)
except FileNotFoundError:
    pass    # if multiple processes are running

AbstractLossNode = loss_tree.AbstractLossNode

def main():
    absLossRoot = AbstractLossNode('loss_root', [
        'self_recon', 
        'kld', 
        AbstractLossNode('predict', [
            'z', 
            'image', 
        ]), 
        AbstractLossNode('supervise', [
            'rnn', 
            AbstractLossNode('vae', [
                'encode', 
                'decode', 
            ]), 
        ]), 
    ])
    with open('losses.py', 'w') as f:
        loss_tree.writeCode(f, absLossRoot)

main()
