print('loss_meta...')
import sys
saved_stdout = sys.stdout
with open('temp.txt', 'w') as f:
    sys.stdout = f
    from torchWork import *
    sys.stdout = saved_stdout
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
