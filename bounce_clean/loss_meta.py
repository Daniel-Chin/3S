from torchWork import *
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
