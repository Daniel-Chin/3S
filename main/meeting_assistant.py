import os
from os import path
import subprocess as sp
from typing import List
from threading import Thread
import platform

import evalDecoder
import evalEncoder

def main():
    while True:
        exp_dirname = input('experiment dirname: ')
        if not exp_dirname:
            break
        exp_path = path.join('./experiments', exp_dirname)
        pdf_search = [
            ['plots.pdf', 'plot.pdf', 'auto_plot_loss.pdf'], 
            ['eval_encoder.pdf', 'auto_eval_encoder.pdf'], 
        ]
        pdfs = []
        for candidates in pdf_search:
            for pdf in candidates:
                pdf = path.join(exp_path, pdf)
                if path.isfile(pdf):
                    pdfs.append(pdf)
                    break
            else:
                pdfs.append(None)
                print('Warn: pdf not found. ')
        for pdf in pdfs:
            if pdf is None:
                continue
            if platform.system() == 'Darwin':       # macOS
                sp.call(('open', pdf))
            elif platform.system() == 'Windows':    # Windows
                os.startfile(pdf)
            else:                                   # linux variants
                sp.call(('xdg-open', pdf))

        assert pdf_search[1][0] == 'eval_encoder.pdf'
        if pdf[1] is None:
            thread = Thread(target=evalEncoder.main, args=(
                exp_path, None, 
            ))
            # thread.start()
            thread.run()
        try:
            evalDecoder.main(exp_path, None)
        except KeyboardInterrupt:
            print('Interrupted.')
            pass
        # thread.join()
        # print('thread joined.')
    print('bye')

main()
