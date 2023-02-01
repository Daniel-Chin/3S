The entire env takes about 15G with ubuntu sigularity.  
Therefore, you may want to allocate ~25G space.  

- pyTorch
  - We use pytorch 1.13.1 with cuda 11.6
  - For latest, https://pytorch.org/get-started/

```bash
pip install -U numpy
pip install -U scipy
pip install -U tabulate
pip install -U GitPython
pip install -U matplotlib
pip install -U opencv-python
pip install -U pynvml
pip install -U librosa
pip install -U pretty_midi
pip install -U music21
```

For xj Repo, you need additional  
```
pip install -U scikit-learn
```

- `torchWork` from https://github.com/Daniel-Chin/torchWork
  - I didn't make it installable. Manually add the parent dir of `torchWork` to `$PYTHONPATH`. 
- `midi2audio` from https://github.com/daniel-chin/midi2audio
  - I didn't make it installable. Manually add the parent dir of `torchWork` to `$PYTHONPATH`, and rename the package as `midi2audio_fork`.  
- `python_lib` from https://github.com/daniel-chin/python_lib
  - I didn't make it installable. Manually add the entire `python_lib` to `$PYTHONPATH`. 
- `ffmpeg`. 
  - Usually conda installs a minimal ffmpeg. Maybe opencv-python required it. 
    - `which ffmpeg`. The minimal version is probably in `/ext3/miniconda3/bin/ffmpeg`. 
    - However, we need the full version. 
  - So, install the full version. 
    - https://launchpad.net/ubuntu/+source/ffmpeg
    - If you are working on NYU hpc, you are prolly looking for: ubuntu "focal" -> latest -> (stable) release -> AMD64 build
      - https://launchpad.net/ubuntu/focal/amd64/ffmpeg/7:4.2.2-1ubuntu1
    - Download the `.deb` file. Use `dpkg -x` to extract the standalone. 
  - Then, make sure the full version is found before the conda version. 
    - Place the standalone full ffmpeg in `$PATH` before everything else. 
      - e.g., if you are using Singularity, alter /ext3/env.sh to prepend $PATH with the ffmpeg dir. 
  - Verify with `which ffmpeg`  
