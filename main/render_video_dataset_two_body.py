from os import path
from threading import Thread, Lock

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
try:
    from interactive import listen
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    if module_name in (
        'interactive', 
    ):
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib')
        input('Press Enter to quit...')
    raise e

from shared import *
from physics_shared import *
from render_video_dataset_shared import *
from physics_two_body import *
from template_two_body import SEQ_LEN

PATH = path.join(
    '../datasets/two_body', 
    # '../datasets/two_body_no_orbit', 
    'train', 
    # 'validate', 
)

CENTER_OF_MASS_STATIONARY = False
REJECTABLE_START = np.inf
# REJECTABLE_START = 7

SPF = .1
DT = 4

# RUNNING_MODE = MODE_LOCATE
# RUNNING_MODE = MODE_OBV_ONLY
RUNNING_MODE = MODE_MAKE_IMG

EYE = np.array([0.0, -11.5, 0.0])  # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 0.0, 0.0])  # 瞄准方向的参考点（默认在坐标原点）

class TwoBodyViewer(BallViewer):
    def __init__(self, running_mode, ball_radius, eye, look_at, SPF) -> None:
        self.lock = Lock()
        self.do_loop_one_traj = False
        self.trajs = []
        self.traj_i = -1
        super().__init__(running_mode, SEQ_LEN, ball_radius, eye, look_at, SPF)
    
    def loop(self):
        with self.lock:
            return super().loop()
    
    def getTrajectory(self):
        if self.do_loop_one_traj:
            return self.trajs[self.traj_i]
        else:
            self.traj_i += 1
            print(f'{self.traj_i = }')
            try:
                trajectory = self.trajs[self.traj_i]
            except IndexError:
                assert self.traj_i == len(self.trajs)
                trajectory, _ = oneLegalRun(
                    DT, SEQ_LEN, 
                    CENTER_OF_MASS_STATIONARY, REJECTABLE_START, 
                    EYE, 
                )
                self.trajs.append(trajectory)
            return trajectory

    def locate_with_ball(self):
        radius = VIEW_RADIUS
        center = (0, 0, 0)

        self.makeBall(*center, r=radius)

def ui(viewer: TwoBodyViewer):
    def h():
        print('''
L: loop one traj. 
T: trace. 
P: prev. 
S: slower. 
F: faster.
H: show this help.
''')
    h()
    while True:
        op = listen(b'ltpsfh')
        if op == b'l':
            viewer.do_loop_one_traj ^= True
            print(f'{viewer.do_loop_one_traj = }')
        if op == b't':
            viewer.leave_trace ^= True
            print(f'{viewer.leave_trace = }')
        if op == b'p':
            with viewer.lock:
                viewer.traj_i -= 1
                viewer.traj_i = max(0, viewer.traj_i)
                print(f'{viewer.traj_i = }')
                if not viewer.do_loop_one_traj:
                    viewer.traj_i -= 1
                viewer.reset()
        if op == b's':
            viewer.SPF *= 2
            print(f'{viewer.SPF = }')
        if op == b'f':
            viewer.SPF /= 2
            print(f'{viewer.SPF = }')
        if op == b'h':
            h()

def main():
    os.makedirs(PATH, exist_ok=True)
    os.chdir(PATH)
    viewer = TwoBodyViewer(
        RUNNING_MODE, BALL_RADIUS, EYE, LOOK_AT, SPF, 
    )
    thread = Thread(target=ui, args=(viewer, ))
    thread.start()
    glutMainLoop()

if __name__ == "__main__":
    main()
