from os import path

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from shared import *
from physics_shared import *
from render_dataset_shared import *
from physics_two_body import *

PATH = path.join(
    # '../datasets/two_body', 
    '../datasets/two_body_no_orbit', 
    # 'train', 
    'validate', 
)

CENTER_OF_MASS_STATIONARY = False
# REJECTABLE_START = np.inf
REJECTABLE_START = 7

SPF = .1
DT = 2

# RUNNING_MODE = MODE_LOCATE
RUNNING_MODE = MODE_OBV_ONLY
# RUNNING_MODE = MODE_MAKE_IMG

EYE = np.array([0.0, -11.5, 0.0])  # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 0.0, 0.0])  # 瞄准方向的参考点（默认在坐标原点）

class TwoBodyViewer(BallViewer):
    def getTrajectory(self):
        trajectory, _ = oneLegalRun(
            DT, SEQ_LEN, 
            CENTER_OF_MASS_STATIONARY, REJECTABLE_START, 
            EYE, 
        )
        return trajectory

    def locate_with_ball(self):
        radius = VIEW_RADIUS
        center = (0, 0, 0)

        self.makeBall(*center, r=radius)

def main():
    os.makedirs(PATH, exist_ok=True)
    os.chdir(PATH)
    viewer = TwoBodyViewer(
        RUNNING_MODE, BALL_RADIUS, EYE, LOOK_AT, SPF, 
    )
    glutMainLoop()  # 进入glut主循环

if __name__ == "__main__":
    main()
