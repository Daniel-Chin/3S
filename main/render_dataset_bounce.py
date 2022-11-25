from os import path

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from shared import *
from physics_shared import *
from render_dataset_shared import *
from physics_bounce import *
from template_bounce import SEQ_LEN

PATH = path.join(
    '../datasets/bounce', 
    # '../datasets/bounce_leave_view', 
    'train', 
    # 'validate', 
)

# REJECTABLE_START = 6
REJECTABLE_START = np.inf

SPF = .2
DT = .15

# RUNNING_MODE = MODE_LOCATE
# RUNNING_MODE = MODE_OBV_ONLY
RUNNING_MODE = MODE_MAKE_IMG

EYE = np.array([0.0, -2.0, 4.0])  # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 10.0, 0.0])  # 瞄准方向的参考点（默认在坐标原点）

class BounceViewer(BallViewer):
    def getTrajectory(self):
        return oneLegalRun(DT, self.SEQ_LEN, REJECTABLE_START)

    def locate_with_ball(self):
        radius = 4
        height = 4
        center = (0, 6, 0)

        # for x in (-1, 1):
        #     for y in (-1, 1):
        #         for z in (-1, 1):
        #             self.makeBall(
        #                 x * radius, 4 + y * radius, z * radius, 
        #             )

        # self.makeBall(0, 8, 0, r=.5)
        # self.makeBall(*center, r=radius)

        for theta in np.linspace(0, 2*np.pi, 36):
            for z in (0, height):
                self.makeBall(
                    center[0] + np.cos(theta) * radius, 
                    center[1] + np.sin(theta) * radius, 
                    z, 
                    0.5
                )

def main():
    os.makedirs(PATH, exist_ok=True)
    os.chdir(PATH)
    viewer = BounceViewer(
        RUNNING_MODE, SEQ_LEN, BALL_RADIUS, EYE, LOOK_AT, SPF, 
    )
    glutMainLoop()  # 进入glut主循环

if __name__ == "__main__":
    main()
