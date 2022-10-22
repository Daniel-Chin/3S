import time
from typing import List
import json
from abc import ABCMeta, abstractmethod

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
from PIL import Image

from shared import *
from physics_shared import *

__all__ = [
    'MODE_LOCATE', 'MODE_OBV_ONLY', 'MODE_MAKE_IMG', 
    'BallViewer', 
]

WIN_W = 320
WIN_H = 320
BACKGROUND = .6
DRAW_GIRD = False

VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 100.0])  # 视景体的left/right/bottom/top/near/far六个面
SCALE_K = np.array([1.0, 1.0, 1.0])  # 模型缩放比例
EYE_UP = np.array([0.0, 0.0, 1.0])  # 定义对观察者而言的上方（默认y轴的正方向）

MODE_LOCATE = 'locate'
MODE_OBV_ONLY = 'obv_only'
MODE_MAKE_IMG = 'make_img'

class BallViewer(metaclass=ABCMeta):
    def __init__(
        self, running_mode, ball_radius, 
        eye, look_at, SPF, 
    ) -> None:
        self.running_mode = running_mode
        self.ball_radius = ball_radius
        self.eye = eye
        self.look_at = look_at
        self.SPF = SPF
        self.trajectory: List[List[Body]] = None
        self.stage: int = None
        self.output_i = 0
        self.render_or_screenshot = 0
        self.frames: List[Image.Image] = None
        self.reset()
        self.initGlut()
    
    def reset(self):
        self.trajectory = self.getTrajectory()
        self.stage = 0
        self.frames = []
    
    @abstractmethod
    def getTrajectory(self) -> List[List[Body]]:
        raise NotImplemented

    def loop(self):
        if self.render_or_screenshot == 0:
            if self.running_mode is MODE_LOCATE:
                self.locate_with_ball()
                return
            
            if self.stage >= SEQ_LEN:
                if self.running_mode is MODE_MAKE_IMG:
                    self.saveVideo()
                self.reset()
                self.output_i += 1
            
            bodies = self.trajectory[self.stage]

            self.makeBodies(*bodies)
            
            # step
            self.stage += 1
            if self.running_mode is MODE_OBV_ONLY:
                time.sleep(self.SPF)
        else:
            bodies = self.trajectory[self.stage - 1]
            self.makeBodies(*bodies)
            if self.running_mode is MODE_MAKE_IMG:
                self.frames.append(screenShot())
        
        self.render_or_screenshot = 1 - self.render_or_screenshot
    
    def makeBall(self, x, y, z, r=1, color3f=(0., 1., 0.)):
        glPushMatrix()
        glColor3f(* color3f)
        glTranslatef(x, y, z)
        quad = gluNewQuadric()
        gluSphere(quad, r, 90, 90)
        gluDeleteQuadric(quad)
        glPopMatrix()
    
    def makeBodies(self, *bodies: Body):
        for body, color in zip(bodies, (
            (0., 1., 0.), 
            (1., 0., 1.), 
        )):
            self.makeBall(*body.position, self.ball_radius, color)

    def saveVideo(self):
        # self.frames[0].save(
        #     f'{self.output_i}.gif', save_all=True, 
        #     append_images=self.frames[1:], 
        #     duration=GIF_INTERVAL, loop=0, 
        # )
        # GIF image compression is uncontrollable
        os.makedirs(str(self.output_i), exist_ok=True)
        os.chdir(str(self.output_i))
        with open(TRAJ_FILENAME, 'w') as f:
            trajectory_json = []
            for bodies in self.trajectory:
                bodies_json = []
                trajectory_json.append(bodies_json)
                for body in bodies:
                    bodies_json.append(body.toJSON())
            json.dump(trajectory_json, f)
        for i, img in enumerate(self.frames):
            img.save(f'{i}.png')
        os.chdir('..')

    @abstractmethod
    def locate_with_ball(self):
        raise NotImplemented

    def drawGrid(self):
        glLineWidth(2)
        luminosity = .7
        glColor3f(luminosity, luminosity, luminosity)
        for x in np.linspace(-7.3, 6.81, 4):
            for y in np.linspace(-6.53, 7.03, 4):
                glBegin(GL_LINES)
                glVertex3f(x, y, -100)
                glVertex3f(x, y, +100)
                glEnd()
                glBegin(GL_LINES)
                glVertex3f(x, -100, y)
                glVertex3f(x, +100, y)
                glEnd()
                glBegin(GL_LINES)
                glVertex3f(-100, x, y)
                glVertex3f(+100, x, y)
                glEnd()

    def initGlut(self):
        glutInit()
        displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
        glutInitDisplayMode(displayMode)

        glutInitWindowSize(WIN_W, WIN_H)
        glutInitWindowPosition(300, 50)
        glutCreateWindow('Ball Throwing Simulation')

        # 初始化画布
        glClearColor(BACKGROUND, BACKGROUND, BACKGROUND, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
        glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
        glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）-
        glEnable(GL_LIGHT0)  # 启用0号光源
        glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(0, 1, 4, 0))  # 设置光源的位置
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, GLfloat_3(0, 0, -1))  # 设置光源的照射方向
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)  # 设置材质颜色
        glEnable(GL_COLOR_MATERIAL)
        
        glutDisplayFunc(self.draw)  # 注册回调函数draw()
        glutIdleFunc(self.draw)
        glutReshapeFunc(reshape)  # 注册响应窗口改变的函数reshape()
        # glutMouseFunc(mouseclick)  # 注册响应鼠标点击的函数mouseclick()
        # glutMotionFunc(mousemotion)  # 注册响应鼠标拖拽的函数mousemotion()
        # glutKeyboardFunc(keydown)  # 注册键盘输入的函数keydown()

    def draw(self):
        self.initRender()
        if DRAW_GIRD:
            self.drawGrid()
        glEnable(GL_LIGHTING)  # 启动光照
        self.loop()
        glDisable(GL_LIGHTING)  # 每次渲染后复位光照状态

        # 把数据刷新到显存上
        glFlush()
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容

    def initRender(self):
        # 清除屏幕及深度缓存
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 设置投影（透视投影）
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        glFrustum(
            VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W,
            VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5], 
        )

        # 设置模型视图
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 几何变换
        glScale(* SCALE_K)

        # 设置视点
        gluLookAt(
            *self.eye,
            *self.look_at, 
            *EYE_UP, 
        )

        # 设置视口
        glViewport(0, 0, WIN_W, WIN_H)

def screenShot():
    glReadBuffer(GL_FRONT)
    # 从缓冲区中的读出的数据是字节数组
    data = glReadPixels(0, 0, WIN_W, WIN_H, GL_RGB, GL_UNSIGNED_BYTE)
    arr = np.zeros((WIN_W * WIN_H * 3), dtype=np.uint8)
    for i in range(0, len(data), 3):
        # 由于opencv中使用的是BGR而opengl使用的是RGB所以arr[i] = data[i+2]，而不是arr[i] = data[i]
        arr[i] = data[i + 2]
        arr[i + 1] = data[i + 1]
        arr[i + 2] = data[i]
    arr = np.reshape(arr, (WIN_H, WIN_W, 3))
    # 因为opengl和OpenCV在Y轴上是颠倒的，所以要进行垂直翻转，可以查看cv2.flip函数
    cv2.flip(arr, 0, arr)
    resized = cv2.resize(arr, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)

def reshape(width, height):
    global WIN_W, WIN_H
    WIN_W, WIN_H = width, height
    glutPostRedisplay()
