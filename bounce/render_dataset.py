import time
from typing import List
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
from PIL import Image
from physics import *

TRAIN_PATH    = './dataset/train'
VALIDATE_PATH = './dataset/validate'

PATH = TRAIN_PATH
# PATH = VALIDATE_PATH

# GIF_INTERVAL = 200

WIN_W = 320
WIN_H = 320
RESOLUTION = 32
RESOLUTION = 32
SPF = .2
DT = .15
SEQ_LEN = 20
DRAW_GIRD = True

MODE_LOCATE = 'locate'
MODE_OBV_ONLY = 'obv_only'
MODE_MAKE_IMG = 'make_img'

# RUNNING_MODE = MODE_LOCATE
# RUNNING_MODE = MODE_OBV_ONLY
RUNNING_MODE = MODE_MAKE_IMG

VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 15.0])  # 视景体的left/right/bottom/top/near/far六个面
SCALE_K = np.array([1.0, 1.0, 1.0])  # 模型缩放比例
EYE = np.array([0.0, -2.0, 4.0])  # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 10.0, 0.0])  # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 0.0, 1.0])  # 定义对观察者而言的上方（默认y轴的正方向）

class BallViewer:
    def __init__(self) -> None:
        self.trajectory: List[List[Body]] = None
        self.stage = None
        self.output_i = 0
        self.render_or_screenshot = 0
        self.frames = None
        self.reset()
    
    def reset(self):
        self.trajectory = oneLegalRun(DT, SEQ_LEN)
        self.stage = 0
        self.frames = []

    def loop(self):
        if self.render_or_screenshot == 0:
            if RUNNING_MODE is MODE_LOCATE:
                locate_with_ball()
                return
            
            if self.stage >= SEQ_LEN:
                if RUNNING_MODE is MODE_MAKE_IMG:
                    self.saveVideo()
                self.reset()
                self.output_i += 1
            
            body: Body = self.trajectory[self.stage]

            # render
            makeBall(*body.position, BALL_RADIUS)
            
            # step
            self.stage += 1
            if RUNNING_MODE is MODE_OBV_ONLY:
                time.sleep(SPF)
        else:
            body: Body = self.trajectory[self.stage - 1]
            makeBall(*body.position, BALL_RADIUS)
            if RUNNING_MODE is MODE_MAKE_IMG:
                self.frames.append(screenShot())
        
        self.render_or_screenshot = 1 - self.render_or_screenshot
    
    def saveVideo(self):
        # self.frames[0].save(
        #     f'{self.output_i}.gif', save_all=True, 
        #     append_images=self.frames[1:], 
        #     duration=GIF_INTERVAL, loop=0, 
        # )
        # GIF image compression is uncontrollable
        os.makedirs(str(self.output_i), exist_ok=True)
        os.chdir(str(self.output_i))
        for i, img in enumerate(self.frames):
            img.save(f'{i}.png')
        os.chdir('..')

def makeBall(x, y, z, r=1, color3f=(0, 1, 0)):
    glPushMatrix()
    glColor3f(* color3f)
    glTranslatef(x, y, z)
    quad = gluNewQuadric()
    gluSphere(quad, r, 90, 90)
    gluDeleteQuadric(quad)
    glPopMatrix()

def locate_with_ball():
    radius = 4
    height = 4
    center = (0, 6, 0)

    # for x in (-1, 1):
    #     for y in (-1, 1):
    #         for z in (-1, 1):
    #             makeBall(
    #                 x * radius, 4 + y * radius, z * radius, 
    #             )

    # makeBall(0, 8, 0, r=.5)
    # makeBall(*center, r=radius)

    for theta in np.linspace(0, 2*np.pi, 36):
        for z in (0, height):
            makeBall(
                center[0] + np.cos(theta) * radius, 
                center[1] + np.sin(theta) * radius, 
                z, 
                0.5
            )

def drawGrid():
    glLineWidth(3)
    glBegin(GL_LINES)
    glColor4f(0.0, 0.0, 0.0, 1)  # 设置当前颜色为黑色不透明
    for i in np.linspace(-15, 15, 19):
        glVertex3f(i, -100, 0)
        glVertex3f(i, +100, 0)
        glVertex3f(-100, i, 0)
        glVertex3f(+100, i, 0)
    glEnd()

def init():
    glutInit()
    displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
    glutInitDisplayMode(displayMode)

    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 50)
    glutCreateWindow('Ball Throwing Simulation')

    # 初始化画布
    glClearColor(0.4, 0.4, 0.4, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
    glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
    glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）-
    glEnable(GL_LIGHT0)  # 启用0号光源
    glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(0, 1, 4, 0))  # 设置光源的位置
    glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, GLfloat_3(0, 0, -1))  # 设置光源的照射方向
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)  # 设置材质颜色
    glEnable(GL_COLOR_MATERIAL)
    
    glutDisplayFunc(draw)  # 注册回调函数draw()
    glutIdleFunc(draw)
    glutReshapeFunc(reshape)  # 注册响应窗口改变的函数reshape()
    # glutMouseFunc(mouseclick)  # 注册响应鼠标点击的函数mouseclick()
    # glutMotionFunc(mousemotion)  # 注册响应鼠标拖拽的函数mousemotion()
    # glutKeyboardFunc(keydown)  # 注册键盘输入的函数keydown()

def draw():
    initRender()
    glEnable(GL_LIGHTING)  # 启动光照
    if DRAW_GIRD:
        drawGrid()
    ballViewer.loop()
    glDisable(GL_LIGHTING)  # 每次渲染后复位光照状态

    # 把数据刷新到显存上
    glFlush()
    glutSwapBuffers()  # 切换缓冲区，以显示绘制内容

def initRender():
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
        * EYE,
        * LOOK_AT, 
        * EYE_UP, 
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

def main():
    global ballViewer
    os.makedirs(PATH, exist_ok=True)
    os.chdir(PATH)
    init()
    ballViewer = BallViewer()
    glutMainLoop()  # 进入glut主循环

if __name__ == "__main__":
    main()
