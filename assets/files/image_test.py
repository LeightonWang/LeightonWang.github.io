import airsim
import cv2
import numpy as np
import os
import pprint
# import setup_path 
import tempfile
import time
from KCF import Tracker, MessageItem
from PID import PIDController
from matplotlib import pyplot as plt

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, "Drone1")
client.enableApiControl(True, "Drone2")
client.armDisarm(True, "Drone1")
client.armDisarm(True, "Drone2")

# Takeoff
f1 = client.takeoffAsync(vehicle_name="Drone1")
f2 = client.takeoffAsync(vehicle_name="Drone2")
f1.join()
f2.join()

f1 = client.moveToZAsync(-6, 5, vehicle_name="Drone1")
f2 = client.moveToZAsync(-2, 5, vehicle_name="Drone2")
f1.join()
f2.join()

client.moveByVelocityBodyFrameAsync(0.5, 0, 0, 4, vehicle_name="Drone2").join()

imgs = client.simGetImages([
    airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)], vehicle_name = "Drone1") 
img1d = np.fromstring(imgs[0].image_data_uint8, dtype=np.uint8) # get numpy array
gFrame = img1d.reshape(imgs[0].height, imgs[0].width, 3) # reshape array to 3 channel image array H * W * 3

print(imgs[0].height, imgs[0].width)
    
# 框选感兴趣区域region of interest
# cv2.destroyWindow("pick frame")
gROI = cv2.selectROI("ROI frame",gFrame,False)
if (not gROI):
    print("空框选，退出")
    quit()

# 初始化追踪器
gTracker = Tracker(tracker_type="KCF")
gTracker.initWorking(gFrame,gROI)

vx_controller = PIDController(0.02)
vy_controller = PIDController(0)

VXs = []

# 循环帧读取，开始跟踪
while True:
    # gCapStatus, gFrame = gVideoDevice.read()
    imgs = client.simGetImages([
        airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)], vehicle_name = "Drone1") 
    img1d = np.fromstring(imgs[0].image_data_uint8, dtype=np.uint8) # get numpy array
    gFrame = img1d.reshape(imgs[0].height, imgs[0].width, 3) # reshape array to 3 channel image array H * W * 3
    if(1):
        # 展示跟踪图片
        _item = gTracker.track(gFrame)
        cv2.imshow("track result", _item.getFrame())

        if _item.getMessage():
            # 打印跟踪数据
            # print(_item.getMessage())
            msg = _item.getMessage()
            lt = msg['coord'][0][0]
            rb = msg['coord'][0][1]
            target_center = ((lt[0] + rb[0]) / 2, (lt[1] + rb[1]) / 2)
            print(target_center)
        else:
            # 丢失，重新用初始ROI初始
            print("丢失，重新使用初始ROI开始")
            gTracker = Tracker(tracker_type="KCF")
            gTracker.initWorking(gFrame, gROI)

        _key = cv2.waitKey(1) & 0xFF
        if (_key == ord('q')) | (_key == 27):
            break
        if (_key == ord('r')) :
            # 用户请求用初始ROI
            print("用户请求用初始ROI")
            gTracker = Tracker(tracker_type="KCF")
            gTracker.initWorking(gFrame, gROI)
        
    else:
        print("捕获帧失败")
        quit()
    
    error = (target_center[0] - 128, target_center[1] - 72)
    vx = vx_controller.update(error[1])
    vy = vy_controller.update(error[0])
    VXs.append(vx)
    print("Error: {}, V:{}".format(error, (vx, vy)))
    f1 = client.moveByVelocityBodyFrameAsync(vx, vy, 0, 0.05)

    f2 = client.moveByVelocityBodyFrameAsync(0.5, 0, 0, 0.05, vehicle_name="Drone2")
    
    f1.join()
    f2.join()


f1, f2 = client.landAsync(3, vehicle_name="Drone1"), client.landAsync(3, vehicle_name="Drone2")
f1.join()
f2.join()

client.armDisarm(False, "Drone1")
client.armDisarm(False, "Drone2")

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False, "Drone1")
client.enableApiControl(False, "Drone2")

N = len(VXs)
t_xlabel = [i * 0.05 for i in range(N)]
REFs = [0.5] * len(VXs)
plt.plot(t_xlabel, VXs, label="vx of the tracer")
plt.plot(t_xlabel, REFs, label="vx of the traced")
plt.xlabel('t')
plt.ylabel('v')

plt.legend()
plt.show()