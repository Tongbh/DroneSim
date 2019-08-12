import pyrealsense2 as rs
import numpy as np
import cv2

np.set_printoptions(threshold=np.inf)
# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
pipe_profile = pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    img_color = np.asanyarray(color_frame.get_data())
    img_depth = np.asanyarray(depth_frame.get_data())

    for i in range(640):
        for j in range(480):
            if depth_frame.get_distance(i,j)>0.5:
                img_color[j][i]=0

    # cv2.circle(img_color, (320,240), 8, [255,0,255], thickness=-1)
    # cv2.putText(img_color,"Dis:"+str(img_depth[320,240]), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])

    cv2.imshow('depth_frame',img_color)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

pipeline.stop()

