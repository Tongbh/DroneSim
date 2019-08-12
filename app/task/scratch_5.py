# import pyrealsense2 as rs
# import numpy as np
# import cv2
#
# np.set_printoptions(threshold=np.inf)
# # Declare pointcloud object, for calculating pointclouds and texture mappings
# pc = rs.pointcloud()
# # We want the points object to be persistent so we can display the last cloud when a frame drops
# points = rs.points()
#
#
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#
# # Start streaming
# pipe_profile = pipeline.start(config)
#
# # Create an align object
# # rs.align allows us to perform alignment of depth frames to others frames
# # The "align_to" is the stream type to which we plan to align depth frames.
# align_to = rs.stream.color
# align = rs.align(align_to)
#
#
# while True:
#     frames = pipeline.wait_for_frames()
#     aligned_frames = align.process(frames)
#     depth_frame = aligned_frames.get_depth_frame()
#     color_frame = aligned_frames.get_color_frame()
#     img_color = np.asanyarray(color_frame.get_data())
#     img_depth = np.asanyarray(depth_frame.get_data())
#     cv2.imshow('depth_frame',img_color)
#     key = cv2.waitKey(1)
#     if key & 0xFF == ord('q'):
#         break

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # cv2.imshow('1',depth_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #print(depth_frame.get_distance(320, 240))
        # Stack both images horizontally
        #cv2.circle(color_image, (320, 240), 8, [0, 0, 0], thickness=-1)
        dots = np.array([[310,230],[310,250],[330,250],[330,230]])
        cv2.fillPoly(depth_colormap,[dots],(0,0,0))
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        dis = 0
        count = 0
        for i in range(21):
            for j in range(21):
                    a = depth_frame.get_distance(310+i,230+j)
                    if a -depth_frame.get_distance(320,240)>1:
                        continue

                    dis = dis +depth_frame.get_distance(310+i,230+j)
                    count = count+1
        dis = dis/count
        print(dis)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


finally:

    # Stop streaming
    pipeline.stop()