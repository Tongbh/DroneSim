import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import cv2
from PIL import Image
a=[36,53,59,17,120,126,176,154,119,271,269,271,238,180,220]
b=np.zeros(15)
for i in range(15):
    b[i]=np.array(480-a[i])
c=[236,322,433,531,204,289,363,455,511,153,258,401,518,572,621]

plt.scatter([236,322,433,531,204,289,363,455,511,153,258,401,518,572,621],b)
plt.xlim((0.640))
plt.ylim((0,480))
plt.savefig('points.png')
plt.show()


'''

points = np.array([[236,322,433,531,204,289,363,455,511,153,258,401,518,572,621],
                  [444,427,421,463,360,354,304,326,361,209,211,209,242,300,260]])
points=points.transpose()

vor=Voronoi(points)
fig=voronoi_plot_2d(vor)
plt.xlim((0.640))
plt.ylim((0,480))
plt.savefig('map.png')
plt.show()

'''
#print(vor.ridge_dict)

#img=cv2.imread('map.png')
#cv2.imshow('1',img)
#cv2.waitKey(0)
#set_point=np.array([255,127,14])

