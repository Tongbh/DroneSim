from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img

# 加载权重
model = load_model('C:\\Users\\ailab\\PycharmProjects\\drone\\models\\col.h5')

# 加载图像
img = load_img('C:\\Users\\ailab\\PycharmProjects\\drone\\1.jpg',target_size=(150, 150))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

predictions = model.predict(img)
print (predictions)