from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# 建立模型
model = Sequential()
model.add(Conv2D(32, 3, activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# 编译
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
# 从图片中直接产生数据和标签
train_generator = train_datagen.flow_from_directory('C:\\Users\\ailab\\PycharmProjects\\drone\\train',
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory('C:\\Users\\ailab\\PycharmProjects\\drone\\validation',
                                                        target_size=(150,150),
                                                        batch_size=32,
                                                        class_mode='binary')
model.fit_generator(train_generator,
                    steps_per_epoch=1000,
                    epochs=3,
                    validation_data=validation_generator,
                    validation_steps=200
                    )
# 保存整个模型
model.save('C:\\Users\\ailab\\PycharmProjects\\drone\\models\\col.h5')
'''
# 保存模型的权重
model.save_weights('C:\\Users\\ailab\\PycharmProjects\\drone\\models\\col_weights.hdf5')

# 保存模型的结构
json_string = model.to_json()
open('C:\\Users\\ailab\\PycharmProjects\\drone\\models\\model_to_json.json','w').write(json_string)
yaml_string = model.to_yaml()
open('C:\\Users\\ailab\\PycharmProjects\\drone\\models\\model_to_yaml.yaml','w').write(json_string)
'''