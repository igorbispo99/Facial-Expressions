import keras_facenet

base_model = keras_facenet.FaceNet().model

from keras.models import Model, Sequential, load_model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

def get_model():
    x = base_model.output
    out = Dense(7, activation = 'softmax')(x)

    model = Model(inputs = base_model.input, outputs = out )

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ["acc"])

    return model

def fit(folder, batch_size):
    datagen = ImageDataGenerator(horizontal_flip=True,\
        validation_split = 0.1, rescale=1/255, rotation_range=65,
        zoom_range=[0.5, 1], height_shift_range=0.2)
    
    train_gen = datagen.flow_from_directory(folder, batch_size = batch_size, target_size = (160, 160), shuffle = True)

    model = get_model()

    model.fit_generator(train_gen, steps_per_epoch= train_gen.samples // batch_size,epochs=10)

fit("./data/", 32)