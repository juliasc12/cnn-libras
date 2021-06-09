from keras.utils import to_categorical, plot_model
from keras.optimizers import SGD, Adam
from keras import backend
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from cnn import Convolucao

import datetime
import h5py
import time

def getDateStr():
        return str('{date:%d_%m_%Y_%H_%M}').format(date=datetime.datetime.now())

def getTimeMin(start, end):
        return (end - start)/60

EPOCHS = 30
CLASS = 21
FILE_NAME = 'cnn_model_LIBRAS_'

print("\n\n ----------------------INICIO --------------------------\n")
print('[INFO] [INICIO]: ' + getDateStr())
print('[INFO] Download dataset usando keras.preprocessing.image.ImageDataGenerator')

train_datagen = ImageDataGenerator(
        rescale=1./255, #rescale: rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (after applying all other transformations).
        shear_range=0.2, #shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        zoom_range=0.2, #zoom_range: Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        horizontal_flip=True, #Boolean. Randomly flip inputs horizontally.
        validation_split=0.25) #validation_split: Float. Fraction of images reserved for validation (strictly between 0 and 1).



test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.05)

# produz base de dados de treino gerada pelo modelo #
training_set = train_datagen.flow_from_directory(
        '../dataset/training',
        target_size=(64, 64),
        color_mode = 'rgb',
        batch_size=32,
        shuffle=True,
        #save_to_dir='../img_generator/train',
        class_mode='categorical')

#Takes the path to a directory & generates batches of augmented data.
test_set = test_datagen.flow_from_directory(
        '../dataset/test', #directory: string, path to the target directory. It should contain one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator. See this script for more details.
        target_size=(64, 64),
        color_mode = 'rgb',
        batch_size=32,
        shuffle=True, #shuffle: Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.
        #save_to_dir='../img_generator/test',
        class_mode='categorical')

# inicializar e otimizar modelo
print("[INFO] Inicializando e otimizando a CNN...")
start = time.time()


early_stopping_monitor = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

model = Convolucao.build(64, 64, 3, CLASS)

model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",
              metrics=["accuracy"])

# treinar a CNN
print("[INFO] Treinando a CNN...")
classifier = model.fit_generator(
        training_set,
        #steps_per_epoch: Integer. Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch. It should typically be equal to ceil(num_samples / batch_size). Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
        #You can set it equal to num_samples // batch_size, which is a typical choice.
        steps_per_epoch=(training_set.n // training_set.batch_size),
        #Iteration is one time processing for forward and backward for a batch of images (say one batch is defined as 16, then 16 images are processed in one iteration). Epoch is once all images are processed one time individually of forward and backward to the network, then that is one epoch.
        epochs=EPOCHS,
        validation_data = test_set,
        validation_steps= (test_set.n // test_set.batch_size),
        #shuffle = False,
        verbose=2,
        callbacks = [early_stopping_monitor]
      )

# atualizo valor da epoca caso o treinamento tenha finalizado antes do valor de epoca que foi iniciado
EPOCHS = len(classifier.history["loss"])

print("[INFO] Salvando modelo treinado ...")

#para todos arquivos ficarem com a mesma data e hora. Armazeno na variavel
file_date = getDateStr()
model.save('../models/'+FILE_NAME+file_date+'.h5')
print('[INFO] modelo: ../models/'+FILE_NAME+file_date+'.h5 salvo!')

end = time.time()

print("[INFO] Tempo de execução da CNN: %.1f min" %(getTimeMin(start,end)))

print('[INFO] Summary: ')
model.summary()

print("\n[INFO] Avaliando a CNN...")
score = model.evaluate_generator(generator=test_set, steps=(test_set.n // test_set.batch_size), verbose=1)
print('[INFO] Accuracy: %.2f%%' % (score[1]*100), '| Loss: %.5f' % (score[0]))

print("[INFO] Sumarizando loss e accuracy para os datasets 'train' e 'test'")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,EPOCHS), classifier.history["loss"], label="train_loss")
plt.plot(np.arange(0,EPOCHS), classifier.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,EPOCHS), classifier.history["acc"], label="train_acc")
plt.plot(np.arange(0,EPOCHS), classifier.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('../models/graphics/'+FILE_NAME+file_date+'.png', bbox_inches='tight')

print('[INFO] Gerando imagem do modelo de camadas da CNN')
plot_model(model, to_file='../models/image/'+FILE_NAME+file_date+'.png', show_shapes = True)

print('\n[INFO] [FIM]: ' + getDateStr())
print('\n\n')
