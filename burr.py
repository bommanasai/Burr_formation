import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
import os
train_dir = r"C:\\Users\\HP\\Downloads\\new_images\\train" #Change path 
test_dir = r"C:\\Users\\HP\\Downloads\\new_images\\test"   #Change path

train_datagen = ImageDataGenerator(rescale=1./255,           
                                   shear_range=0.2,          
                                   zoom_range=0.2,           
                                   horizontal_flip=True,
                                   rotation_range=30,       
                                   width_shift_range=0.2,  
                                   height_shift_range=0.2)      

test_datagen = ImageDataGenerator(rescale=1./255)    

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),     # Resize images to a fixed size
    batch_size=32,            # Process images in batches
    class_mode='categorical'  # Use categorical labels (for multi-class classification)
)

# Load test data from the directory
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

import os
print(os.listdir(train_dir))  # Check the image list in dataset
print(os.listdir(test_dir))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3),padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(train_generator.num_classes, activation='softmax'))

from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=50, 
                    validation_data=test_generator)
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")
