import os

from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Images:
    def __init__(self, input_shape, directory, fine_tuning=False):
        self.input_shape = input_shape
        self.directory = directory
        self.fine_tuning = fine_tuning

        self.test_dir = os.path.join(self.directory, "test")
        self.train_dir = os.path.join(self.directory, "train")
        self.validation_dir = os.path.join(self.directory, "validation")

        if len([name for name in os.listdir(self.train_dir) if os.path.isfile(os.path.join(self.train_dir, name))]) < 5000:
            self.data_augmentation = True
        else:
            self.data_augmentation = False

        self.extract_features()

        self.model = self.build()

    def extract_features(self):
        if self.data_augmentation:
            self.train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode="nearest"
            )
        else:
            self.train_datagen = ImageDataGenerator(rescale=1./255)

        # Validation data should not be augmented!
        self.test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=20,
            class_mode="binary"
        )

        self.validation_generator = self.test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=20,
            class_mode="binary"
        )

    def build(self):
        conv_base = VGG16(weights="imagenet", include_top=False, input_shape=self.input_shape)

        if self.fine_tuning:
            conv_base.trainable = True
            set_trainable = False
            for layer in conv_base.layers:
                if layer.name == "block5_conv1":
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False
        else:
            conv_base.trainable = False

        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))

        return model

    def train(self):
        self.model.compile(loss="binary_crossentropy",
                           optimizer=optimizers.RMSprop(lr=2e-5),
                           metrics=["acc"])

        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=100,
            epochs=30,
            validation_data=self.validation_generator,
            validation_steps=50,
            verbose=0)

    def evaluate(self):
        test_generator = self.test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=20,
            class_mode="binary")

        test_loss, test_acc = self.model.evaluate_generator(test_generator, steps=20)

        return test_acc
