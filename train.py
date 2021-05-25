import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.keras.mixed_precision import experimental



def train(path):

    imgsize = (224, 224)
    imgshape = imgsize + (3,)

    label_names = sorted(['CR', 'MUR', 'other', 'S', 'SRC', 'T'])
    label_names = {i: l for i, l in enumerate(label_names)}

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=imgsize,
        batch_size=16,
        label_mode='categorical',
        seed=134,
        #shuffle=False,
        validation_split=0.4,
        subset='training',
    )
    #train_ds = train_ds.map(
    #    lambda x, y: (tf.keras.applications.resnet_v2.preprocess_input(x), y)
    #)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=imgsize,
        batch_size=16,
        label_mode='categorical',
        seed=134,
        #shuffle=False,
        validation_split=0.4,
        subset='validation',
    )
    #val_ds = val_ds.map(
    #    lambda x, y: (tf.keras.applications.resnet_v2.preprocess_input(x), y)
    #)

    #train_ds.prefetch(16)
    #val_ds.prefetch(16)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.025),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
        ]
    )

   
    # Plot
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            #plt.axis("off")
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            lbl = label_names[(np.argmax(labels[0], axis=0))]
            plt.xlabel(lbl)
            print(labels[0])
        plt.show()



    inputs = tf.keras.layers.Input(shape=imgshape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)



    resnet_model = tf.keras.applications.ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=imgsize + (3,)
    )
    for layer in resnet_model.layers:
        layer.trainable = False
    
    x = resnet_model(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    #x = tf.keras.layers.Dense(128, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(6, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print(model.summary())

    optimizer = tf.keras.optimizers.Adam(lr=0.005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # 1: Train with ResNet model fixed
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=2
    )

    # 2: Unfreeze highest block, train more
    unfreeze_blocks = ['conv5_block3', 'conv5_block2']
    for layer in resnet_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            for block in unfreeze_blocks:
                if layer.name.startswith(block):
                    print('unfreezing', layer.name)
                    layer.trainable = True

    print(model.summary())

    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        verbose=2
    )

    # Plot along with predictions
    plt.figure(figsize=(10, 10))
    for images, labels in val_ds.take(1):
        preds = model.predict(images)
        for i in range(images.shape[0]):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            true_label = np.argmax(labels[i], axis=0) 
            true_label = label_names[true_label]
            best_pred = np.argmax(preds[i], axis=0)
            best_pred = label_names[best_pred]
            plt.xlabel(f'true: {true_label}, pred: {best_pred}')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.show()



if __name__ == '__main__':

    train('copied_images/streetview_apr_26')

    