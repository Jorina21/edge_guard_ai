# training/model_architecture.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Builds a MobileNetV2-based classifier for EdgeGuard AI with fine-tuning.

    Args:
        input_shape: Input image size.
        num_classes: Number of output classes.

    Returns:
        A compiled TensorFlow Keras model.
    """

    # Load MobileNetV2 base (pretrained)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # -------------------------------------------------------
    # FINE-TUNING SECTION
    # -------------------------------------------------------
    # First, freeze ALL layers
    base_model.trainable = False

    # Then unfreeze the last 60 layers for fine-tuning
    for layer in base_model.layers[-60:]:
        layer.trainable = True

    # -------------------------------------------------------

    inputs = layers.Input(shape=input_shape)

    # Preprocessing for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Feature extraction
    x = base_model(x, training=True)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # Compile with lower learning rate since we're fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Print model summary for confirmation
    print(model.summary())

    return model
