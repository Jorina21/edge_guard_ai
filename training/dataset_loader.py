# training/dataset_loader.py

import tensorflow as tf
import pandas as pd
import numpy as np
import os

class EdgeGuardDataset:
    def __init__(self, data_dir="../data/raw/", labels_file="../data/labels.csv",
                 img_size=(224, 224), batch_size=32, val_split=0.2):

        self.data_dir = data_dir
        self.labels_file = labels_file
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split

        # Load the CSV containing filenames + labels
        self.df = pd.read_csv(self.labels_file)
        self.class_names = sorted(self.df['label'].unique())

        print(f"[INFO] Loaded {len(self.df)} labeled images")
        print(f"[INFO] Classes: {self.class_names}")

    def preprocess_image(self, image_path, label):
        """
        Loads image from path, decodes it, resizes, normalizes.
        """
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        img = img / 255.0  # normalize [0,1]

        return img, label

    def augment_image(self, img, label):
        """
        Data augmentation to improve robustness.
        """
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)

        return img, label

    def prepare_dataset(self):
        """
        Converts dataframe into TensorFlow datasets.
        Splits into train/validation sets.
        Returns: train_ds, val_ds, class_names
        """

        # Build full paths for each image
        self.df['path'] = self.df['filename'].apply(lambda x: os.path.join(self.data_dir, x))

        # Encode labels to integers
        label_to_index = {name: i for i, name in enumerate(self.class_names)}
        self.df['label_id'] = self.df['label'].map(label_to_index)

        # Shuffle dataset
        df_sampled = self.df.sample(frac=1, random_state=42)

        # Train/validation split
        val_size = int(len(df_sampled) * self.val_split)
        df_val = df_sampled.iloc[:val_size]
        df_train = df_sampled.iloc[val_size:]

        print(f"[INFO] Training samples: {len(df_train)}")
        print(f"[INFO] Validation samples: {len(df_val)}")

        # Build tf.data pipelines
        train_ds = tf.data.Dataset.from_tensor_slices((df_train['path'], df_train['label_id']))
        val_ds = tf.data.Dataset.from_tensor_slices((df_val['path'], df_val['label_id']))

        train_ds = train_ds.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(500).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = val_ds.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, self.class_names
