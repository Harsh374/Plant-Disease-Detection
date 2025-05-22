import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import cv2
import streamlit as st
from PIL import Image
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMAGE_SIZE = 224  # Standard size for many CNN architectures
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# We'll use 5 classes as specified in the requirements
# You can modify this list based on which classes you want to focus on
SELECTED_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot', 
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Tomato___healthy'
]

def create_model(num_classes):
    """Create a CNN model for plant disease classification"""
    # Use a pre-trained model as the base
    base_model = applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data(data_dir, selected_classes):
    """Prepare training, validation, and test datasets"""
    # Define the data directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Split the original data into train, validation, and test sets
    split_dataset(data_dir, train_dir, val_dir, test_dir, selected_classes)
    
    # Create data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def split_dataset(source_dir, train_dir, val_dir, test_dir, selected_classes, split_ratio=(0.7, 0.15, 0.15)):
    """Split the dataset into train, validation, and test sets"""
    import shutil
    
    # Create class directories in train, val, and test
    for class_name in selected_classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    # The source directory contains color images
    source_color_dir = os.path.join(source_dir, 'color')
    
    # Process each selected class
    for class_name in selected_classes:
        class_dir = os.path.join(source_color_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        np.random.shuffle(images)
        
        # Calculate split sizes
        train_size = int(len(images) * split_ratio[0])
        val_size = int(len(images) * split_ratio[1])
        
        # Split images
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, class_name, img))
        
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_dir, class_name, img))
        
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, class_name, img))

def train_model(model, train_generator, validation_generator, model_save_path='best_model.h5'):
    """Train the model with callbacks for early stopping and model checkpointing"""
    # Create callbacks
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping]
    )
    
    return history, model

def evaluate_model(model, test_generator):
    """Evaluate the model on the test dataset"""
    # Get the model's predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Calculate accuracy
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate classification report
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    
    # Plot training & validation accuracy and loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return test_accuracy, cm, class_names

def preprocess_image(image_path):
    """Preprocess an image for prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_disease(model, image_path, class_names):
    """Predict the disease for a given image"""
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[predicted_class_index]
    
    return predicted_class, confidence

# Main training script
if __name__ == "__main__":
    # Path to the dataset
    data_dir = "plantvillage dataset"
    
    # Prepare datasets
    train_generator, validation_generator, test_generator = prepare_data(data_dir, SELECTED_CLASSES)
    
    # Get number of classes
    num_classes = len(SELECTED_CLASSES)
    
    # Create and train model
    model = create_model(num_classes)
    history, model = train_model(model, train_generator, validation_generator)
    
    # Evaluate model
    test_accuracy, confusion_matrix, class_names = evaluate_model(model, test_generator)
    
    # Save model for later use
    model.save('plant_disease_model.h5')
    
    print("Model trained and saved successfully!")
