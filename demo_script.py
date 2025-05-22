import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def load_and_preprocess_image(image_path, image_size=224):
    """Load and preprocess an image for prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def visualize_prediction(image_path, predicted_class, confidence, class_names, image_size=224):
    """Visualize the prediction result with the original image"""
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Prediction visualization
    plt.subplot(1, 2, 2)
    
    # Create a bar chart for confidence scores
    confidences = np.zeros(len(class_names))
    confidences[class_names.index(predicted_class)] = confidence
    
    # Sort by confidence
    sorted_indices = np.argsort(confidences)
    class_names_sorted = [class_names[i].replace('___', ': ').replace('_', ' ') for i in sorted_indices]
    confidences_sorted = confidences[sorted_indices]
    
    # Display only top 3 predictions
    top_n = 3
    top_indices = sorted_indices[-top_n:][::-1]
    top_classes = [class_names[i].replace('___', ': ').replace('_', ' ') for i in top_indices]
    top_confidences = confidences[top_indices]
    
    # Bar chart
    plt.barh(range(len(top_classes)), top_confidences * 100)
    plt.yticks(range(len(top_classes)), top_classes)
    plt.xlabel('Confidence (%)')
    plt.title('Prediction Confidence')
    plt.xlim([0, 100])
    
    for i, v in enumerate(top_confidences):
        plt.text(v * 100 + 1, i, f'{v*100:.1f}%')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.close()

def evaluate_model_on_test_data(model, test_data_dir, class_names, image_size=224, batch_size=32):
    """Evaluate model performance on test data"""
    # Create test data generator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[c.replace('___', '\n') for c in class_names],
                yticklabels=[c.replace('___', '\n') for c in class_names])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_evaluation.png')
    plt.close()
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    return test_acc, report_df

def run_demo():
    """Run a demo of the plant disease detection system"""
    # Load trained model
    model_path = 'plant_disease_model.h5'
    model = load_model(model_path)
    
    # Define class names (must match the order used during training)
    class_names = [
        'Apple___Apple_scab',
        'Apple___Black_rot', 
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Tomato___healthy'
    ]
    
    # Test directory path
    test_dir = 'plantvillage dataset/test'
    
    # Select a few test images
    test_images = []
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_images.append(os.path.join(class_dir, images[0]))
    
    # Run prediction on test images
    print("Demo predictions on test images:")
    for image_path in test_images:
        # Extract class name from path
        true_class = image_path.split(os.path.sep)[-2]
        
        # Preprocess image
        img_batch = load_and_preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict(img_batch)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = prediction[predicted_class_index]
        
        # Display result
        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"True class: {true_class.replace('___', ': ')}")
        print(f"Predicted class: {predicted_class.replace('___', ': ')}")
        print(f"Confidence: {confidence * 100:.2f}%")
        
        # Visualize result
        visualize_prediction(image_path, predicted_class, confidence, class_names)
    
    # Evaluate model on test data
    print("\nEvaluating model on test data...")
    test_acc, report_df = evaluate_model_on_test_data(model, test_dir, class_names)
    
    # Save evaluation results
    report_df.to_csv('evaluation_report.csv')
    print("\nClassification Report:")
    print(report_df)
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    # Execute the demo
    run_demo()
