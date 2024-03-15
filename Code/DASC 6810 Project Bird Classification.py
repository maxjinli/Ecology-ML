# There are two datasets (class 10 and class 100) to be analyzed with two approaches (ML method and DL method)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import machine learning libraries
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

# Import deep learning libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

##

# Machine learning approach with class 10

# Load the datasets
train_df = pd.read_csv('/content/drive/My Drive/DASC6810 Project/Data/Birds-10-Species/train_features.csv')
test_df = pd.read_csv('/content/drive/My Drive/DASC6810 Project/Data/Birds-10-Species/test_features.csv')
valid_df = pd.read_csv('/content/drive/My Drive/DASC6810 Project/Data/Birds-10-Species/valid_features.csv')

# Split into features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']
X_valid = valid_df.drop('label', axis=1)
y_valid = valid_df['label']

# Initialize classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Machine": SVC()
}

# Create dictionaries to store the metrics
precisions = {}
recalls = {}
f1_scores = {}
error_rates = {}
accuracies = {}

# Train, predict and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)

    # Compute accuracy
    accuracy = accuracy_score(y_valid, y_pred)

    # Compute precision (macro-averaged)
    precision = precision_score(y_valid, y_pred, average='macro')

    # Compute recall (macro-averaged)
    recall = recall_score(y_valid, y_pred, average='macro')

    # Compute F1-score (macro-averaged)
    f1 = f1_score(y_valid, y_pred, average='macro')

    # Compute error rate
    error_rate = 1 - accuracy

    # Store the metrics in dictionaries
    accuracies[name] = accuracy
    precisions[name] = precision
    recalls[name] = recall
    f1_scores[name] = f1
    error_rates[name] = error_rate

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Precision (Macro): {precision:.4f}")
    print(f"{name} Recall (Macro): {recall:.4f}")
    print(f"{name} F1-Score (Macro): {f1:.4f}")
    print(f"{name} Error Rate: {error_rate:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_valid, y_pred)

    # Display a larger heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Optionally, you can also aggregate classes for better visualization
    # and display the confusion matrix for the top N classes
    N = 10  # Set the number of top classes to display
    top_classes = np.argsort(np.sum(cm, axis=0))[-N:]  # Find the N classes with the highest sum of true positives
    top_cm = cm[top_classes][:, top_classes]

    # Display the confusion matrix for the top N classes
    plt.figure(figsize=(10, 7))
    sns.heatmap(top_cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {name} (Top {N} Classes)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


##

# Machine learning approach with class 100

# Load the datasets
train_df = pd.read_csv('/content/drive/My Drive/DASC6810 Project/Data/Birds-100-Species/train_features_100.csv')
test_df = pd.read_csv('/content/drive/My Drive/DASC6810 Project/Data/Birds-100-Species/test_features_100.csv')
valid_df = pd.read_csv('/content/drive/My Drive/DASC6810 Project/Data/Birds-100-Species/valid_features_100.csv')

# Split into features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']
X_valid = valid_df.drop('label', axis=1)
y_valid = valid_df['label']

# Initialize classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Machine": SVC()
}

# Create dictionaries to store the metrics
precisions = {}
recalls = {}
f1_scores = {}
error_rates = {}
accuracies = {}

# Train, predict and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)

    # Compute accuracy
    accuracy = accuracy_score(y_valid, y_pred)

    # Compute precision (macro-averaged)
    precision = precision_score(y_valid, y_pred, average='macro')

    # Compute recall (macro-averaged)
    recall = recall_score(y_valid, y_pred, average='macro')

    # Compute F1-score (macro-averaged)
    f1 = f1_score(y_valid, y_pred, average='macro')

    # Compute error rate
    error_rate = 1 - accuracy

    # Store the metrics in dictionaries
    accuracies[name] = accuracy
    precisions[name] = precision
    recalls[name] = recall
    f1_scores[name] = f1
    error_rates[name] = error_rate

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Precision (Macro): {precision:.4f}")
    print(f"{name} Recall (Macro): {recall:.4f}")
    print(f"{name} F1-Score (Macro): {f1:.4f}")
    print(f"{name} Error Rate: {error_rate:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_valid, y_pred)

    # Display a larger heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Optionally, you can also aggregate classes for better visualization
    # and display the confusion matrix for the top N classes
    N = 10  # Set the number of top classes to display
    top_classes = np.argsort(np.sum(cm, axis=0))[-N:]  # Find the N classes with the highest sum of true positives
    top_cm = cm[top_classes][:, top_classes]

    # Display the confusion matrix for the top N classes
    plt.figure(figsize=(10, 7))
    sns.heatmap(top_cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {name} (Top {N} Classes)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

##

# Deep learning approach with class 10

# Load and inspect data
def create_df_from_directory(dir_path):
    filepaths = []
    labels = []
    for folder_name in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path) and not folder_name.startswith('.'):
            for filename in os.listdir(folder_path):
                if not filename.startswith('.'):
                    file_path = os.path.join(folder_path, filename)
                    filepaths.append(file_path)
                    labels.append(folder_name)
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})

train_dir = '/content/drive/My Drive/DASC6810 Project/Data/Birds/train'
valid_dir = '/content/drive/My Drive/DASC6810 Project/Data/Birds/valid'
test_dir = '/content/drive/My Drive/DASC6810 Project/Data/Birds/test'

# Create dataframes
train_df = create_df_from_directory(train_dir)
valid_df = create_df_from_directory(valid_dir)
test_df = create_df_from_directory(test_dir)

# Display first 5 rows of each dataframe
print(train_df.head())
print(valid_df.head())
print(test_df.head())

# Define parameters
batch_size = 64
img_size = (150, 150)
channels = 3
img_shape = img_size + (channels,)  # Create a tuple (150, 150, 3)

# Initialize ImageDataGenerator objects
data_gen_args = dict(
    target_size=img_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    batch_size=batch_size
)

# Create training, validation and test generators
train_gen = ImageDataGenerator().flow_from_dataframe(
    dataframe=train_df,
    x_col='filepaths',
    y_col='labels',
    **data_gen_args
)

data_gen_args['shuffle'] = True
valid_gen = ImageDataGenerator().flow_from_dataframe(
    dataframe=valid_df,
    x_col='filepaths',
    y_col='labels',
    **data_gen_args
)

data_gen_args['shuffle'] = False
test_gen = ImageDataGenerator().flow_from_dataframe(
    dataframe=test_df,
    x_col='filepaths',
    y_col='labels',
    **data_gen_args
)

# Retrieve class indices and names
class_indices = train_gen.class_indices
class_names = list(class_indices.keys())

# Function to plot images in a grid
def plot_images(images, labels, classes, rows=4, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.ravel()

    for i in np.arange(0, rows * cols):
        axes[i].imshow(images[i] / 255)  # Scale image pixel values to [0, 1]
        index = np.argmax(labels[i])
        class_name = classes[index]
        axes[i].set_title(class_name, color='black', fontsize=10)
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0.5)

# Plot a batch of images from the training generator
images, labels = next(train_gen)
plot_images(images, labels, class_names)

plt.show()

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Ensure that this matches the flattened size
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten the output for the dense layer
        x = x.view(x.size(0), -1)  # Adjusted for variable batch sizes
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Check if GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model and move it to the GPU if available
model = SimpleCNN(num_classes=10).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define transformations for your image data
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root='/content/drive/My Drive/DASC6810 Project/Data/Birds/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

valid_dataset = datasets.ImageFolder(root='/content/drive/My Drive/DASC6810 Project/Data/Birds/valid', transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, drop_last=True)

test_dataset = datasets.ImageFolder(root='/content/drive/My Drive/DASC6810 Project/Data/Birds/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, drop_last=True)

# Placeholder for the metrics
train_loss_history = []
valid_accuracy_history = []

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation Loop
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss_history.append(running_loss/len(train_loader))
    valid_accuracy_history.append(100 * correct / total)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, '
          f'Validation Accuracy: {100 * correct / total}%')

# Testing Loop
model.eval()
correct = 0
total = 0
for images, labels in test_loader:
    print('labels', labels)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()


print(f'Test Accuracy: {100 * correct / total}%')

# Visualization of the training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Visualization of the validation accuracy
plt.subplot(1, 2, 2)
plt.plot(valid_accuracy_history, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

##

# Deep learning with class 100

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Ensure that this matches the flattened size
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten the output for the dense layer
        x = x.view(x.size(0), -1)  # Adjusted for variable batch sizes
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Check if GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model and move it to the GPU if available
model = SimpleCNN(num_classes=100).to(device)
# model = SimpleCNN(num_classes=10).cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define transformations for your image data
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root='/content/drive/My Drive/DASC6810 Project/Data/Birds-100-Species/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

valid_dataset = datasets.ImageFolder(root='/content/drive/My Drive/DASC6810 Project/Data/Birds-100-Species/valid', transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=128, drop_last=True)

test_dataset = datasets.ImageFolder(root='/content/drive/My Drive/DASC6810 Project/Data/Birds-100-Species/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=50, drop_last=True)

# Placeholder for the metrics
train_loss_history = []
valid_accuracy_history = []

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation Loop
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss_history.append(running_loss/len(train_loader))
    valid_accuracy_history.append(100 * correct / total)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, '
          f'Validation Accuracy: {100 * correct / total}%')

# Testing Loop
model.eval()
correct = 0
total = 0
for images, labels in test_loader:
    print('labels', labels)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()


print(f'Test Accuracy: {100 * correct / total}%')

# Visualization of the training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Visualization of the validation accuracy
plt.subplot(1, 2, 2)
plt.plot(valid_accuracy_history, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

# Arrays to hold the true labels and predictions
all_labels = []
all_predictions = []

# Testing loop to collect the labels and predictions
model.eval()
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    # Append current results to the lists
    all_labels.extend(labels.cpu().numpy())
    all_predictions.extend(predicted.cpu().numpy())

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Define the plot
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Labels, title and ticks
label_names = train_dataset.classes
plt.ylabel('Actual labels')
plt.xlabel('Predicted labels')
plt.title('Confusion Matrix')
plt.xticks(ticks=np.arange(len(label_names)) + 0.5, labels=label_names, rotation=90)
plt.yticks(ticks=np.arange(len(label_names)) + 0.5, labels=label_names, rotation=0)

# Show the plot
plt.show()
