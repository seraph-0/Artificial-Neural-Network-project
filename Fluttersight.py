from google.colab import drive
drive.mount('/content/drive')
!unzip /content/drive/MyDrive/archive.zip -d /content/
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import kagglehub

# Download latest version
path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")

print("Path to dataset files:", path)
df = pd.read_csv("/content/Training_set.csv")
df.head(10)
len(df)
class_counts = df['label'].value_counts().sort_index()

plt.figure(figsize=(14, 8))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Distribution of Butterfly Classes')
plt.xlabel('Butterfly Classes')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
colors = sns.color_palette("viridis", len(class_counts))
class_counts = df['label'].value_counts().sort_index()


plt.figure(figsize=(10, 10))
plt.pie(class_counts.values, labels=None, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Percentage of Each Butterfly Class')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.legend(class_counts.index, loc="best", bbox_to_anchor=(1, 0.5))
plt.show()
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import regularizers
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=r"Your PyDataset class should call super().__init__\(\*\*kwargs\)")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.impute import SimpleImputer

# Define image directory
image_dir = "/content/train"

# Sample 4 random images
sample_images = df.sample(4, random_state=42)

# Display the images and extracted matrix values after imputation
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Initialize matrices for predictors (X1, X2, X3) and target (X4)
predictors = []
targets = []

# Preprocessing and matrix extraction
for index, row in sample_images.iterrows():
    img_path = os.path.join(image_dir, row['filename'])
    try:
        img = load_img(img_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0  # Normalize the image

        # Divide the image into quadrants
        h, w, c = img_array.shape
        half_h, half_w = h // 2, w // 2

        X1 = np.mean(img_array[:half_h, :half_w, :])  # Top-left quadrant
        X2 = np.mean(img_array[:half_h, half_w:, :])  # Top-right quadrant
        X3 = np.mean(img_array[half_h:, :half_w, :])  # Bottom-left quadrant
        X4 = np.mean(img_array[half_h:, half_w:, :])  # Bottom-right quadrant (target)

        predictors.append([X1, X2, X3])
        targets.append(X4)
    except Exception as e:
        # Handle any image loading errors
        predictors.append([np.nan, np.nan, np.nan])
        targets.append(np.nan)

# Convert to numpy arrays
predictors = np.array(predictors)
targets = np.array(targets)

# Impute missing values
predictor_imputer = SimpleImputer(strategy='mean')  # Use mean imputation for predictors
target_imputer = SimpleImputer(strategy='mean')     # Use mean imputation for targets

predictors = predictor_imputer.fit_transform(predictors)
targets = target_imputer.fit_transform(targets.reshape(-1, 1)).flatten()



for i, (index, row) in enumerate(sample_images.iterrows()):
    img_path = os.path.join(image_dir, row['filename'])
    ax = axes[i // 2, i % 2]
    try:
        img = load_img(img_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0  # Normalize the image

        ax.imshow(img_array)
        ax.set_title(
            f"X1: {predictors[i][0]:.2f}, X2: {predictors[i][1]:.2f}, "
            f"X3: {predictors[i][2]:.2f}, X4 (target): {targets[i]:.2f}"
        )
    except Exception as e:
        # If image loading failed, show a placeholder
        ax.text(0.5, 0.5, 'Image Load Failed', ha='center', va='center', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

image_dir = "/content/train"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
model_CNN = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(75, activation='softmax')
])

model_CNN.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model_CNN.summary()
history = model_CNN.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)