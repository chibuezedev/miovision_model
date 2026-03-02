import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import tensorflow as tf
from tensorflow import keras  # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Dense,
    Conv1D,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.preprocessing.image import ( # type: ignore
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras import regularizers # type: ignore

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

import glob

warnings.filterwarnings("ignore")


class Config:
    BASE_DIR = os.getcwd()
    TABULAR_DATA_PATH = os.path.join(BASE_DIR, "DATASET2", "myopia.CSV")
    IMAGE_DATA_DIR = os.path.join(BASE_DIR, "DATASET2", "IMAGEDATA")
    OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUTS")
    MODEL_DIR = os.path.join(BASE_DIR, "SAVED_MODELS")

    EPOCHS_CNN_BASIC = 10
    EPOCHS_HYBRID = 40
    EPOCHS_IMAGE = 50
    BATCH_SIZE = 16
    IMG_SIZE = (224, 224)
    LEARNING_RATE = 0.00009
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    PREDICTION_THRESHOLD = 0.5


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)

print("=" * 80)
print("HYBRID PREDICTIVE MODEL FOR EARLY DETECTION OF MYOPIA")
print("=" * 80)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Output Directory: {Config.OUTPUT_DIR}")
print(f"Model Directory: {Config.MODEL_DIR}")
print("=" * 80)


def save_figure(filename, dpi=300):
    """Save figure with timestamp"""
    filepath = os.path.join(Config.OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"✓ Saved: {filename}")


def plot_training_history(history, model_name, save=True):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], "-o", label="Train", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], "-o", label="Validation", linewidth=2)
    axes[0].set_title(f"{model_name} - Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], "-o", label="Train", linewidth=2)
    axes[1].plot(history.history["val_loss"], "-o", label="Validation", linewidth=2)
    axes[1].set_title(f"{model_name} - Loss", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        save_figure(f"{model_name.lower().replace(' ', '_')}_training_history.png")
    plt.show()


def plot_confusion_matrix(
    y_true, y_pred, model_name, labels=["NORMAL", "MYOPIA"], save=True
):
    """Plot confusion matrix with additional metrics"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{model_name} - Confusion Matrix", fontsize=14, fontweight="bold")

    if save:
        save_figure(f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.show()

    return cm


def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive performance metrics"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "specificity": None,
        "sensitivity": None,
        "auc_roc": None,
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0

    if y_pred_proba is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            print(f"Warning: unable to compute AUC-ROC: {e}")
            pass

    return metrics


def calculate_optimal_threshold(y_true, y_pred_proba):
    """Calculate optimal threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def print_metrics_report(metrics, model_name):
    """Print formatted metrics report"""
    print("\n" + "=" * 60)
    print(f"{model_name} - PERFORMANCE METRICS")
    print("=" * 60)
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric.replace('_', ' ').title():.<40} {value * 100:.2f}%")
    print("=" * 60)


def save_metrics_to_csv(metrics_dict, filename="model_metrics.csv"):
    """Save all metrics to CSV"""
    df = pd.DataFrame(metrics_dict).T
    filepath = os.path.join(Config.OUTPUT_DIR, filename)
    df.to_csv(filepath)
    print(f"✓ Metrics saved to: {filename}")


print("\n" + "=" * 80)
print("PART 1: TABULAR DATA ANALYSIS")
print("=" * 80)

df = pd.read_csv(Config.TABULAR_DATA_PATH)
df = pd.concat([df, df, df, df], ignore_index=True)  # Data augmentation

df["MYOPIC"] = df["MYOPIC"].astype(int)

print(f"Dataset shape: {df.shape}")
print("\nDataset info:")
print(df.info())
print("\nTarget distribution:")
print(df["MYOPIC"].value_counts())
print(f"  0 (NORMAL): {(df['MYOPIC'] == 0).sum()}")
print(f"  1 (MYOPIA): {(df['MYOPIC'] == 1).sum()}")


def plot_target_distribution(df):
    """Plot target variable distribution"""
    myopia_counts = df["MYOPIC"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        ["NORMAL (0)", "MYOPIA (1)"],
        [myopia_counts[0], myopia_counts[1]],
        color=["#3498db", "#e74c3c"],
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_title(
        "Myopia Detection - Target Distribution", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    save_figure("target_distribution.png")
    plt.show()


plot_target_distribution(df)

y = df["MYOPIC"].values
X_data = df.drop(["MYOPIC"], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler.fit_transform(X_data)

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized,
    y,
    test_size=Config.TEST_SIZE,
    random_state=Config.RANDOM_STATE,
    stratify=y,
)

# Standardize
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

# Reshape for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(
    f"Training labels - NORMAL: {(y_train == 0).sum()}, MYOPIA: {(y_train == 1).sum()}"
)
print(f"Testing labels - NORMAL: {(y_test == 0).sum()}, MYOPIA: {(y_test == 1).sum()}")


print("\n" + "=" * 80)
print("TRAINING: BASIC CNN MODEL")
print("=" * 80)

cnn_model = Sequential(
    [
        Conv1D(32, 2, activation="relu", input_shape=X_train[0].shape),
        BatchNormalization(),
        Dropout(0.5),
        Conv1D(64, 2, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ],
    name="Basic_CNN",
)

cnn_model.compile(
    optimizer=Adam(learning_rate=Config.LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print(cnn_model.summary())

cnn_callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        os.path.join(Config.MODEL_DIR, "cnn_basic_best.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

history_cnn = cnn_model.fit(
    X_train,
    y_train,
    epochs=Config.EPOCHS_CNN_BASIC,
    validation_data=(X_test, y_test),
    callbacks=cnn_callbacks,
    verbose=1,
)

cnn_model.save(os.path.join(Config.MODEL_DIR, "cnn_basic_final.h5"))
print("✓ Model saved: cnn_basic_final.h5")

plot_training_history(history_cnn, "Basic CNN")

y_pred_proba_cnn = cnn_model.predict(X_test).flatten()

optimal_threshold_cnn = calculate_optimal_threshold(y_test, y_pred_proba_cnn)
print(f"\n✓ Optimal threshold (Basic CNN): {optimal_threshold_cnn:.4f}")

y_pred_cnn = (y_pred_proba_cnn >= optimal_threshold_cnn).astype(int)

cnn_metrics = calculate_comprehensive_metrics(y_test, y_pred_cnn, y_pred_proba_cnn)
print_metrics_report(cnn_metrics, "Basic CNN")

plot_confusion_matrix(y_test, y_pred_cnn, "Basic CNN")

print("\nClassification Report - Basic CNN:")
print(classification_report(y_test, y_pred_cnn, target_names=["NORMAL", "MYOPIA"]))


print("\n" + "=" * 80)
print("TRAINING: HYBRID CNN MODEL WITH REGULARIZATION")
print("=" * 80)

hybrid_model = Sequential(
    [
        Conv1D(32, 2, activation="relu", input_shape=X_train[0].shape),
        BatchNormalization(),
        Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5),
        ),
        Dropout(0.5),
        Conv1D(64, 2, activation="relu"),
        BatchNormalization(),
        Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5),
        ),
        Dropout(0.5),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5),
        ),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ],
    name="Hybrid_CNN",
)

hybrid_model.compile(
    optimizer=Adam(learning_rate=Config.LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print(hybrid_model.summary())

hybrid_callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    ModelCheckpoint(
        os.path.join(Config.MODEL_DIR, "hybrid_cnn_best.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

history_hybrid = hybrid_model.fit(
    X_train,
    y_train,
    epochs=Config.EPOCHS_HYBRID,
    validation_data=(X_test, y_test),
    callbacks=hybrid_callbacks,
    verbose=1,
)

hybrid_model.save(os.path.join(Config.MODEL_DIR, "hybrid_cnn_final.h5"))
print("✓ Model saved: hybrid_cnn_final.h5")

plot_training_history(history_hybrid, "Hybrid CNN")

y_pred_proba_hybrid = hybrid_model.predict(X_test).flatten()

optimal_threshold_hybrid = calculate_optimal_threshold(y_test, y_pred_proba_hybrid)
print(f"\n✓ Optimal threshold (Hybrid CNN): {optimal_threshold_hybrid:.4f}")

y_pred_hybrid = (y_pred_proba_hybrid >= optimal_threshold_hybrid).astype(int)

hybrid_metrics = calculate_comprehensive_metrics(
    y_test, y_pred_hybrid, y_pred_proba_hybrid
)
print_metrics_report(hybrid_metrics, "Hybrid CNN")

plot_confusion_matrix(y_test, y_pred_hybrid, "Hybrid CNN")

print("\nClassification Report - Hybrid CNN:")
print(classification_report(y_test, y_pred_hybrid, target_names=["NORMAL", "MYOPIA"]))


def plot_model_comparison(metrics_dict):
    """Plot comparison of all models"""
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]["accuracy"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(models, accuracies, color=colors[: len(models)])

    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height / 2,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="white",
        )

    ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylim([0, 105])
    ax.grid(axis="y", alpha=0.3)

    save_figure("model_comparison.png")
    plt.show()


print("\n" + "=" * 80)
print("PART 2: IMAGE-BASED DETECTION")
print("=" * 80)

# Setup directories
train_dir = os.path.join(Config.IMAGE_DATA_DIR, "train")
test_dir = os.path.join(Config.IMAGE_DATA_DIR, "test")
val_dir = os.path.join(Config.IMAGE_DATA_DIR, "val")

# Count images
train_images = glob.glob(os.path.join(train_dir, "**/*.jpg"), recursive=True)
test_images = glob.glob(os.path.join(test_dir, "**/*.jpg"), recursive=True)
val_images = glob.glob(os.path.join(val_dir, "**/*.jpg"), recursive=True)

print(f"Training images: {len(train_images)}")
print(f"Testing images: {len(test_images)}")
print(f"Validation images: {len(val_images)}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.4,
    fill_mode="nearest",
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_set = train_datagen.flow_from_directory(
    train_dir,
    class_mode="binary",
    batch_size=Config.BATCH_SIZE,
    target_size=Config.IMG_SIZE,
    shuffle=True,
    classes=["NORMAL", "MYOPIA"],
)

validation_set = val_test_datagen.flow_from_directory(
    val_dir,
    class_mode="binary",
    batch_size=Config.BATCH_SIZE,
    target_size=Config.IMG_SIZE,
    shuffle=False,
    classes=["NORMAL", "MYOPIA"],
)

test_set = val_test_datagen.flow_from_directory(
    test_dir,
    class_mode="binary",
    batch_size=Config.BATCH_SIZE,
    target_size=Config.IMG_SIZE,
    shuffle=False,
    classes=["NORMAL", "MYOPIA"],
)

print("\n" + "=" * 60)
print("LABEL MAPPING VERIFICATION")
print("=" * 60)
print(f"ImageDataGenerator class indices: {train_set.class_indices}")
print("Expected: {'NORMAL': 0, 'MYOPIA': 1}")
print(f"Match: {train_set.class_indices == {'NORMAL': 0, 'MYOPIA': 1}}")
print("=" * 60)


print("\n" + "=" * 80)
print("TRAINING: IMAGE CNN MODEL")
print("=" * 80)

image_model = Sequential(
    [
        Conv2D(
            32, (3, 3), activation="relu", padding="same", input_shape=(224, 224, 3)
        ),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ],
    name="Image_CNN",
)

image_model.compile(
    optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"]
)

print(image_model.summary())

image_callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    ModelCheckpoint(
        os.path.join(Config.MODEL_DIR, "image_cnn_best.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

history_image = image_model.fit(
    train_set,
    epochs=Config.EPOCHS_IMAGE,
    validation_data=validation_set,
    steps_per_epoch=len(train_set),
    validation_steps=len(validation_set),
    callbacks=image_callbacks,
    verbose=1,
)

image_model.save(os.path.join(Config.MODEL_DIR, "image_cnn_final.h5"))
print("✓ Model saved: image_cnn_final.h5")

plot_training_history(history_image, "Image CNN")

test_loss, test_accuracy = image_model.evaluate(test_set)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

y_pred_image_proba = image_model.predict(test_set)
y_true_image = test_set.classes

optimal_threshold_image = calculate_optimal_threshold(
    y_true_image, y_pred_image_proba.flatten()
)
print(f"\n✓ Optimal threshold (Image CNN): {optimal_threshold_image:.4f}")

Config.PREDICTION_THRESHOLD = optimal_threshold_image

y_pred_image = (y_pred_image_proba >= optimal_threshold_image).astype(int).flatten()

image_metrics = calculate_comprehensive_metrics(
    y_true_image, y_pred_image, y_pred_image_proba
)
print_metrics_report(image_metrics, "Image CNN")

plot_confusion_matrix(y_true_image, y_pred_image, "Image CNN")


print("\n" + "=" * 80)
print("FINAL SUMMARY - ALL MODELS")
print("=" * 80)

all_metrics = {
    "Basic CNN": cnn_metrics,
    "Hybrid CNN": hybrid_metrics,
    "Image CNN": image_metrics,
}

save_metrics_to_csv(all_metrics)

plot_model_comparison(all_metrics)

summary_df = pd.DataFrame(all_metrics).T * 100
summary_df = summary_df.round(2)
print("\n" + summary_df.to_string())

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"✓ All models saved to: {Config.MODEL_DIR}")
print(f"✓ All outputs saved to: {Config.OUTPUT_DIR}")
print("✓ Optimal thresholds calculated and applied")
print("=" * 80)


def predict_myopia(image_path, model_path=None, threshold=None):
    """
    Make prediction on a single image with correct label mapping

    Args:
        image_path: Path to the image
        model_path: Path to saved model (optional)
        threshold: Custom threshold (optional, uses optimal if None)

    Returns:
        prediction, probability
    """
    if model_path is None:
        model_path = os.path.join(Config.MODEL_DIR, "image_cnn_best.h5")

    if threshold is None:
        threshold = Config.PREDICTION_THRESHOLD

    model = keras.models.load_model(model_path)

    img = load_img(image_path, target_size=Config.IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prob = model.predict(img_array)[0][0]

    prediction = "MYOPIA" if prob >= threshold else "NORMAL"

    # Display
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(
        f"Prediction: {prediction}\nProbability: {prob:.4f}\nThreshold: {threshold:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"\n{'=' * 60}")
    print(f"Prediction: {prediction}")
    print(f"Raw Probability: {prob:.4f}")
    print(f"Threshold Used: {threshold:.4f}")
    print(
        f"Classification: {'Class 1 (MYOPIA)' if prob >= threshold else 'Class 0 (NORMAL)'}"
    )
    print(f"{'=' * 60}")

    return prediction, prob
