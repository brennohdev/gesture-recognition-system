import pandas as pd
import os
import sys
import joblib  # This is used to save our model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Classifiers ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# --- Add src to Python path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Constants ---
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'gestures.csv')
MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
MODEL_FILE = os.path.join(MODELS_PATH, 'gesture_model.joblib')

# Make sure the /data/models directory exists
os.makedirs(MODELS_PATH, exist_ok=True)

def main():
    print(f"🚀 Starting model training...")
    
    # 1. --- Load Data ---
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        print("Please run 'scripts/collect_data.py' first.")
        return
        
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} samples from {DATA_FILE}")

    # 2. --- Split Features (X) and Labels (y) ---
    X = df.drop('label', axis=1)  # All columns EXCEPT 'label'
    y = df['label']               # Only the 'label' column

    # --- THIS IS THE FIX ---
    # Instead of dropping rows, we drop any COLUMNS (axis=1)
    # that contain NaN values. This removes the bad f17-f20 columns
    # but keeps all your collected data.
    X = X.dropna(axis=1, how='any')
    # --- END OF FIX ---
    
    # ML models need numbers, not text.
    # LabelEncoder turns 'fist', 'peace' into 0, 1, etc.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Found {len(le.classes_)} classes: {le.classes_}")

    # 3. --- Train-Test Split ---
    # We'll use 80% for training, 20% for testing
    # random_state=42 ensures we get the same "shuffle" every time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples.")

    # 4. --- Define Models ---
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    best_accuracy = 0.0
    best_model_name = ""
    best_model = None

    print("-" * 30)
    
    # 5. --- Train and Evaluate ---
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  -> {name} Accuracy: {accuracy * 100:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model

    print("-" * 30)
    print(f"🏆 Best model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy.")

    # 6. --- Save the Best Model ---
    if best_model:
        # We save both the model and the LabelEncoder (so we can decode 0,1,2..)
        model_data = {
            'model': best_model,
            'label_encoder': le
        }
        
        joblib.dump(model_data, MODEL_FILE)
        print(f"✅ Best model saved to: {MODEL_FILE}")
    else:
        print("Error: No model was trained successfully.")

if __name__ == "__main__":
    main()