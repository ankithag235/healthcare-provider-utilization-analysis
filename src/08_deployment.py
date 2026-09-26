import os
import joblib

MODEL_PATH = "models/best_model_cv_tuned.joblib"

print("Checking model...")
print("Model path:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found!")
else:
    print("Model file found!")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")
    print("Model type:", type(model))

    if hasattr(model, "steps"):
        print("\nPipeline steps:")

        for name, step in model.steps:
            print(f"  {name}: {type(step)}")

    if hasattr(model, "get_params"):
        print("\nModel parameters:")
        print(model.get_params())