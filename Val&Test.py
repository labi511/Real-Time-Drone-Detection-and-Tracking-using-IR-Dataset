# Load the best trained model
model = YOLO('/kaggle/working/runs/drone-bird-detector2/weights/best.pt')

# Evaluate on validation set
metrics_val = model.val(data='data.yaml', split='val')  # prints metrics

metrics_test = model.val(data='data.yaml', split='test')

def print_metrics(results):
    print("\n--- Evaluation Metrics ---")
    print(f"Precision:     {results.box.p.mean():.4f}")
    print(f"Recall:        {results.box.r.mean():.4f}")
    print(f"mAP@0.5:       {results.box.map50.mean():.4f}")
    print(f"mAP@0.5:0.95:  {results.box.map.mean():.4f}")

    # Confusion Matrix (if available)
    try:
        print("\n--- Confusion Matrix ---")
        print(results.confusion_matrix.matrix)
    except:
        print("Confusion matrix not available.")

print_metrics(metrics_test)
