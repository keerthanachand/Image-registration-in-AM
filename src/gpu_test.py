import tensorflow as tf

print("✅ TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"🎉 {len(gpus)} GPU(s) available:")
    for gpu in gpus:
        print("   🔹", gpu)
else:
    print("❌ No GPU found. Using CPU instead.")

