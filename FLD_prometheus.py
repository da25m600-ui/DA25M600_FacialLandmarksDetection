import time
import torch
import matplotlib.pyplot as plt
import mlflow.pytorch
import pandas as pd
from prometheus_client import start_http_server, Counter, Gauge, Summary

# --- Prometheus Metrics Definition ---
# 1. End-to-end Latency
LATENCY_GAUGE = Gauge(
    'inference_latency_seconds',
    'End-to-end latency for the inference request in seconds'
)

# 2. Request Success / Failure Tracking
REQUEST_COUNTER = Counter(
    'inference_requests_total',
    'Total number of inference requests',
    ['status']
)

# 3. Input and Output Token Counts
TOKEN_COUNTER = Counter(
    'inference_tokens_total',
    'Total number of tokens processed',
    ['type']
)

# 4. Throughput (Images per second)
THROUGHPUT_SUMMARY = Summary(
    'inference_throughput_images_per_second',
    'Number of images processed per second'
)

# Define constants
MODEL_NAME = "Resnet_Facial_Landmarks_Model"
MODEL_VERSION = "1"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"


def run_inference():
    print(f"Loading model from registry: {MODEL_URI}...")

    # Track request status
    status = "failed"
    start_time = time.time()

    try:
        # Load the registered model as a PyFunc model
        model = mlflow.pytorch.load_model(model_uri=MODEL_URI)

        with torch.no_grad():
            best_network = model
            best_network.cuda()
            best_network.load_state_dict(torch.load('/content/face_landmarks.pth'))
            best_network.eval()

            # Assuming valid_loader and valid_dataset are defined globally in your notebook environment
            images, landmarks = next(iter(valid_loader))
            num_images = images.size(0)

            # --- Token Calculation Logic ---
            # For CV models, we simulate token count using image patches (e.g., 16x16 patches for a 224x224 image = 196 tokens)
            # Output tokens represent the 68 coordinates generated per image
            tokens_per_image_in = 196
            tokens_per_image_out = 68

            input_tokens = num_images * tokens_per_image_in
            output_tokens = num_images * tokens_per_image_out

            # Log tokens to Prometheus
            TOKEN_COUNTER.labels(type='input').inc(input_tokens)
            TOKEN_COUNTER.labels(type='output').inc(output_tokens)

            # Inference execution
            images = images.cuda()
            landmarks = (landmarks + 0.5) * 224
            predictions = (best_network(images).cpu() + 0.5) * 224
            predictions = predictions.view(-1, 68, 2)

            # Plotting logic
            plt.figure(figsize=(10, 40))
            for img_num in range(min(8, num_images)):
                plt.subplot(8, 1, img_num + 1)
                plt.imshow(images[img_num].cpu().numpy().transpose(1, 2, 0).squeeze(), cmap='gray')
                plt.scatter(predictions[img_num, :, 0], predictions[img_num, :, 1], c='r', s=5)
                plt.scatter(landmarks[img_num, :, 0], landmarks[img_num, :, 1], c='g', s=5)

        print('Total number of test images: {}'.format(len(valid_dataset)))
        status = "success"

    except Exception as e:
        status = "failed"
        print(f"An error occurred during inference: {e}")
        raise e

    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time

        # --- Update Prometheus Metrics ---
        LATENCY_GAUGE.set(elapsed_time)
        REQUEST_COUNTER.labels(status=status).inc()

        if status == "success" and elapsed_time > 0:
            throughput = num_images / elapsed_time
            THROUGHPUT_SUMMARY.observe(throughput)
            print(f"Throughput: {throughput:.2f} images/sec")

        print("Elapsed Time : {}".format(elapsed_time))


if __name__ == "__main__":
    # Start Prometheus scraping server on port 8000
    # Keep this port open so your Prometheus server can scrape data from http://localhost:8000
    PROMETHEUS_PORT = 8000
    print(f"Starting Prometheus metrics server on port {PROMETHEUS_PORT}...")
    start_http_server(PROMETHEUS_PORT)

    # Run the inference function
    run_inference()

    # If running as a persistent background service, prevent the script from exiting instantly:
    # while True: time.sleep(1)
