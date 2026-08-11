import mlflow.pytorch
import pandas as pd

# Define constants
# TODO: Replace '123456' with your actual roll number
#ROLL_NUMBER = "da25m600"
MODEL_NAME = "Resnet_Facial_Landmarks_Model"
MODEL_VERSION = "1"  # Or use a stage like 'Production' (e.g., f"models:/{MODEL_NAME}/Production")
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

#INPUT_FILE = "/content/test_processed_da25m600.parquet"
#OUTPUT_FILE = f"predictions_{ROLL_NUMBER}.csv"


def run_inference():
    print(f"Loading model from registry: {MODEL_URI}...")
    # Load the registered model as a PyFunc model
    model = mlflow.pytorch.load_model(model_uri=MODEL_URI)
    start_time = time.time()
    with torch.no_grad():
      best_network = model
      best_network.cuda()
      best_network.load_state_dict(torch.load('/content/face_landmarks.pth'))
      best_network.eval()
      images, landmarks = next(iter(valid_loader))
      images = images.cuda()
      landmarks = (landmarks + 0.5) * 224
      predictions = (best_network(images).cpu() + 0.5) * 224
      predictions = predictions.view(-1,68,2)
      plt.figure(figsize=(10,40))
      for img_num in range(8):
        plt.subplot(8,1,img_num+1)
        plt.imshow(images[img_num].cpu().numpy().transpose(1,2,0).squeeze(), cmap='gray')
        plt.scatter(predictions[img_num,:,0], predictions[img_num,:,1], c = 'r', s = 5)
        plt.scatter(landmarks[img_num,:,0], landmarks[img_num,:,1], c = 'g', s = 5)

    print('Total number of test images: {}'.format(len(valid_dataset)))
    end_time = time.time()
    print("Elapsed Time : {}".format(end_time - start_time))

if __name__ == "__main__":
    run_inference()