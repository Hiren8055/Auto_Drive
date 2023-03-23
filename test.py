import os
import glob
import torch
import tqdm
import cv2
from tqdm import tqdm_notebook
from main.preprocess import tensorize_image
import numpy as np
from main.constant import *
# from train import *

if not os.path.exists(PREDICT_DIR): #PREDICT_DIR yolunda predicts klasörü yoksa yeni klasör oluştur.
    os.mkdir(PREDICT_DIR)

#### PARAMETERS #####
cuda = True
model_name = "Unet_2.pt"

predict_path = os.path.join(PREDICT_DIR, model_name.split(".")[0])
if not os.path.exists(predict_path): #predict_path yolunda predicts klasörü yoksa yeni klasör oluştur.
    os.mkdir(predict_path)

model_path = os.path.join(MODELS_DIR, model_name)
input_shape = input_shape
#####################

# LOAD MODEL
model = torch.load(model_path)
#Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
#Failing to do this will yield inconsistent inference results.
model.eval()
print("model is loaded and evaluated")

'''if cuda:
    model = model.cuda()'''



# PREDICT
device = "cpu"
def predict(model):


        # for image in tqdm_notebook(images):
    image_path = r"D:\Autonomous\Freespace_Segmentation-Ford_Otosan_Intern\src\final_image_o\masked 0.png"
    #batch_test = tensorize_image([image], input_shape, cuda)
    img = cv2.imread(image_path) # Access and read image
    
    zeros_img = np.zeros((1920, 1208))
    norm_img = cv2.normalize(img, zeros_img, 0, 255, cv2.NORM_MINMAX)
    
    img = cv2.resize(norm_img, output_shape, interpolation = cv2.INTER_NEAREST) # Resize the image according to defined shape
    
    # Change input structure according to pytorch input structure
    from main.preprocess import torchlike_data
    torchlike_image = torchlike_data(img)
    batch_images = []
    batch_images.append(torchlike_image) # Add into the list

    # Convert from list structure to torch tensor
    image_array = np.array(batch_images, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()
    # The tensor should be in [batch
    output = model(torch_image)
    out = torch.argmax(output, axis=1)


    out = out.cpu()

    outputs_list  = out.detach().numpy()
    mask = np.squeeze(outputs_list, axis=0)

    mask_uint8 = mask.astype('uint8')
    #mask_resize = cv2.resize(mask_uint8, (1920, 1208), interpolation = cv2.INTER_CUBIC)

    img = cv2.imread(image_path)
    #img_resize = cv2.resize(img, input_shape)
    mask_ind = mask_resize = 1
    #copy_img = img_resize.copy()
    copy_img = img.copy()
    img[mask_uint8==1, :] = (255, 0, 125)
    opac_image = (img/2 + copy_img/2).astype(np.uint8)
    cv2.imwrite(os.path.join(predict_path, image_path.split("/")[-1]), opac_image)
    #print("mask size from model: ", mask.shape),
    #print("resized mask size: ", mask_resize.shape)

if __name__ == "__main__":
    predict(model)
