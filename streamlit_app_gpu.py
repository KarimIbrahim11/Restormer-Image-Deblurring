'''

This streamlit service can only run locally on your machine as it support GPU inference.
'streamlit run streamlit_app_gpu.py' will do the job :)

'''

import torch
import os
import torch.nn.functional as F
import streamlit as st
from runpy import run_path
from PIL import Image, ImageFilter
from skimage.util import img_as_ubyte
from io import BytesIO
import numpy as np


# Loading the model Once for Optimization
@st.cache_resource
def loadModel():
    # Get model weights and parameters
    parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8],
                  'num_refinement_blocks': 4,
                  'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66, 'bias': False, 'LayerNorm_type': 'WithBias',
                  'dual_pixel_task': False}
    weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    # Load Model Arch and convert to CUDA
    load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)
    model.cuda()
    # Load weights&Params to Model
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])
    print("************** Model Loaded **************")
    return model, weights


# Gets the download Link
def downloadImgBtn(img):
    # Original image came from cv2 format, fromarray convert into PIL format
    # Convert to Bytes
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return st.download_button(
        label="Download Restored Image",
        data=byte_im,
        file_name="restored.jpeg",
        mime="image/jpeg",
    )


if __name__ == '__main__':
    # Task
    task = 'Motion_Deblurring'

    # Model
    model, weights = loadModel()

    st.title("Text Image Enhancement")
    st.caption("An Assessment Task for Blnk")

    # Image Uploader
    st.subheader("Image Uploader:")
    extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
    uploaded_file = st.file_uploader("Choose an image...", type=extensions)
    if uploaded_file is not None:
        # Displaying the image
        st.subheader("Uploaded Image")
        pilImg = Image.open(uploaded_file)
        st.image(pilImg, caption='Uploaded Image', use_column_width=True)
        pilImg.save("img." + pilImg.format)

        # Give the user an on-screen indication that we are working
        onscreen = st.empty()
        onscreen.text('Restoring...')

        # Inference
        img_multiple_of = 8
        print(f"\n ==> Running {task} with weights {weights}\n ")
        with torch.no_grad():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = Image.open(uploaded_file).convert('RGB')
            img = np.array(img)
            # # Convert RGB to BGR
            # img = open_cv_image[:, :, ::-1].copy()
            # img = cv2.cvtColor(cv2.imread("img." + pilImg.format), cv2.COLOR_BGR2RGB)
            input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).cuda()

            # Pad the input if not_multiple_of 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (w + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - h if h % img_multiple_of != 0 else 0
            padw = W - w if w % img_multiple_of != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model(input_)
            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:, :, :h, :w]
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])

            # # Post Processing
            # # Image Sharpening
            sharpen_filter = [
                0, -1, 0,
                -1, 5, -1,
                0, -1, 0
            ]
            im = Image.fromarray(restored.astype('uint8'), 'RGB')
            # Create a kernel filter using the defined matrix
            kernel_filter = ImageFilter.Kernel((3, 3), sharpen_filter)
            # Apply the kernel filter to the image
            restored = im.filter(kernel_filter)

            # Show the user that we have finished
            onscreen.empty()
            st.subheader("Restored Image")
            st.image(restored, caption='Restored Image', use_column_width=True)

            # Downloading Image Button
            btn = downloadImgBtn(restored)
