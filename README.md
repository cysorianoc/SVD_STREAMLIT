# Image Compression with Truncated SVD

This Streamlit app demonstrates image compression using Truncated Singular Value Decomposition (SVD). It allows users to explore how images can be approximated using a reduced number of components while retaining most of the original information.

## Features

- **Default sample image**: The app starts with a standard grayscale test image.
- **User image upload**: Users can upload their own images in JPG or PNG format. Uploaded images are automatically converted to grayscale.
- **Interactive SVD components**: A slider allows users to select the number of SVD components (`k`) used for reconstruction. Increasing `k` improves image quality, while decreasing `k` increases compression.
- **Side-by-side comparison**: The original and reconstructed images are displayed next to each other for easy visual comparison.
- **Compression metrics**: The app calculates and displays approximate image sizes in memory and the reconstruction mean squared error (MSE) to quantify quality.
- **Cumulative explained variance**: A plot shows how much of the image’s information is captured as the number of components increases. A 95% reference line indicates the number of components needed to retain most of the information.
- **Recommended components**: The app suggests an optimal `k` that retains roughly 95% of the original image variance.
- **Deployment guidance**: Instructions and a `requirements.txt` listing all necessary Python packages are provided for easy deployment on Streamlit or other platforms.

## Educational Purpose

This app is intended to help users understand:

- How SVD decomposes an image into singular values and vectors.
- The trade-off between image quality and compression.
- How dimensionality reduction can be applied.

By interacting with the slider and viewing side-by-side results, users can visually and quantitatively explore the effects of truncating singular values and gain intuition about data compression using SVD.
