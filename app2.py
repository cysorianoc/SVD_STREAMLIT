# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from PIL import Image
from skimage import data
from io import BytesIO

# Function to get image size in KB
def get_image_kb(np_img):
    pil_img = Image.fromarray(np_img.astype(np.uint8))
    buf = BytesIO()
    pil_img.save(buf, format="PNG")  # save as PNG in memory
    size_kb = len(buf.getvalue()) / 1024
    return size_kb

st.title("📉 Image Compression with Truncated SVD")

# Step 1: Default sample image
sample_img = data.camera()   # grayscale sample (512x512)
I = sample_img.astype(np.float32)
H, W = I.shape

# Step 2: User upload
uploaded_file = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("L")  # convert to grayscale
    I = np.array(pil_img).astype(np.float32)
    H, W = I.shape
    st.success(f"✅ Custom image loaded ({H}×{W})")
else:
    st.info("ℹ️ No image uploaded. Using default sample (camera test image).")

# Step 3: Choose SVD components
k = st.slider("Number of SVD components (k)", 1, min(H, W), 50)

# Step 4: Apply Truncated SVD
svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
scores = svd.fit_transform(I)
I_hat = svd.inverse_transform(scores)


# Step 4b: Recommend optimal k for 95% variance
full_k = min(H, W)  # max possible components
svd_full = TruncatedSVD(n_components=full_k, n_iter=7, random_state=42)
_ = svd_full.fit_transform(I)
cumvar_full = svd_full.explained_variance_ratio_.cumsum()
k95 = int(np.argmax(cumvar_full >= 0.95) + 1)  # smallest k reaching 95%
st.info(f"💡 Recommended number of components to retain ~95% variance: k = {k95}")




# Clip and convert for display
I_hat_disp = np.clip(I_hat, 0, 255).astype(np.uint8)


# Compute sizes
orig_kb = get_image_kb(I)
recon_kb = get_image_kb(I_hat_disp)


# Step 5: Show original and reconstructed images side by side
col_orig, col_recon = st.columns(2)

col_orig.subheader("Original Image")
col_orig.image(I.astype(np.uint8), use_container_width=True, caption=f"Size: {orig_kb:.1f} KB")

col_recon.subheader(f"Reconstructed  (k={k})")
col_recon.image(I_hat_disp, use_container_width=True, caption=f"Size: {recon_kb:.1f} KB")

# Step 6: Plot cumulative explained variance
fig, ax = plt.subplots()
cumvar = svd.explained_variance_ratio_.cumsum()
ax.plot(np.arange(1, len(cumvar) + 1), cumvar, marker="o")
ax.axhline(0.95, color="r", linestyle="--", label="95% threshold")
ax.set_xlabel("Number of components (k)")
ax.set_ylabel("Cumulative explained variance")
ax.set_ylim([0, 1.02])
ax.legend()
st.pyplot(fig)

# Step 7: Show numeric quality metrics
mse = np.mean((I - I_hat) ** 2)
st.write(f"Reconstruction MSE: {mse:.2f}")

# Step 8: Tips for performance
st.markdown(
    """
    **Tips**
    - If the "Compute full SVD" step is slow, try downsampling the image before computing (e.g., resize to 256×256).
    - Use `k` around the `k95` recommendation if you want ~95% of variance.
    - `TruncatedSVD` treats each **row** as a sample and each column as a feature (so the algorithm is performing SVD on the image matrix).
    """
)

# Step 9: Show requirements.txt for deployment
st.markdown("**Note**: To deploy the app you need a `requirements.txt` file with:")

st.code(
    """streamlit
scikit-learn
numpy
matplotlib
pillow
scikit-image""",
    language="text"
)
