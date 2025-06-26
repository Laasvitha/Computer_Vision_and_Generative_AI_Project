# 🖼️ AI Photo Editing with SAM & Stable Diffusion

This project demonstrates the integration of Meta’s **Segment Anything Model (SAM)** with **Stable Diffusion Inpainting** from Hugging Face Diffusers to create a powerful AI image editing tool.

The tool allows users to **select a subject** in an image by clicking on it, and then either:
- Replace the **background** with an AI-generated scene using a text prompt, or
- Replace the **subject itself** while keeping the background intact.

It’s a fun, interactive demonstration of modern **computer vision** and **generative AI** working together.

---

## 🌟 Features

- Select a subject in any uploaded image using point-based segmentation.
- Use a **text prompt** to generate a new background or subject.
- Live inpainting using the **Stable Diffusion XL Inpainting pipeline**.
- **Interactive Gradio app** for easy testing and experimentation.
- Mask visualization in real-time.
- Support for **prompt engineering**, seed control, and guidance scale tuning.

---

## 🚀 Getting Started

You can run the project in:
- Google Colab (recommended for access to GPU and easier setup)
- A local machine with a CUDA-enabled GPU (PyTorch + NVIDIA GPU)

---

## 📦 Dependencies

Make sure the following Python packages are installed:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate Pillow gradio opencv-python matplotlib

Install them with:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate Pillow gradio opencv-python matplotlib
```

---

## ⚙️ Installation Steps

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/sam-inpainting-app.git
cd sam-inpainting-app
```

2. **Download Pretrained Models**

SAM and the inpainting pipeline are automatically downloaded when first used:

* SAM (from `facebook/sam-vit-base`)
* Stable Diffusion XL Inpainting (from `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`)

> Make sure you're connected to the internet and logged in to the Hugging Face Hub (if required).

3. **Run the Notebook**

```bash
jupyter notebook ai_photo_editing_project.ipynb
```

Or open in Google Colab and execute cell-by-cell.

---

## 🖼️ How It Works

1. **Upload an Image**: Any image of a person, object, or animal.
2. **Click on the Subject**: Provide two points on the subject to help the SAM model understand what to segment.
3. **Mask Generation**: SAM returns a binary mask that highlights the object.
4. **Prompt Input**: Enter a prompt like `"a tropical jungle"` or `"a futuristic city"`.
5. **AI Inpainting**: The selected region (background or subject) is replaced using Stable Diffusion XL.
6. **Visual Result**: The final image is displayed interactively.

---

## 🧪 Testing & Usage

Once you’ve run the notebook:

1. ✅ Try segmenting different objects (cars, people, animals).
2. ✅ Replace only the **background** by using the default mask.
3. ✅ Replace the **subject** by inverting the mask (`~mask`).
4. ✅ Try multiple prompts and vary:

   * **Guidance Scale**: Lower (3–6) = looser, Higher (12–15) = more precise.
   * **Seed**: Controls randomness (use the same value to replicate output).
   * **Negative Prompt**: Helps remove unwanted artifacts.

---

## 🎮 Gradio App

To launch the interactive web app:

```python
import app
my_app = app.generate_app(get_processed_inputs, inpaint)
```

Then click the **public URL** to try it out live in your browser.

Inside the app, you can:

* Upload custom images
* Select regions
* Modify prompts
* View results instantly

To stop the app:

```python
my_app.close()
```



## 🧰 Built With

* **[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)** – Used for subject segmentation from clicks.
* **[Hugging Face Diffusers](https://huggingface.co/docs/diffusers)** – Used for AI-powered inpainting.
* **[Transformers](https://huggingface.co/docs/transformers)** – Used for tokenization and model loading.
* **[Gradio](https://gradio.app/)** – Used to create the interactive web UI.
* **[Pillow & OpenCV](https://pillow.readthedocs.io/)** – Used for image processing and mask manipulation.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE.txt) file for details.

---

## 🙌 Acknowledgements

* [Meta AI](https://ai.facebook.com/research) for SAM
* [Hugging Face](https://huggingface.co/) for the Diffusers & Transformers libraries
* [Stability AI](https://stability.ai/) for the Stable Diffusion models


