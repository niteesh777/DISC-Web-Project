# DISC Web App (Digital Image Speckle Correlation): Asynchronous Facial Symmetry Analysis Platform

This Django-based web application provides a frontend for uploading paired facial images and performing a full symmetry analysis pipeline using:

* **Face detection & cropping**
* **Digital Image Correlation (PyDIC)**
* **Facial region segmentation (SAM - Segment Anything Model)**
* **Vector and heatmap visualization**
* **Symmetry score extraction for cheeks and forehead**

All image processing is executed asynchronously using **Celery + Redis** for efficient background task handling.

---

## 📁 Project Structure Overview

```
.
├── manage.py
├── ImageProcessingWeb/               # Django project folder
│   ├── settings.py                   # Celery, media paths, Redis config
├── processing_app/                  # Custom Django app
│   ├── views.py                     # Handles upload & Celery trigger
│   ├── templates/                   # Upload form + result page
│   ├── your_processing_function.py  # Main DISC + SAM integration logic
│   ├── pydic.py                     # Local DIC processor
├── media/                           # Uploaded and processed files
│   ├── uploads/                     # Raw input images
│   └── results/                     # All outputs (heatmaps, csv, etc.)
├── models/                          # Contains sam_vit_h_4b8939.pth
├── requirements.txt
├── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/niteesh777/DISC-Web-Project.git
cd DISC-Web-Project
```

### 2. Create and activate virtual environment

```bash
python3 -m venv disc
source disc/bin/activate            # On Windows: disc\Scripts\activate
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

### 4. Download SAM model checkpoint

Upload the `sam_vit_h_4b8939.pth` file manually to the following path:

```bash
imageapp/models/sam_vit_h_4b8939.pth
```

This file is too large to be stored in GitHub and must be downloaded from the official Segment Anything repository:
[https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

---

## 🔧 Running the Application

### Apply Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### Run Django Server

```bash
python manage.py runserver
```

### Start Celery Worker (in another terminal)

```bash
celery -A ImageProcessingWeb worker -l info
```

### Start Redis Server (in another terminal)

```bash
redis-server
```

---

## ⚙️ Why Celery and Redis?

**Celery** allows running long background tasks (e.g., DIC + SAM processing) without blocking the web UI.

**Redis** acts as a **message broker** for Celery. When a user uploads images:

* Django creates a Celery job
* Celery fetches the job from Redis and runs it
* Results are saved and made accessible once ready

Benefits:

* Avoids request timeouts
* Allows scaling workers separately
* Prepares app for production-level traffic

---

## 🖼️ Web Application Flow (Screenshots)

### 📏 Why Image Alignment Matters

For accurate **Digital Image Speckle Correlation (DISC)** analysis, the face in both images must remain in the **same position and orientation** within the frame. Even slight movements can distort the deformation analysis, leading to incorrect symmetry scores.

#### ✅ Correct Alignment

When faces are well-aligned, the system can reliably detect corresponding facial regions across both images.

#### ❌ Incorrect Alignment

If the subject has moved significantly (e.g., head tilted, changed expression, shifted in frame), the results may be invalid, and you'll receive an error asking to re-upload better-aligned images.

1. **Home Page**
   <img width="1702" alt="Intro1" src="https://github.com/user-attachments/assets/202f0e3b-f1a2-497a-b88d-c26a4cf8d4c1" />


2. **Image Upload Interface**
   <img width="1702" alt="Step2" src="https://github.com/user-attachments/assets/df8d4b7f-617a-4f1c-a52d-efe733e66b1d" />

  3. **Correct Image Alignment Feedback**
     <img width="1708" alt="RIght_alignement_Success_Processing4 1" src="https://github.com/user-attachments/assets/d81d2790-d423-4bed-a2a6-6db43214e8ac" />

     **Incorrect Alignment Warning**
    <img width="1708" alt="Wrong_Alignment of Images4 2" src="https://github.com/user-attachments/assets/9e4169c4-5aa6-4f6e-ac15-14626b6e34b4" />


5. **Successful Processing Example**
   <img width="645" alt="Final_Result" src="https://github.com/user-attachments/assets/5c2ae95c-f9ae-40a0-8969-72a79d703646" />


---

## 📤 Output Artifacts

Saved under `/media/results/`:

* `Cropped/` – Cropped face images
* `csv/` – Filtered DIC displacement data
* `OVERLAY/Heatmap/` – Region heatmaps
* `Symmetry_Scores/` – Overlayed visuals with scores
* `symmetry_scores.csv` – Quantified region asymmetry

---

## 📦 Requirements

See `requirements.txt`. Includes:

* Django, Celery, Redis
* face\_recognition, dlib, SAM
* matplotlib, numpy, scipy, pandas, opencv

---

## 📄 License

MIT License — free to modify, distribute, and build upon.

---

## 👤 Author

**Niteesh Reddy Adavelli**

[GitHub](https://github.com/niteesh777)

---

For improvements, contributions, or issues, feel free to open a pull request or GitHub issue.

