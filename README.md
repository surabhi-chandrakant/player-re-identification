# Player Re-identification in a Single Feed

This project implements a basic real-time player re-identification and tracking system for a single video feed. It uses a pre-trained YOLOv11 model for object detection and employs IoU-based tracking with a feature-based re-identification mechanism to maintain player identities even when they temporarily leave the frame.

## Objective

Given a 15-second video (`15sec_input_720p.mp4`), the goal is to:
1.  Detect all players throughout the clip.
2.  Assign unique IDs to players based on their initial appearance.
3.  Maintain the same ID for players who go out of frame and re-appear later in the video (e.g., near a goal event).
4.  Simulate real-time re-identification and player tracking, outputting a video with identified players.

## Setup and Running Instructions

Follow these steps to set up your environment and run the code.

### Prerequisites

* **Python 3.x:** Ensure you have Python installed. You can download it from [python.org](https://www.python.org/).
* **Git (Optional but Recommended):** For cloning repositories and managing code.

### 1. Project Setup

1.  **Create a Project Directory:**
    ```bash
    mkdir re_identification_assignment
    cd re_identification_assignment
    ```
2.  **Download the YOLOv11 Model:**
    * Go to the provided Google Drive link: [https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)
    * Download the `yolov11_model.pt` file.
    * Place this downloaded file directly into your `re_identification_assignment` directory.

3.  **Place the Input Video:**
    * Place your input video file, named `15sec_input_720p.mp4`, directly into the `re_identification_assignment` directory.

4.  **Save the Python Script:**
    * Save the provided `main.py` code (from this response) into the `re_identification_assignment` directory.

### 2. Environment Setup

It's highly recommended to use a virtual environment to manage dependencies.

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```
2.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    (You will see `(venv)` prefix in your terminal, indicating the environment is active.)

### 3. Install Dependencies

Install the required Python packages using pip:

```bash
pip install opencv-python numpy ultralytics