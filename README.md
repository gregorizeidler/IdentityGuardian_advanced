# Identity Guardian

![Identity Guardian](https://img.shields.io/badge/Status-Beta-yellow)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

An advanced identity verification system using computer vision, OCR, and AI to validate user identities through facial and document analysis.

## 🚀 Features

- ✅ **Facial Detection & Analysis**: Detects and analyzes facial features 
- 🔍 **Image Quality Assessment**: Verifies brightness, contrast, and clarity
- 🔒 **Manipulation Detection**: Identifies possible face swaps or edited images
- 📝 **Document OCR**: Extracts text from identity documents
- 👥 **Facial Comparison**: Compares faces between photo and document
- 🤳 **Liveness Detection**: Anti-spoofing technology to prevent fraud
- 👤 **Age & Gender Estimation**: Verifies declared information
- 🏆 **Unified Trust Score System**: Calculates a reliability score based on multiple verifications
- 🖥️ **Webcam Detection**: Identifies if the image was captured by webcam for more accurate assessment
- 📊 **Depth Analysis**: Detects if the image is flat (like a photo of another photo) [Requires PyTorch]
- 🔎 **Advanced Screen/Spoofing Detection**: Detects screen/monitor patterns [Requires PyTorch]
- 🧠 **CNN Document Classification**: Automatically identifies document types [Requires PyTorch/TensorFlow]
- 🔍 **AI-Powered Advanced Analysis**: Optional advanced analysis using OpenAI

## 📋 Prerequisites

- Python 3.9 or higher
- Tesseract OCR installed on your system
- Camera for live validation (optional)
- OpenAI API key (optional, for advanced analysis)
- PyTorch (optional, for advanced detection features)
- TensorFlow (optional, for CNN document classification)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/IdentGuardian/IdentityGuardian.git
   cd IdentityGuardian
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate on macOS/Linux
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

3. **Install basic dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install advanced dependencies (optional)**
   ```bash
   # For advanced features (depth detection, CNN classification, etc.)
   pip install torch torchvision tensorflow
   ```

5. **Configure Tesseract OCR**
   - On macOS: `brew install tesseract`
   - On Ubuntu: `sudo apt install tesseract-ocr`
   - On Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

6. **Configure OpenAI API (optional)**
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your_key_here`

## 💻 Usage

1. **Start the application**
   ```bash
   # From the virtual environment
   python -m streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser at `http://localhost:8501`

3. **Upload documents and photos**
   - Follow the on-screen instructions to upload identity documents and personal photos
   - Fill in the required personal information

4. **Review verification results**
   - The system will analyze the documents and photos
   - A comprehensive report will be displayed with verification results and trust score

## 📝 Usage Tips

### Form Interaction

- **Multiple Buttons**: When using the main form, note that there are two submit buttons ("Verify Photo Quality" and "Preview Document Analysis"). In Streamlit, only one submit button per form will return `True` when clicked.
  
- **Recommended Verification Flow**:
  1. First, upload the images and fill in personal data
  2. Click "Verify Photo Quality" to get immediate feedback on quality issues
  3. After confirming the images are adequate, click "Analyze" to start the full verification

- **Processing Times**: Complete analysis may take a few seconds, especially when advanced features are enabled. Be patient during processing.

### Optimizing Results

- For best verification results, ensure that:
  - The personal photo has good lighting and is sharp
  - The document is photographed on a flat surface with all information visible
  - There are no significant reflections or shadows in the images
  - The face is clearly visible, without obstructions like sunglasses or masks

## 🔢 Trust Score System

Identity Guardian includes a unified scoring system that combines:

- Face swap detection
- Liveness detection
- Image editing detection
- Facial similarity score
- Depth analysis (when available)
- Screen/spoofing detection (when available)

Each component contributes to a final score of 0-100%, providing a clear assessment of identity verification reliability.

## 🏗 Project Structure

```
IdentityGuardian/
├── app.py                   # Main Streamlit application
├── models/                  # Models for advanced detection
├── uploads/                 # Directory for uploaded files
├── temp/                    # Temporary files
├── requirements.txt         # Project dependencies
├── .env                     # Environment variables (API keys)
└── README.md                # Documentation
```

## 🔍 How It Works

1. **Document Processing**
   - OCR technology extracts text from identity documents
   - Advanced image preprocessing improves extraction accuracy
   - Pattern recognition identifies key information (name, date of birth, gender)

2. **Facial Analysis**
   - Facial detection locates faces in photos and documents
   - Quality assessment analyzes lighting, contrast, and sharpness
   - Manipulation detection identifies potential edits or spoofing

3. **Advanced Anti-Spoofing Analysis**
   - Webcam detection analyzes image capture characteristics
   - Depth analysis detects if the image is flat (photo of another photo)
   - Screen/spoofing detection identifies monitor and display patterns

4. **Verification Logic**
   - Facial comparison correlates selfie with document photo
   - Demographic verification compares extracted data with declared information
   - Multiple validation points ensure comprehensive verification
   - Unified trust score provides overall assessment

5. **AI-Enhanced Analysis**
   - Optional integration with OpenAI for advanced analysis
   - Sophisticated fraud detection and risk assessment
   - Natural language reports of verification results

## 🔮 Advanced Features

The system is designed to work with different capability levels:

- **Basic Mode**: Uses only OpenCV and standard libraries for verification
- **Advanced Mode**: When PyTorch/TensorFlow are available, activates:
  - Depth analysis for photo-of-photo detection
  - Advanced screen/spoofing detection with Moiré pattern analysis
  - CNN document classification for automatic document type identification

The system automatically detects available libraries and enables corresponding features.

## 🤝 Contributing

Contributions are welcome! Feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Legal Disclaimer

This system is intended for demonstration and educational purposes. Implementation in production environments should include additional security measures and comply with relevant privacy regulations.

---

Made with ❤️ by IdentGuardian
