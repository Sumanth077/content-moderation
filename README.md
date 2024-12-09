# **Image Moderation Demo**  

## **Introduction to Content Moderation**  
Content moderation uses AI to automatically analyze and evaluate images, identifying inappropriate, unsafe, or harmful content. This ensures compliance, safety, and quality standards across platforms, making it essential for managing user-generated content.  

### **Key Benefits**  
- Automates the review of large volumes of content.  
- Enhances platform safety by detecting inappropriate material.  
- Supports compliance with regulations and community guidelines.  

---

## **Overview**  
This Streamlit application demonstrates the use of **Content Moderation** models trained on the Clarifai platform. It supports the following functionalities:  

### **1. Image Upload**  
- Upload images via URL or local file (Supported formats: JPEG, JPG, PNG, GIF).  

### **2. Model Analysis**  
- Analyze the uploaded image with advanced AI models.  
- Generate three detailed outputs:  
  1. **Visual Classification**:  
      - Classifies the image using models such as:  
        - `general-image-recognition`  
        - `moderation-recognition`  
        - `nsfw-recognition`  
  2. **Workflow Analysis**:  
      - Employs the [Image Moderation Workflow](https://clarifai.com/clarifai/Momio-Debug/workflows/Momio-Image-moderation-3), which includes:  
        - **`moderation-all-resnext-2`**: Detects offensive or inappropriate content.  
        - **`nsfw-recognition`**: Flags NSFW elements like nudity or adult themes.  
        - **`weapon-detection`**: Identifies weapons or security threats.  
        - **`moderation-multilingual-text-classification`**: Analyzes text within images.  
  3. **LLVM (Large Language Visual Models)**:  
      - Provides contextual descriptions of the image using models like GPT-4o and Claude-3 Vision.  

### **3. Sidebar Customization**  
- Select specific models for visual classification.  
- Adjust model parameters such as temperature and max tokens.  
- Customize the output display to meet your requirements.  

---

## **Setup**  
```bash
pip install -r requirements.txt  
streamlit run app.py  
```  

---

## **Configuration**  
- **Model Selection**: Choose from multiple pre-trained and custom-trained models.  
- **Parameter Adjustment**: Modify thresholds, temperature, and token limits.  
- **Output Customization**: Adjust colors and visualization options for results.  