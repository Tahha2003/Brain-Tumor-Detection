import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, segmentation
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy import ndimage
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class BrainTumorDetector:
    def __init__(self):
        self.original_image = None
        self.gray_image = None
        self.binary_image = None
        self.watershed_image = None
        self.morphology_tumor = None
        self.threshold_image = None
        
        # Pixel size for area calculation (same as MATLAB)
        self.pixel_w = 0.0508
        self.pixel_h = 0.0508
        
    def load_image(self, image_path):
        """Load and preprocess the MRI image"""
        # Read image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError("Could not load image")
            
        # Resize to 200x200 (same as MATLAB)
        self.original_image = cv2.resize(self.original_image, (200, 200))
        
        # Convert to grayscale
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        return self.gray_image
    
    def create_binary_image(self, threshold=0.6):
        """Create binary image using threshold (equivalent to imbinarize)"""
        if self.gray_image is None:
            raise ValueError("No image loaded")
            
        # Normalize to 0-1 range
        normalized = self.gray_image.astype(np.float32) / 255.0
        
        # Apply threshold
        self.binary_image = (normalized > threshold).astype(np.uint8) * 255
        
        return self.binary_image
    
    def watershed_segmentation(self):
        """Apply Sobel filter and watershed segmentation"""
        if self.binary_image is None:
            raise ValueError("Binary image not created")
            
        # Convert to float for gradient calculation
        binary_float = self.binary_image.astype(np.float32) / 255.0
        
        # Sobel filters (equivalent to MATLAB's fspecial('sobel'))
        sobel_x = cv2.Sobel(binary_float, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(binary_float, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Watershed segmentation
        markers = np.zeros_like(gradient_magnitude, dtype=np.int32)
        markers[gradient_magnitude < 0.1] = 1
        markers[gradient_magnitude > 0.8] = 2
        
        # Apply watershed
        labels = watershed(gradient_magnitude, markers)
        
        # Convert to RGB for visualization
        self.watershed_image = (labels * 127).astype(np.uint8)
        
        return self.watershed_image
    
    def morphological_processing(self):
        """Apply morphological operations to detect tumor (white areas)"""
        if self.binary_image is None:
            raise ValueError("Binary image not created")
            
        # Create disk-shaped structuring element (equivalent to strel('disk',5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        # Opening operation (equivalent to imopen)
        opened = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)
        
        # Reconstruction (approximated using closing and opening)
        reconstructed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Dilation
        dilated = cv2.dilate(reconstructed, kernel, iterations=1)
        
        # Final reconstruction and complement operations
        # This approximates the MATLAB morphological reconstruction
        complement = cv2.bitwise_not(dilated)
        final_recon = cv2.morphologyEx(complement, cv2.MORPH_CLOSE, kernel)
        
        # Final tumor mask (white areas represent tumor)
        self.morphology_tumor = cv2.bitwise_not(final_recon)
        
        return self.morphology_tumor
    
    def threshold_segmentation(self):
        """Apply Otsu thresholding for segmentation"""
        if self.gray_image is None:
            raise ValueError("No image loaded")
            
        # Otsu thresholding (equivalent to graythresh + imbinarize)
        _, self.threshold_image = cv2.threshold(
            self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Remove small objects (equivalent to bwareaopen)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            self.threshold_image, connectivity=8
        )
        
        # Filter out small components (area < 50 pixels)
        min_area = 50
        filtered = np.zeros_like(self.threshold_image)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == i] = 255
                
        self.threshold_image = filtered
        
        return self.threshold_image
    
    def calculate_tumor_area(self):
        """Calculate tumor area and classify"""
        if self.morphology_tumor is None:
            raise ValueError("Morphological processing not completed")
            
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            self.morphology_tumor, connectivity=8
        )
        
        # Calculate total tumor area in pixels
        tumor_pixels = 0
        for i in range(1, num_labels):  # Skip background
            tumor_pixels += stats[i, cv2.CC_STAT_AREA]
        
        # Convert to cm²
        tumor_cm2 = tumor_pixels * self.pixel_w * self.pixel_h
        
        # Classify tumor
        if tumor_cm2 == 0:
            category = "No Tumor"
        elif tumor_cm2 <= 2.37:
            category = "Benign Tumor"
        else:
            category = "Malignant Tumor"
            
        return tumor_cm2, category
    
    def process_complete_pipeline(self, image_path):
        """Run the complete tumor detection pipeline"""
        # Load image
        self.load_image(image_path)
        
        # Create binary image
        self.create_binary_image()
        
        # Watershed segmentation
        self.watershed_segmentation()
        
        # Morphological processing (main tumor detection)
        self.morphology_tumor = self.morphological_processing()
        
        # Threshold segmentation
        self.threshold_segmentation()
        
        # Calculate tumor area
        tumor_area, category = self.calculate_tumor_area()
        
        return {
            'original': self.gray_image,
            'binary': self.binary_image,
            'watershed': self.watershed_image,
            'morphology_tumor': self.morphology_tumor,
            'threshold': self.threshold_image,
            'tumor_area_cm2': tumor_area,
            'category': category
        }

def main():
    """Main function to run tumor detection"""
    # Create detector instance
    detector = BrainTumorDetector()
    
    # File dialog to select image
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select MRI Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    
    if not file_path:
        print("No file selected")
        return
    
    try:
        # Process the image
        results = detector.process_complete_pipeline(file_path)
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Brain Tumor Detection Results', fontsize=16)
        
        # Original grayscale
        axes[0, 0].imshow(results['original'], cmap='gray')
        axes[0, 0].set_title('Grayscale MRI')
        axes[0, 0].axis('off')
        
        # Binary image
        axes[0, 1].imshow(results['binary'], cmap='gray')
        axes[0, 1].set_title('Binary Image')
        axes[0, 1].axis('off')
        
        # Watershed segmentation
        axes[0, 2].imshow(results['watershed'], cmap='viridis')
        axes[0, 2].set_title('Watershed Segmentation')
        axes[0, 2].axis('off')
        
        # Morphological tumor (WHITE areas are tumor)
        axes[1, 0].imshow(results['morphology_tumor'], cmap='gray')
        axes[1, 0].set_title('Morphological Tumor (White = Tumor)')
        axes[1, 0].axis('off')
        
        # Threshold segmentation
        axes[1, 1].imshow(results['threshold'], cmap='gray')
        axes[1, 1].set_title('Thresholding Segmentation')
        axes[1, 1].axis('off')
        
        # Text results
        axes[1, 2].text(0.1, 0.7, 'Tumor Analysis', fontsize=14, fontweight='bold')
        axes[1, 2].text(0.1, 0.6, '-------------------', fontsize=12)
        axes[1, 2].text(0.1, 0.5, f'Area (cm²): {results["tumor_area_cm2"]:.4f}', fontsize=12)
        axes[1, 2].text(0.1, 0.4, f'Category: {results["category"]}', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print results
        print("\n" + "="*50)
        print("BRAIN TUMOR DETECTION RESULTS")
        print("="*50)
        print(f"Tumor Area: {results['tumor_area_cm2']:.4f} cm²")
        print(f"Classification: {results['category']}")
        print("="*50)
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()