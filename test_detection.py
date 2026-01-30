#!/usr/bin/env python3
"""
Test script for brain tumor detection
"""

import os
import glob
from brain_tumor_detection import BrainTumorDetector
import matplotlib.pyplot as plt

def test_all_images():
    """Test tumor detection on all available images"""
    
    # Find all image files in current directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(ext))
    
    if not image_files:
        print("No image files found in current directory")
        return
    
    print(f"Found {len(image_files)} image files")
    print("Testing brain tumor detection...")
    print("="*50)
    
    detector = BrainTumorDetector()
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n{i}. Processing: {image_file}")
        
        try:
            results = detector.process_complete_pipeline(image_file)
            
            print(f"   Tumor Area: {results['tumor_area_cm2']:.4f} cm²")
            print(f"   Classification: {results['category']}")
            
            # Save result visualization
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            fig.suptitle(f'Results for {image_file}', fontsize=14)
            
            # Display all processing steps
            axes[0, 0].imshow(results['original'], cmap='gray')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(results['binary'], cmap='gray')
            axes[0, 1].set_title('Binary')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(results['watershed'], cmap='viridis')
            axes[0, 2].set_title('Watershed')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(results['morphology_tumor'], cmap='gray')
            axes[1, 0].set_title('Tumor (White)')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(results['threshold'], cmap='gray')
            axes[1, 1].set_title('Threshold')
            axes[1, 1].axis('off')
            
            # Results text
            axes[1, 2].text(0.1, 0.7, f'Area: {results["tumor_area_cm2"]:.4f} cm²', fontsize=12)
            axes[1, 2].text(0.1, 0.5, f'Type: {results["category"]}', fontsize=12)
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Save the result
            output_name = f"result_{os.path.splitext(image_file)[0]}.png"
            plt.savefig(output_name, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   Result saved as: {output_name}")
            
        except Exception as e:
            print(f"   Error processing {image_file}: {e}")
    
    print("\n" + "="*50)
    print("Testing completed!")

if __name__ == "__main__":
    test_all_images()