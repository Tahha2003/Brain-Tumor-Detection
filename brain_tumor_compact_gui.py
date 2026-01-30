import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from brain_tumor_detection import BrainTumorDetector

class CompactBrainTumorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain MRI Tumor Detection - Compact")
        
        # Set a reasonable window size that fits most screens
        self.root.geometry("900x600")
        self.root.minsize(800, 500)
        
        # Make window resizable
        self.root.resizable(True, True)
        
        self.detector = BrainTumorDetector()
        self.results = None
        self.current_view = 0  # Track which image set we're showing
        
        self.setup_gui()
    
    def setup_gui(self):
        # Configure root grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_container.columnconfigure(0, weight=2)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # Top control frame
        control_frame = ttk.Frame(main_container)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Load button
        self.load_btn = ttk.Button(
            control_frame, 
            text="Load MRI Image", 
            command=self.load_image,
            width=15
        )
        self.load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # View selection buttons
        ttk.Label(control_frame, text="View:").pack(side=tk.LEFT, padx=(10, 5))
        
        self.view_btn1 = ttk.Button(
            control_frame, 
            text="Original & Binary", 
            command=lambda: self.change_view(0),
            width=15
        )
        self.view_btn1.pack(side=tk.LEFT, padx=2)
        
        self.view_btn2 = ttk.Button(
            control_frame, 
            text="Watershed & Tumor", 
            command=lambda: self.change_view(1),
            width=15
        )
        self.view_btn2.pack(side=tk.LEFT, padx=2)
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_container, text="Image Analysis", padding="5")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure - smaller and more compact
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.fig.suptitle('Brain Tumor Detection', fontsize=12)
        
        # Create subplots
        self.axes = self.fig.subplots(1, 2)
        for ax in self.axes:
            ax.axis('off')
        
        # Canvas for matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results panel
        results_frame = ttk.LabelFrame(main_container, text="Analysis Results", padding="5")
        results_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Quick results display
        self.quick_results = ttk.Label(
            results_frame, 
            text="Load an image to start analysis",
            font=('Arial', 10, 'bold'),
            foreground='blue'
        )
        self.quick_results.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Detailed results text
        self.results_text = tk.Text(
            results_frame, 
            height=15, 
            width=30,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=('Consolas', 9)
        )
        self.results_text.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Scrollbar for text
        text_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        text_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=text_scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_container, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def load_image(self):
        """Load and process MRI image"""
        file_path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set("Processing image...")
            self.root.update()
            
            # Process the image
            self.results = self.detector.process_complete_pipeline(file_path)
            
            # Update displays
            self.update_quick_results()
            self.change_view(0)  # Start with first view
            self.update_detailed_results()
            
            self.status_var.set(f"Analysis complete - {self.results['category']}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred")
    
    def change_view(self, view_index):
        """Change which images are displayed"""
        if self.results is None:
            return
        
        self.current_view = view_index
        
        # Clear axes
        for ax in self.axes:
            ax.clear()
            ax.axis('off')
        
        if view_index == 0:
            # Show Original and Binary
            self.axes[0].imshow(self.results['original'], cmap='gray')
            self.axes[0].set_title('Original Grayscale MRI', fontsize=10)
            
            self.axes[1].imshow(self.results['binary'], cmap='gray')
            self.axes[1].set_title('Binary Image', fontsize=10)
            
        elif view_index == 1:
            # Show Watershed and Morphological Tumor
            self.axes[0].imshow(self.results['watershed'], cmap='viridis')
            self.axes[0].set_title('Watershed Segmentation', fontsize=10)
            
            self.axes[1].imshow(self.results['morphology_tumor'], cmap='gray')
            self.axes[1].set_title('Tumor Detection\n(White = Tumor)', fontsize=10)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_quick_results(self):
        """Update the quick results display"""
        if self.results is None:
            return
        
        area = self.results['tumor_area_cm2']
        category = self.results['category']
        
        # Color code the results
        if 'Malignant' in category:
            color = 'red'
        elif 'Benign' in category:
            color = 'orange'
        else:
            color = 'green'
        
        result_text = f"Area: {area:.3f} cm² | {category}"
        self.quick_results.config(text=result_text, foreground=color)
    
    def update_detailed_results(self):
        """Update the detailed results text area"""
        if self.results is None:
            return
        
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        text_content = f"""BRAIN TUMOR ANALYSIS
{'='*25}

RESULTS:
Area: {self.results['tumor_area_cm2']:.4f} cm²
Classification: {self.results['category']}

PROCESSING STEPS:
{'='*25}
✓ Grayscale conversion
✓ Binary thresholding (60%)
✓ Watershed segmentation
✓ Morphological operations
✓ Tumor detection

CLASSIFICATION CRITERIA:
{'='*25}
• No Tumor: 0 cm²
• Benign: ≤ 2.37 cm²
• Malignant: > 2.37 cm²

TUMOR DETECTION:
{'='*25}
White areas in the tumor 
detection image represent 
detected tumor regions.

Use the view buttons above 
to switch between different 
processing stages.

NAVIGATION:
{'='*25}
• "Original & Binary": 
  Shows input and threshold
• "Watershed & Tumor": 
  Shows segmentation and 
  final tumor detection
"""
        
        self.results_text.insert(1.0, text_content)
        self.results_text.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = CompactBrainTumorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()