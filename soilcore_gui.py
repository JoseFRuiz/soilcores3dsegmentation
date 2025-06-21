import sys
import os
import numpy as np
import nibabel as nib
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QSlider, QHBoxLayout, QComboBox, QMessageBox, QSpinBox, QDialog)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils import (segment_single_file, MODEL_CONFIGS, process_images_for_topology, 
                   create_root_topology_plot)
from PIL import Image
import cv2
from skimage.morphology import skeletonize
from scipy.spatial.distance import euclidean
import pandas as pd
import glob
from tqdm import tqdm

MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'

# Map model file to display name and model_name for runsegmentation
MODEL_OPTIONS = [
    ('best_metric_modelunet_dataset_2_default.pth', 'UNet (Default)', 'unet_dataset_2_default', 'unet'),
    ('best_metric_modelunet_dataset_2_100k.pth', 'UNet (100k)', 'unet_dataset_2_100k', 'unet'),
    ('best_metric_modelsegresnet_dataset_2_default.pth', 'SegResNet', 'segresnet_dataset_2_default', 'segresnet'),
    ('best_metric_modeldynunet_dataset_2_100k.pth', 'DynUNet', 'dynunet_dataset_2_100k', 'dynunet'),
    ('best_metric_modeldataset_2adamw_100k_num_heads_2.pth', 'UNETR (AdamW, 100k, 2 heads)', 'dataset_2adamw_100k_num_heads_2', 'unetr'),
]

class NiftiViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Soilcore NIfTI Viewer')
        self.resize(900, 900)
        self.nii_img = None
        self.data = None
        self.original_data = None  # Store original data for thresholding
        self.slice_axis = 0
        self.current_slice = 0
        self.seg_output_path = None
        self.image_folder_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.info_label = QLabel('No file loaded.')
        layout.addWidget(self.info_label)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel('Segmentation Model:'))
        self.model_combo = QComboBox()
        for _, display, _, _ in MODEL_OPTIONS:
            self.model_combo.addItem(display)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        self.canvas = FigureCanvas(Figure(figsize=(4, 8)))
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()

        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton('Load NIfTI File')
        self.load_btn.clicked.connect(self.load_nifti)
        btn_layout.addWidget(self.load_btn)

        self.run_btn = QPushButton('Run Segmentation')
        self.run_btn.clicked.connect(self.run_segmentation)
        self.run_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)

        self.save_btn = QPushButton('Save Thresholded Images')
        self.save_btn.clicked.connect(self.save_thresholded_images)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)

        self.axis_slider = QSlider(Qt.Horizontal)
        self.axis_slider.setMinimum(0)
        self.axis_slider.setMaximum(2)
        self.axis_slider.setValue(0)
        self.axis_slider.setTickPosition(QSlider.TicksBelow)
        self.axis_slider.setTickInterval(1)
        self.axis_slider.valueChanged.connect(self.change_axis)
        btn_layout.addWidget(QLabel('Axis:'))
        btn_layout.addWidget(self.axis_slider)

        layout.addLayout(btn_layout)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.update_slice)
        layout.addWidget(self.slice_slider)

        # Threshold controls
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel('Lower Threshold:'))
        self.lower_threshold_slider = QSlider(Qt.Horizontal)
        self.lower_threshold_slider.setMinimum(0)
        self.lower_threshold_slider.setMaximum(100)
        self.lower_threshold_slider.setValue(0)
        self.lower_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.lower_threshold_slider.setTickInterval(10)
        self.lower_threshold_slider.valueChanged.connect(self.apply_thresholds)
        threshold_layout.addWidget(self.lower_threshold_slider)
        
        self.lower_threshold_label = QLabel('0.00')
        threshold_layout.addWidget(self.lower_threshold_label)
        
        threshold_layout.addWidget(QLabel('Upper Threshold:'))
        self.upper_threshold_slider = QSlider(Qt.Horizontal)
        self.upper_threshold_slider.setMinimum(0)
        self.upper_threshold_slider.setMaximum(100)
        self.upper_threshold_slider.setValue(100)
        self.upper_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.upper_threshold_slider.setTickInterval(10)
        self.upper_threshold_slider.valueChanged.connect(self.apply_thresholds)
        threshold_layout.addWidget(self.upper_threshold_slider)
        
        self.upper_threshold_label = QLabel('1.00')
        threshold_layout.addWidget(self.upper_threshold_label)
        
        layout.addLayout(threshold_layout)

        # Root topology analysis controls
        topology_layout = QHBoxLayout()
        topology_layout.addWidget(QLabel('Root Topology Analysis:'))
        
        self.select_folder_btn = QPushButton('Select Image Folder')
        self.select_folder_btn.clicked.connect(self.select_image_folder)
        topology_layout.addWidget(self.select_folder_btn)
        
        topology_layout.addWidget(QLabel('Pixel Range:'))
        self.pixel_range_spin = QSpinBox()
        self.pixel_range_spin.setMinimum(1)
        self.pixel_range_spin.setMaximum(10)
        self.pixel_range_spin.setValue(2)
        topology_layout.addWidget(self.pixel_range_spin)
        
        topology_layout.addWidget(QLabel('Num Ranges:'))
        self.num_ranges_spin = QSpinBox()
        self.num_ranges_spin.setMinimum(3)
        self.num_ranges_spin.setMaximum(10)
        self.num_ranges_spin.setValue(5)
        topology_layout.addWidget(self.num_ranges_spin)
        
        self.analyze_btn = QPushButton('Analyze Root Topology')
        self.analyze_btn.clicked.connect(self.analyze_root_topology)
        self.analyze_btn.setEnabled(False)
        topology_layout.addWidget(self.analyze_btn)
        
        layout.addLayout(topology_layout)

        self.setLayout(layout)

    def load_nifti(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open NIfTI file', '', 'NIfTI Files (*.nii *.nii.gz)')
        if file_path:
            self.nii_img = nib.load(file_path)
            self.data = self.nii_img.get_fdata()
            self.original_data = self.data.copy()  # Store original data
            self.slice_axis = 0
            self.current_slice = self.data.shape[self.slice_axis] // 2
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(self.data.shape[self.slice_axis] - 1)
            self.slice_slider.setValue(self.current_slice)
            self.update_info(file_path)
            self.show_slice()
            self.input_path = file_path
            self.run_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.seg_output_path = None
            # Reset thresholds when loading new file
            self.reset_thresholds()

    def reset_thresholds(self):
        """Reset threshold sliders to default values"""
        if self.original_data is not None:
            data_min = self.original_data.min()
            data_max = self.original_data.max()
            self.lower_threshold_slider.setValue(0)
            self.upper_threshold_slider.setValue(100)
            self.lower_threshold_label.setText(f'{data_min:.2f}')
            self.upper_threshold_label.setText(f'{data_max:.2f}')

    def apply_thresholds(self):
        """Apply lower and upper thresholds to the data"""
        if self.original_data is None:
            return
            
        data_min = self.original_data.min()
        data_max = self.original_data.max()
        
        # Convert slider values (0-100) to actual data range
        lower_val = data_min + (data_max - data_min) * self.lower_threshold_slider.value() / 100
        upper_val = data_min + (data_max - data_min) * self.upper_threshold_slider.value() / 100
        
        # Update labels
        self.lower_threshold_label.setText(f'{lower_val:.2f}')
        self.upper_threshold_label.setText(f'{upper_val:.2f}')
        
        # Apply thresholds
        self.data = np.copy(self.original_data)
        self.data[self.data < lower_val] = 0
        self.data[self.data > upper_val] = 0
        
        # Update display
        self.show_slice()

    def update_info(self, file_path):
        info = f"Loaded: {os.path.basename(file_path)}\nShape: {self.data.shape}\nMin: {self.data.min():.2f}, Max: {self.data.max():.2f}"
        self.info_label.setText(info)

    def change_axis(self, value):
        self.slice_axis = value
        if self.data is not None:
            self.slice_slider.setMaximum(self.data.shape[self.slice_axis] - 1)
            self.current_slice = self.data.shape[self.slice_axis] // 2
            self.slice_slider.setValue(self.current_slice)
            self.show_slice()

    def update_slice(self, value):
        self.current_slice = value
        self.show_slice()

    def show_slice(self):
        if self.data is not None:
            self.ax.clear()
            if self.slice_axis == 0:
                img = self.data[self.current_slice, :, :]
            elif self.slice_axis == 1:
                img = self.data[:, self.current_slice, :]
            else:
                img = self.data[:, :, self.current_slice]
            self.ax.imshow(img.T, cmap='gray', origin='lower')
            self.ax.set_title(f'Slice {self.current_slice} (Axis {self.slice_axis})')
            self.canvas.draw()

    def save_thresholded_images(self):
        """Save the current thresholded data as 2D PNG images"""
        if self.original_data is None:
            QMessageBox.warning(self, 'No Data', 'Please load a NIfTI file first.')
            return
        
        # Get output directory from user
        output_dir = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if not output_dir:
            return
        
        try:
            # Apply current thresholds to get the data to save
            data_min = self.original_data.min()
            data_max = self.original_data.max()
            
            lower_val = data_min + (data_max - data_min) * self.lower_threshold_slider.value() / 100
            upper_val = data_min + (data_max - data_min) * self.upper_threshold_slider.value() / 100
            
            # Apply thresholds to create binary data
            thresholded_data = np.copy(self.original_data)
            thresholded_data[thresholded_data < lower_val] = 0
            thresholded_data[thresholded_data > upper_val] = 0
            
            # Binarize: anything above 0 becomes 1
            binary_data = np.where(thresholded_data > 0, 1, 0)
            
            # Get base filename without extension
            if hasattr(self, 'input_path') and self.input_path:
                base_name = os.path.splitext(os.path.basename(self.input_path))[0]
                if base_name.endswith('.nii'):
                    base_name = base_name[:-4]
            else:
                base_name = "thresholded_data"
            
            # Create subdirectory for this dataset
            dataset_dir = os.path.join(output_dir, base_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Save 2D slices as PNG images
            # Assuming the data is in format (depth, height, width) like in gendata4RV.py
            # We'll save slices along the first axis (depth)
            data_to_save = np.swapaxes(binary_data, 0, 2)  # Swap axes like in gendata4RV.py
            
            for i in range(data_to_save.shape[1]):
                slice_data = data_to_save[:, i, :]
                # Convert to uint8 and scale to 0-255
                slice_image = Image.fromarray(np.uint8(slice_data * 255))
                slice_image.save(os.path.join(dataset_dir, f"{base_name}_{i:03}.png"))
            
            QMessageBox.information(self, 'Save Complete', 
                                  f'Saved {data_to_save.shape[1]} thresholded images to:\n{dataset_dir}\n\n'
                                  f'Threshold range: {lower_val:.2f} - {upper_val:.2f}')
            
        except Exception as e:
            QMessageBox.critical(self, 'Save Error', f'Error saving images: {str(e)}')

    def select_image_folder(self):
        """Select a folder containing 2D image files"""
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Image Folder')
        if folder_path:
            self.image_folder_path = folder_path
            # Check if folder contains image files
            image_files = glob.glob(os.path.join(folder_path, "*.png")) + \
                         glob.glob(os.path.join(folder_path, "*.jpg")) + \
                         glob.glob(os.path.join(folder_path, "*.tif"))
            if image_files:
                self.analyze_btn.setEnabled(True)
                QMessageBox.information(self, 'Folder Selected', 
                                      f'Selected folder: {folder_path}\nFound {len(image_files)} image files.')
            else:
                QMessageBox.warning(self, 'No Images', 'No image files found in the selected folder.')

    def analyze_root_topology(self):
        """Analyze root topology from 2D images and generate CSV with bar plot"""
        if not hasattr(self, 'image_folder_path'):
            QMessageBox.warning(self, 'No Folder', 'Please select an image folder first.')
            return
        
        try:
            # Disable button during processing
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText('Processing...')
            QApplication.processEvents()
            
            # Get parameters
            pixels_per_range = self.pixel_range_spin.value()
            num_ranges = self.num_ranges_spin.value()
            
            # Process images using utils function
            all_features = process_images_for_topology(
                self.image_folder_path, pixels_per_range, num_ranges
            )
            
            if all_features:
                # Generate CSV
                output_csv = os.path.join(self.image_folder_path, "root_length_by_diameter.csv")
                combined_features = pd.concat(all_features, ignore_index=True)
                combined_features.to_csv(output_csv, index=False)
                
                # Create and display bar plot using utils function
                plot_path = os.path.join(self.image_folder_path, "root_topology_plot.png")
                fig, ax = create_root_topology_plot(combined_features, pixels_per_range, num_ranges, plot_path)
                
                # Show the plot in a new window
                plot_dialog = QDialog(self)
                plot_dialog.setWindowTitle('Root Topology Analysis Results')
                plot_dialog.resize(800, 600)
                
                from PyQt5.QtWidgets import QVBoxLayout
                plot_layout = QVBoxLayout()
                plot_canvas = FigureCanvas(fig)
                plot_layout.addWidget(plot_canvas)
                plot_dialog.setLayout(plot_layout)
                
                plot_dialog.show()
                
                QMessageBox.information(self, 'Analysis Complete', 
                                      f'Root topology analysis complete!\nCSV saved to: {output_csv}')
            else:
                QMessageBox.warning(self, 'No Data', 'No valid root segments found in the images.')
                
        except Exception as e:
            QMessageBox.critical(self, 'Analysis Error', f'Error during analysis: {str(e)}')
        finally:
            # Re-enable button
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText('Analyze Root Topology')

    def run_segmentation(self):
        if not hasattr(self, 'input_path') or not self.input_path:
            QMessageBox.warning(self, 'No Input', 'Please load a NIfTI file first.')
            return
        
        model_idx = self.model_combo.currentIndex()
        _, _, model_name, _ = MODEL_OPTIONS[model_idx]
        
        try:
            # Disable the run button during processing
            self.run_btn.setEnabled(False)
            self.run_btn.setText('Processing...')
            QApplication.processEvents()  # Update the GUI
            
            # Run segmentation using utils function
            output_path = segment_single_file(
                input_file=self.input_path,
                model_name=model_name,
                output_dir=None,  # Will use default outputs/model_name
                threshold_value=None
            )
            
            # Load and display the result
            self.nii_img = nib.load(output_path)
            self.data = self.nii_img.get_fdata()
            self.original_data = self.data.copy()  # Store original segmented data
            self.slice_axis = 0
            self.current_slice = self.data.shape[self.slice_axis] // 2
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(self.data.shape[self.slice_axis] - 1)
            self.slice_slider.setValue(self.current_slice)
            self.update_info(output_path)
            self.show_slice()
            self.seg_output_path = output_path
            
            # Reset thresholds for the new segmented data
            self.reset_thresholds()
            
            # Enable save button after segmentation
            self.save_btn.setEnabled(True)
            
            QMessageBox.information(self, 'Segmentation Complete', f'Segmentation complete. Output saved to {output_path}')
            
        except Exception as e:
            QMessageBox.critical(self, 'Segmentation Error', f'Error during segmentation: {str(e)}')
        finally:
            # Re-enable the run button
            self.run_btn.setEnabled(True)
            self.run_btn.setText('Run Segmentation')

def main():
    app = QApplication(sys.argv)
    viewer = NiftiViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 