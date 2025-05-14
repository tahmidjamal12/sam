import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from matplotlib.backends.backend_pdf import PdfPages

def create_contour_visualization(image_path, segments, image_id):
    """
    Create a visualization with segment contours overlaid on the original image.
    
    Args:
        image_path (str): Path to the original image
        segments (dict): Dictionary of segment arrays
        image_id (str): ID of the image
    """
    # Read the original image
    img = np.array(Image.open(image_path))
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Display the original image
    plt.imshow(img)
    
    # Overlay contours for each segment
    colors = ['red', 'green', 'blue', 'yellow', 'cyan']  # Different colors for different segments
    legend_elements = []
    
    for i, (segment_id, segment) in enumerate(segments.items()):
        color = colors[i % len(colors)]
        # Create contour plot
        plt.contour(segment, colors=color, linewidths=2, alpha=0.7)
        # Add to legend elements
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=f'Segment {segment_id}'))
    
    plt.title(f'Image {image_id} with Segment Contours')
    plt.axis('off')
    
    # Add legend
    plt.legend(handles=legend_elements,
              loc='upper right',
              bbox_to_anchor=(1.15, 1),
              frameon=True,
              facecolor='white',
              edgecolor='black')
    
    return plt.gcf()

def save_all_contours_to_pdf():
    """
    Process all images and save their contour visualizations to a single PDF file.
    """
    # Create PDF file
    pdf_path = "segment_contours.pdf"
    with PdfPages(pdf_path) as pdf:
        # Process main directory
        if os.path.exists("annotations"):
            for file in os.listdir("annotations"):
                if file.endswith("_annotation.hdf5"):
                    image_id = file.split('_')[0]
                    file_path = os.path.join("annotations", file)
                    
                    # Try to find the image in different possible locations
                    possible_image_paths = [
                        f"rgb_images/{image_id}_rgb.png",
                        f"rgb_images/Easy/{image_id}_rgb.png",
                        f"rgb_images/Medium/{image_id}_rgb.png",
                        f"rgb_images/Hard/{image_id}_rgb.png"
                    ]
                    
                    image_path = None
                    for path in possible_image_paths:
                        if os.path.exists(path):
                            image_path = path
                            break
                    
                    if image_path is None:
                        print(f"Warning: Could not find image for {image_id}")
                        continue
                    
                    # Read segments from HDF5 file
                    with h5py.File(file_path, 'r') as f:
                        segments = {}
                        for key in f['segments'].keys():
                            segments[key] = f[f'segments/{key}'][:]
                        
                        # Create and save visualization
                        fig = create_contour_visualization(image_path, segments, image_id)
                        pdf.savefig(fig)
                        plt.close(fig)
        
        # Process subdirectories
        subdirs = ["Easy", "Medium", "Hard"]
        for subdir in subdirs:
            annotations_dir = os.path.join("annotations", subdir)
            if os.path.exists(annotations_dir):
                for file in os.listdir(annotations_dir):
                    if file.endswith("_annotation.hdf5"):
                        image_id = file.split('_')[0]
                        file_path = os.path.join(annotations_dir, file)
                        
                        # Try to find the image in different possible locations
                        possible_image_paths = [
                            f"rgb_images/{image_id}_rgb.png",
                            f"rgb_images/Easy/{image_id}_rgb.png",
                            f"rgb_images/Medium/{image_id}_rgb.png",
                            f"rgb_images/Hard/{image_id}_rgb.png"
                        ]
                        
                        image_path = None
                        for path in possible_image_paths:
                            if os.path.exists(path):
                                image_path = path
                                break
                        
                        if image_path is None:
                            print(f"Warning: Could not find image for {image_id}")
                            continue
                        
                        # Read segments from HDF5 file
                        with h5py.File(file_path, 'r') as f:
                            segments = {}
                            for key in f['segments'].keys():
                                segments[key] = f[f'segments/{key}'][:]
                            
                            # Create and save visualization
                            fig = create_contour_visualization(image_path, segments, image_id)
                            pdf.savefig(fig)
                            plt.close(fig)
    
    print(f"\nContour visualizations have been saved to {pdf_path}")

def count_images():
    """
    Count all images in rgb_images directory and its subdirectories.
    """
    total_count = 0
    counts = {
        "main": 0,
        "Easy": 0,
        "Medium": 0,
        "Hard": 0
    }
    
    # Count in main directory
    if os.path.exists("rgb_images"):
        for file in os.listdir("rgb_images"):
            if file.endswith("_rgb.png"):
                counts["main"] += 1
                total_count += 1
    
    # Count in subdirectories
    for subdir in ["Easy", "Medium", "Hard"]:
        dir_path = os.path.join("rgb_images", subdir)
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith("_rgb.png"):
                    counts[subdir] += 1
                    total_count += 1
    
    # Print results
    print("\nImage Count Summary:")
    print("-------------------")
    print(f"Main directory: {counts['main']} images")
    print(f"Easy directory: {counts['Easy']} images")
    print(f"Medium directory: {counts['Medium']} images")
    print(f"Hard directory: {counts['Hard']} images")
    print(f"Total images: {total_count}")
    
    return counts, total_count

def read_hdf5_file(file_path):
    """
    Read an HDF5 file and print its structure and contents.
    
    Args:
        file_path (str): Path to the HDF5 file
    """
    try:
        # Get the image ID from the annotation file path
        image_id = file_path.split('/')[-1].split('_')[0]
        
        # Try to find the image in different possible locations
        possible_image_paths = [
            f"rgb_images/{image_id}_rgb.png",
            f"rgb_images/Easy/{image_id}_rgb.png",
            f"rgb_images/Medium/{image_id}_rgb.png",
            f"rgb_images/Hard/{image_id}_rgb.png"
        ]
        
        image_path = None
        for path in possible_image_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if image_path is None:
            print(f"Warning: Could not find image for {image_id}")
            return
        
        # Open the HDF5 file in read mode
        with h5py.File(file_path, 'r') as f:
            # Print the structure of the file
            print(f"\nProcessing {image_id}:")
            print("HDF5 file structure:")
            print("-------------------")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}")
                    print(f"  Shape: {obj.shape}")
                    print(f"  Dtype: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
            
            # Walk through the file structure
            f.visititems(print_structure)
            
            # Count the number of segments
            num_segments = len([k for k in f['segments'].keys()])
            
            # Create a grid of subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # Load and display the original image in the first subplot
            try:
                img = Image.open(image_path)
                axes[0].imshow(np.array(img))
                axes[0].set_title(f'Original Image: {image_id}')
                axes[0].axis('off')
            except Exception as e:
                print(f"Error loading image: {str(e)}")
            
            # Display each segment
            for i in range(num_segments):
                if i+1 < len(axes):  # +1 because we used the first subplot for the image
                    # Read segment data
                    segment = f[f'segments/{i}'][:]
                    
                    # Display segment
                    axes[i+1].imshow(segment, cmap='gray')
                    axes[i+1].set_title(f'Segment {i}')
                    axes[i+1].axis('off')
            
            # Remove any unused subplots
            for j in range(num_segments + 1, len(axes)):
                fig.delaxes(axes[j])
            
            plt.tight_layout()
            plt.show()
                
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error reading HDF5 file: {str(e)}")

def process_all_images():
    """
    Process all images in both main directories and subdirectories.
    """
    # Process main directory
    if os.path.exists("annotations"):
        for file in os.listdir("annotations"):
            if file.endswith("_annotation.hdf5"):
                file_path = os.path.join("annotations", file)
                read_hdf5_file(file_path)
                input("Press Enter to continue to the next image...")
    
    # Process subdirectories
    subdirs = ["Easy", "Medium", "Hard"]
    for subdir in subdirs:
        annotations_dir = os.path.join("annotations", subdir)
        if os.path.exists(annotations_dir):
            for file in os.listdir(annotations_dir):
                if file.endswith("_annotation.hdf5"):
                    file_path = os.path.join(annotations_dir, file)
                    read_hdf5_file(file_path)
                    input("Press Enter to continue to the next image...")

def create_individual_segments_visualization(image_path, segments, image_id):
    """
    Create a visualization showing individual segments for an image.
    
    Args:
        image_path (str): Path to the original image
        segments (dict): Dictionary of segment arrays
        image_id (str): ID of the image
    """
    # Read the original image
    img = np.array(Image.open(image_path))
    
    # Calculate grid dimensions
    n_segments = len(segments)
    n_cols = min(3, n_segments)  # Maximum 3 columns
    n_rows = (n_segments + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    # Add title
    plt.suptitle(f'Individual Segments for Image {image_id}', fontsize=16, y=0.95)
    
    # Plot each segment
    for i, (segment_id, segment) in enumerate(segments.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        # Show original image
        plt.imshow(img)
        # Overlay segment contour
        plt.contour(segment, colors='red', linewidths=2, alpha=0.7)
        plt.title(f'Segment {segment_id}')
        plt.axis('off')
    
    # Remove any empty subplots
    for j in range(n_segments, n_rows * n_cols):
        fig.delaxes(plt.subplot(n_rows, n_cols, j + 1))
    
    plt.tight_layout()
    return fig

def save_individual_segments_to_pdf():
    """
    Process all images and save their individual segment visualizations to a PDF file.
    """
    # Create PDF file
    pdf_path = "individual_segments.pdf"
    with PdfPages(pdf_path) as pdf:
        # Process main directory
        if os.path.exists("annotations"):
            for file in os.listdir("annotations"):
                if file.endswith("_annotation.hdf5"):
                    image_id = file.split('_')[0]
                    file_path = os.path.join("annotations", file)
                    
                    # Try to find the image in different possible locations
                    possible_image_paths = [
                        f"rgb_images/{image_id}_rgb.png",
                        f"rgb_images/Easy/{image_id}_rgb.png",
                        f"rgb_images/Medium/{image_id}_rgb.png",
                        f"rgb_images/Hard/{image_id}_rgb.png"
                    ]
                    
                    image_path = None
                    for path in possible_image_paths:
                        if os.path.exists(path):
                            image_path = path
                            break
                    
                    if image_path is None:
                        print(f"Warning: Could not find image for {image_id}")
                        continue
                    
                    # Read segments from HDF5 file
                    with h5py.File(file_path, 'r') as f:
                        segments = {}
                        for key in f['segments'].keys():
                            segments[key] = f[f'segments/{key}'][:]
                        
                        # Create and save visualization
                        fig = create_individual_segments_visualization(image_path, segments, image_id)
                        pdf.savefig(fig)
                        plt.close(fig)
        
        # Process subdirectories
        subdirs = ["Easy", "Medium", "Hard"]
        for subdir in subdirs:
            annotations_dir = os.path.join("annotations", subdir)
            if os.path.exists(annotations_dir):
                for file in os.listdir(annotations_dir):
                    if file.endswith("_annotation.hdf5"):
                        image_id = file.split('_')[0]
                        file_path = os.path.join(annotations_dir, file)
                        
                        # Try to find the image in different possible locations
                        possible_image_paths = [
                            f"rgb_images/{image_id}_rgb.png",
                            f"rgb_images/Easy/{image_id}_rgb.png",
                            f"rgb_images/Medium/{image_id}_rgb.png",
                            f"rgb_images/Hard/{image_id}_rgb.png"
                        ]
                        
                        image_path = None
                        for path in possible_image_paths:
                            if os.path.exists(path):
                                image_path = path
                                break
                        
                        if image_path is None:
                            print(f"Warning: Could not find image for {image_id}")
                            continue
                        
                        # Read segments from HDF5 file
                        with h5py.File(file_path, 'r') as f:
                            segments = {}
                            for key in f['segments'].keys():
                                segments[key] = f[f'segments/{key}'][:]
                            
                            # Create and save visualization
                            fig = create_individual_segments_visualization(image_path, segments, image_id)
                            pdf.savefig(fig)
                            plt.close(fig)
    
    print(f"\nIndividual segment visualizations have been saved to {pdf_path}")

# Example usage
if __name__ == "__main__":
    # Count images first
    count_images()
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. View images with segments")
    print("2. Create PDF with contour visualizations")
    print("3. Create PDF with individual segment visualizations")
    print("4. All of the above")
    choice = input("Enter your choice (1/2/3/4): ")
    
    if choice in ['1', '4']:
        process_all_images()
    if choice in ['2', '4']:
        save_all_contours_to_pdf()
    if choice in ['3', '4']:
        save_individual_segments_to_pdf() 