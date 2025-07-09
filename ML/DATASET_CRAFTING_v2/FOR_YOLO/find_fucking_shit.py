import os

def find_bounding_boxes(annotation_folder):
    # List to store image filenames with bounding box annotations
    bounding_box_image_filenames = []

    # Loop over all annotation files in the folder
    for annotation_file in os.listdir(annotation_folder):
        if annotation_file.endswith('.txt'):
            #print(annotation_file)
            with open(os.path.join(annotation_folder, annotation_file), 'r') as f:
                lines = f.readlines()

            # Check each line in the annotation file
            for line in lines:
                # Split the line to get the elements
                elements = line.strip().split()

                # YOLOv8 object detection (bounding box) should have 5 elements: class_id, x_center, y_center, width, height
                # Polygon annotations will have more elements (for instance segmentation)
                if len(elements) == 5:
                    bounding_box_image_filenames.append(annotation_file.replace('.txt', '.jpg'))  # Assuming image is .jpg

    return bounding_box_image_filenames


# Define the path to your YOLOv8 annotations folder
annotation_folder = "/vol1/KSH/READY/lebedka/lebedka_13.01_116/labels/val"

# Find all images with bounding box annotations
bounding_box_image_filenames = find_bounding_boxes(annotation_folder)
# '002768_106 PROC 18.11.jpg', '017062_PC PROC 18.11.jpg', '016839_PC PROC 18.11.jpg', '010932_PC PROC 18.11.jpg', '018959_106 PROC 18.11.jpg', '018074_106 PROC 18.11.jpg' val '008139_106 PROC 16.11.jpg'
print("Found bounding boxes in the following images:", bounding_box_image_filenames)