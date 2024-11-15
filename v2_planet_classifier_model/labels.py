from main import ImageDataGenerator

# Set up the ImageDataGenerator to point to the training data directory
IDG = ImageDataGenerator(rescale=1/255)
train_path = '/Users/gregchu/Downloads/planets_data/training'

# Initialize the generator to retrieve the labels
train_gen = IDG.flow_from_directory(
    directory=train_path, 
    target_size=(224, 224), 
    class_mode='categorical', 
    batch_size=10
)

# Get the class-to-index mapping and create the index-to-class mapping
class_indices = train_gen.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

# Print the mapping to confirm
print("Class indices to class names mapping:")
for idx, class_name in index_to_class.items():
    print(f"Class index {idx}: Class name '{class_name}'")