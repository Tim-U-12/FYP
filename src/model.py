from dataset import get_generators
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from utils import singleTestModel

if __name__ == "__main__":
    # Input shape should match your image size, e.g., 224x224x3
    input_tensor = Input(shape=(224, 224, 3))

    # Load ResNet50 with pretrained ImageNet weights, excluding top
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Freeze base model layers (optional, for initial training)
    for layer in base_model.layers:
        layer.trainable = False

    # Add your own classification head
    x = GlobalAveragePooling2D()(base_model.output)

    # Example: 3 separate outputs
    age_output = Dense(10, activation='softmax', name='age_output')(x)     # e.g., 10 age bins
    gender_output = Dense(2, activation='softmax', name='gender_output')(x)  # Male/Female
    race_output = Dense(5, activation='softmax', name='race_output')(x)     # 5 races?

    # Build model
    model = Model(inputs=input_tensor, outputs=[age_output, gender_output, race_output])
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={
            'age_output': 'categorical_crossentropy',
            'gender_output': 'categorical_crossentropy',
            'race_output': 'categorical_crossentropy'
        },
        metrics={
            'age_output': 'accuracy',
            'gender_output': 'accuracy',
            'race_output': 'accuracy'
        }
    )