from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from utils import UTKLabelType

def buildModel(labelType: UTKLabelType):
    input_tensor = Input(shape=(224, 224, 3))
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Optionally freeze base model for transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)

    if labelType == UTKLabelType.AGE:
        output = Dense(10, activation='softmax', name='data_output')(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    elif labelType == UTKLabelType.GENDER:
        output = Dense(2, activation='softmax', name='data_output')(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    elif labelType == UTKLabelType.RACE:
        output = Dense(5, activation='softmax', name='data_output')(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        raise ValueError("Invalid label type")

    model = Model(inputs=input_tensor, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=loss,
        metrics=metrics
    )

    return model