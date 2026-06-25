import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

DEFAULT_L2 = 1e-4

def build_baseline_cnn(
    input_shape: tuple = (25, 25, 3),
    num_classes: int = 16,
    dense_units: int = 256,
    dropout_fc1: float = 0.35,
    dropout_fc2: float = 0.30,
    l2_reg: float = DEFAULT_L2,
) -> Model:
    """
    Build a standard Baseline CNN for HSI classification.
    
    NOTE: This is a placeholder model for the public showcase repository. 
    The novel architecture (Multi-Scale CNN with spatial attention and 
    specialized feature extractors) has been replaced with this standard 
    CNN to protect intellectual property.
    """
    inputs = Input(shape=input_shape)
    
    # Standard CNN architecture
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(dense_units, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_fc1)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs, name="baseline_cnn")

def build_baseline_cnn_from_config(config: dict) -> Model:
    """Build model using configuration dictionary."""
    model_config = config.get("model", {})
    # Note: num_classes is defined under 'dataset' in the yaml config
    num_classes = config.get("dataset", {}).get("num_classes", 21)

    return build_baseline_cnn(
        input_shape=tuple(model_config.get("input_shape", (25, 25, 3))),
        num_classes=num_classes,
        dense_units=model_config.get("dense_units", 256),
        dropout_fc1=model_config.get("dropout_fc1", 0.35),
        dropout_fc2=model_config.get("dropout_fc2", 0.30),
        l2_reg=model_config.get("l2_reg", DEFAULT_L2)
    )
