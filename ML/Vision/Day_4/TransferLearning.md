# Transfer Learning
Reuse pretrained models

Load pretrained model > Replace final layers > Train Network > Predict and Assess Network Accuracy > Deploy Result  
    Replace only task specific layers. We keep generic layers.
    Minimal changes - Change Fully Connected Layers.
                Last Layer Only (Not fully optimized); Few last layers; Train complete layers (No transfer learning)


conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=)
model = Sequential()
model.add(conv_base)
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
conv_base.trainable = False

model.summary()

# Model Interpretability/Explainability
LIME and SHAP
https://christophm.github.io/interpretable-ml-book

Deep Learning with Python by Francois Chollet
ai.googleblog.com

    Feature Importance - You can get average weight from models. (global)
        But for few rows (local), some different features may be important.. That's why LIME and SHAP.
