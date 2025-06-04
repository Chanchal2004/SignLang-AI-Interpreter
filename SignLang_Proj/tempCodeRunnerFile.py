# Step 1 - Building the CNN
# classifier = Sequential()

# # First convolution layer and pooling
# classifier.add(Conv2D(64, (3, 3), input_shape=(sz, sz, 1), activation='relu', padding='same'))
# classifier.add(BatchNormalization())  # Normalize activations
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.25))  # Add dropout to reduce overfitting

# # Second convolution layer and pooling
# classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# classifier.add(BatchNormalization())
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.25))

# # Third convolution layer and pooling
# classifier.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# classifier.add(BatchNormalization())
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.25))

# # Flattening the layers
# classifier.add(Flatten())

# # Fully connected layers
# classifier.add(Dense(512, activation='relu'))
# classifier.add(BatchNormalization())
# classifier.add(Dropout(0.5))  # Higher dropout for fully connected layers
# classifier.add(Dense(256, activation='relu'))
# classifier.add(BatchNormalization())
# classifier.add(Dropout(0.5))
# classifier.add(Dense(26, activation='softmax'))  # Output layer for 26 classes

# # Compiling the CNN
# optimizer = Adam(learning_rate=0.001)  # Adjust learning rate
# classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # Step 2 - Preparing the train/test data and training the model
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2,
#     rotation_range=10,  # Add rotation for better generalization
#     width_shift_range=0.1,  # Add horizontal shift
#     height_shift_range=0.1,  # Add vertical shift
# )

# training_set = train_datagen.flow_from_directory(
#     'AtoZ_3.1',
#     target_size=(sz, sz),
#     batch_size=32,  # Increased batch size for better generalization
#     color_mode='grayscale',
#     subset="training"
# )

# test_set = train_datagen.flow_from_directory(
#     'AtoZ_3.1',
#     target_size=(sz, sz),
#     batch_size=32,
#     color_mode='grayscale',
#     subset="validation"
# )

# # Callbacks for better training
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
# model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# # Training the model
# history = classifier.fit(
#     training_set,
#     steps_per_epoch=training_set.samples // training_set.batch_size,
#     epochs=100,  # Reduced epochs due to early stopping
#     validation_data=test_set,
#     validation_steps=test_set.samples // test_set.batch_size,
#     callbacks=[reduce_lr, model_checkpoint]
# )

# # Saving the model
# model_json = classifier.to_json()
# with open("model-bw.json", "w") as json_file:
#     json_file.write(model_json)
# print('Model Saved')

# classifier.save_weights('model-bw.weights.h5')
# print('Weights saved')