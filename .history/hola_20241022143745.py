import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Cargar los datos
df = pd.read_csv('bank_data.csv', sep=';')

# convertir
cat_int_feats = ['default', 'housing', 'loan', 'contact', 'education', 'poutcome', 'marital', 'job', 'month']
cat_str_feats = []  
num_feats = ['age', 'balance', 'campaign', 'pdays', 'previous']


for feat in cat_int_feats:
    df[feat] = df[feat].astype('category').cat.codes  

# Convertir 
df['y'] = df['y'].map({'no': 0, 'yes': 1})  


feats_ordered = cat_int_feats + cat_str_feats + num_feats + ['y']
df = df[feats_ordered]


train = df.sample(frac=0.8, random_state=100)
test = df.drop(train.index)
val = train.sample(frac=0.2, random_state=100)
train = train.drop(val.index)

print(train.shape)
print(val.shape)
print(test.shape)


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("y") 
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train)
val_ds = dataframe_to_dataset(val)
test_ds = dataframe_to_dataset(test)


type(train_ds)


for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)


batch_size = 32
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

def encode_numerical_feature(feature, name, dataset):
    normalizer = keras.layers.Normalization()
    feature_ds = dataset.map(lambda x, y: x[name])  
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))  
    normalizer.adapt(feature_ds)  
    encoded_feature = normalizer(feature)  
    return encoded_feature


def encode_categorical_feature(feature, name, dataset):
    lookup = keras.layers.IntegerLookup(output_mode="binary")  
    feature_ds = dataset.map(lambda x, y: x[name])  
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))  
    lookup.adapt(feature_ds)  
    encoded_feature = lookup(feature)  
    return encoded_feature


inputs = []
for i in cat_int_feats:
    inputs.append(keras.Input(shape=(1,), name=i, dtype="int64"))
for i in cat_str_feats:
    inputs.append(keras.Input(shape=(1,), name=i, dtype="string"))
for i in num_feats:
    inputs.append(keras.Input(shape=(1,), name=i))


feats_encoded = []
for i, feat in enumerate(cat_int_feats):
    feats_encoded.append(
        encode_categorical_feature(inputs[i], feat, train_ds)
    )

len_feats = len(feats_encoded)
for i, feat in enumerate(cat_str_feats):
    feats_encoded.append(
        encode_categorical_feature(inputs[len_feats + i], feat, train_ds)
    )

len_feats = len(feats_encoded)
for i, feat in enumerate(num_feats):
    feats_encoded.append(
        encode_numerical_feature(inputs[len_feats + i], feat, train_ds)
    )


all_feats = tf.concat(feats_encoded, axis=1)
model_layers = keras.layers.Dense(64, activation='relu')(all_feats)
model_layers = keras.layers.Dense(1, activation='sigmoid')(model_layers)

# MODELO
model = keras.Model(inputs, model_layers)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


model.fit(train_ds, epochs=50, validation_data=val_ds)

#_________________________________________________

test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

y_true = []
y_pred = []

for x, y in test_ds:
    y_true.extend(y.numpy())
    predictions = model.predict(x)
    y_pred.extend((predictions > 0.5).astype(int).flatten())  

# MATRIZ DE CONFUCIÓN
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.show()

print(report)
