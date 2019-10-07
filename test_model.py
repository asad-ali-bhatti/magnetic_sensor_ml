import tensorflow
from tensorflow import keras
import pandas
import psycopg2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np

data = keras.datasets.fashion_mnist
(ti, tl), (test_i, test_l) = data.load_data()

# print(data)
# print('------------------00000000------------------')
# print(ti[0])
# print(tl[0])
db_client = psycopg2.connect(host="localhost", database="magnetic_sensor_readings_dev")

query = '''
  SELECT device_name, reading, device_id, row FROM(
    SELECT d.name as device_name, r.magnitude as reading, d.id as device_id, row_number() OVER (PARTITION BY d.id ) as row 
    FROM readings r INNER JOIN devices d ON r.device_id = d.id
    WHERE r.reading_type != 'predict'
    ORDER BY r.created_at DESC
  ) as readings_per_device 
 '''

devices_query = '''SELECT name, id from devices'''
pd_devices = pandas.read_sql_query(devices_query, db_client)
pd_query = pandas.read_sql_query(query, db_client)
data_frame = pandas.DataFrame(pd_query, columns=['device_name', 'reading', 'device_id'])
train, test = train_test_split(data_frame, test_size=0.2)
devices_data_frame = pandas.DataFrame(pd_devices, columns=['name', 'id'])
no_of_devices = devices_data_frame['name'].size
print(pd_query)

train_devices = train['device_id'] - 1
train_readings = train['reading']
test_devices = test['device_id'] - 1
test_readings = test['reading']
# print(make_sublist_of_data(data_frame['device_name']))
print(train['reading'].shape)
print(train['device_id'].shape)
model = keras.Sequential([
    keras.layers.Dense(500, input_shape=(1,)),
    keras.layers.Dense(50, name='hiddenLayer-1', activation='relu'),
    keras.layers.Dense(50, name='hiddenLayer-2', activation='relu'),
    keras.layers.Dense(50, name='hiddenLayer-3', activation='relu'),
    keras.layers.Dense(no_of_devices, name='outputLayer', activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_readings, train_devices, validation_data=[test_readings, test_devices], epochs=5,
                    batch_size=16, verbose=1)
keras.utils.plot_model(model, show_shapes=True, to_file='graph.png')
print(model.summary())
# =============TRAINING HISTORY==============
# Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
#
# plt.title('Model accuracy')
# plt.ylabel('Accuracy/Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Accuracy', 'Validation Accuracy', 'Loss', 'Validation Loss'], loc='upper right')
# plt.show()
#
# input("Press Enter to continue...")
# =============END TRAINING HISTORY==============
# devices = test['device_id']-1
# readings = test['reading']
# loss, acc = model.evaluate(readings, devices)
# print('Accuracy: ', acc)
# #
predict_query = '''
    SELECT d.name as device_name, r.magnitude as reading, d.id as device_id
    FROM readings r
    INNER JOIN devices d ON r.device_id = d.id
    WHERE r.reading_type = 'predict'
    ORDER BY r.created_at DESC
    LIMIT 10
 '''

pd_query = pandas.read_sql_query(predict_query, db_client)
data_frame = pandas.DataFrame(pd_query, columns=['device_name', 'reading', 'device_id'])
devices = data_frame['device_id'] - 1
readings = data_frame['reading']
prediction = model.predict(readings)
print('prediction: ', prediction)
predicted_device = devices_data_frame['name'][np.argmax(prediction[0])]
print(devices_data_frame)
print('Predicted Device: ', predicted_device)


def select_winner(predictions):
    candidates = {}
    for pred in predictions:
        print('predict: ', pred)
        max_index = np.argmax(pred)
        print('max', max_index)
        if candidates.get(max_index) and candidates.get(max_index) >= 1:
            candidates[max_index] = candidates.get(max_index) + 1
        else:
            candidates[max_index] = 1

    print(candidates)

    return max(candidates, key=candidates.get)


select_winner(prediction)
