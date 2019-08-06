import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from plot_utils import visualize_reconstruction_error
from recurrent import LstmAutoEncoder,BidirectionalLstmAutoEncoder
import datetime

def main():
    data_dir_path = './data'
    model_dir_path = './model'
    #ecg_data = pd.read_csv(data_dir_path + '/ecg_discord_test.csv', header=None)
    data2 = pd.read_csv('~//Trane/bad/00C0A6F4.csv')
    data2.dropna(inplace=True)
    data2.drop('Tsph', axis='columns', inplace=True)
    data2.drop('RHi', axis='columns', inplace=True)
    data2.drop('RHo', axis='columns', inplace=True)
    data2.drop('Hi', axis='columns', inplace=True)
    data2.drop('Ho', axis='columns', inplace=True)
    data2.drop('DToi', axis='columns', inplace=True)
    data2.drop('DAHoi', axis='columns', inplace=True)
    data2.iloc[:, 0] = pd.to_datetime(data2.iloc[:, 0])
    data2.iloc[:, 0] = data2.iloc[:,0].diff().dt.total_seconds()
    data2.dropna(inplace=True)

    #print(ecg_data.head())
    #ecg_np_data = ecg_data.as_matrix()
    data2 = data2.as_matrix()
    scaler = MinMaxScaler()
    #ecg_np_data = scaler.fit_transform(ecg_np_data)
    ecg_np_data = scaler.fit_transform(data2)
    print(ecg_np_data.shape)

    ae = BidirectionalLstmAutoEncoder()

    # fit the data and save model into model_dir_path
    ae.fit(ecg_np_data[:, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.99)

    # load back the model saved in model_dir_path detect anomaly
    #ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(ecg_np_data[:, :])
    reconstruction_error = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
        reconstruction_error.append(dist)

    visualize_reconstruction_error(reconstruction_error, ae.threshold)


if __name__ == '__main__':
    main()