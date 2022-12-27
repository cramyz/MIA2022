import pandas
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch

def get_dataset(path):
    df_recuento_datos = pandas.read_csv(path)
    #df_recuento_datos = df_recuento_datos.drop(columns= {'Unnamed: 0'})   # type: ignore
    return df_recuento_datos



def normalizar_dataset(df_recuento_datos: pandas.DataFrame):
    complete_df = df_recuento_datos.copy()
    columnas = complete_df.columns.to_list()

    for columna in columnas:
        if columna != "type" and columna!= 'Player':
            complete_df[columna] = MinMaxScaler(feature_range=(0, 1)).fit_transform(complete_df[columna].values.reshape(-1, 1),)
    
    complete_df["type_normalized"] = complete_df.apply(lambda row: 0 if row["type"] == "human" else 1, axis=1)

    return complete_df

def convertir_datos_a_formato_red(X_train, X_test, device):
    y_train = X_train['type_normalized']
    X_train = X_train.drop(['type_normalized'], axis=1)

    y_test = X_test['type_normalized']
    X_test = X_test.drop(['type_normalized'], axis=1)

    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    X_train = torch.from_numpy(X_train).to(device)
    X_test = torch.from_numpy(X_test).to(device)

    return  X_train, y_train, X_test, y_test