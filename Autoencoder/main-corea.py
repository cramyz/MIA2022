import gc

import statistics
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
import mlflow

from data.dataService import convertir_datos_a_formato_red, get_dataset, normalizar_dataset
from model.autoencoder import generar_modelo 
from model.helper import calcular_metricas, entrenar_modelo, evaluar_modelo, generar_predicciones



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
PATH_DATOS = r"dataset\DatosCorea\consolidado-corea-k6.csv"
#PATH_DATOS = r"dataset\DatosCorea\consolidado.csv"


def generar_hiperparametros():
    hiper_parametros = []

    epochs = [100]
    minibatch_size = [128]
    learning_rate = [1e-3]
    weight_decay = [0]

    for e in epochs:
        for mbz in minibatch_size:
            for lr in learning_rate:
                for wd in weight_decay:
                    hiper_parametros.append(
                        {
                            "num_epochs" : e,
                            "minibatch_size" : mbz,
                            "learning_rate" : lr,
                            "weight_decay" : wd,
                        }
                    )

    return hiper_parametros


def experimento_kfold(X_train, X_test, hiper_parametros):
    splits_numbers = min(10, max(2, len(X_train)//len(X_test)))

    metricas_acumuladas = []

    kf = KFold(n_splits=splits_numbers, shuffle=True)


    for fold,(train_idx,test_idx) in enumerate(kf.split(X_train)):
        text = f"{fold}-fold"
        print(f"{text:*^30}")

        k_X_train = X_train.iloc[train_idx,:]
        k_X_test = X_train.iloc[test_idx,:]

        k_X_test = pd.concat([X_test, k_X_test])


        k_X_train, k_y_train, k_X_test, k_y_test = convertir_datos_a_formato_red(k_X_train, k_X_test, device)

        k_train_loader = DataLoader(k_X_train, batch_size = hiper_parametros["minibatch_size"])
        k_test_loader = DataLoader(k_X_test, batch_size=1, shuffle=False)


        model, criterion, optimizer = generar_modelo(k_X_test.shape[1], hiper_parametros, device)

        #print(model)
        

        model, history = entrenar_modelo(model, criterion, optimizer, hiper_parametros["num_epochs"], k_train_loader, device=device, is_debug=True, name_experiment = f"k-{fold}_mloss")

        #graficar_entrenamiento(history, f"{text} - mean loss")

        k_resultados = evaluar_modelo(model, criterion, k_test_loader, k_y_test)

        k_metricas = calcular_metricas(k_resultados)

        #graficar_metricas(k_metricas, k_resultados)


        k_metricas_optimo_clasificacion =  generar_predicciones(k_metricas["limite_optimo"], k_resultados)
        k_metricas_equilibrado_clasificacion =  generar_predicciones(k_metricas["limite_equilibrado"], k_resultados)

        metricas_acumuladas.append({
            # "model": model,
            # "criterion": criterion, 
            # "optimizer": optimizer,
            # "history": history,
            "k_resultados": k_resultados,
            "k_metricas_entrenamiento": k_metricas,
            "k_metricas_optimo_evaluacion": k_metricas_optimo_clasificacion,
            "k_metricas_equilibrado_evaluacion": k_metricas_equilibrado_clasificacion,
            "k_limite_optimo_accuracy": k_metricas_optimo_clasificacion["classification_report"]["accuracy"],
            "k_limite_equilibrado_accuracy": k_metricas_equilibrado_clasificacion["classification_report"]["accuracy"],
            "k_promedio_recall_optimo":k_metricas_optimo_clasificacion["classification_report"]["macro avg"]["recall"],
            "k_promedio_precision_optimo":k_metricas_optimo_clasificacion["classification_report"]["macro avg"]["precision"],
            "k_promedio_f1score_optimo":k_metricas_optimo_clasificacion["classification_report"]["macro avg"]["f1-score"],
            "k_promedio_recall_equilibrado":k_metricas_equilibrado_clasificacion["classification_report"]["macro avg"]["recall"],
            "k_promedio_precision_equilibrado":k_metricas_equilibrado_clasificacion["classification_report"]["macro avg"]["precision"],
            "k_promedio_f1score_equilibrado":k_metricas_equilibrado_clasificacion["classification_report"]["macro avg"]["f1-score"]
        })

        mlflow.log_metric(f"k-limite_optimo", k_metricas["limite_optimo"], fold)
        mlflow.log_metric(f"k-limite_equilibrado", k_metricas["limite_equilibrado"], fold)
        mlflow.log_metric(f"k-thresholdHumanos", k_metricas["thresholdHumanos"], fold)
        mlflow.log_metric(f"k-thresholdBots", k_metricas["thresholdBots"], fold)
        mlflow.log_metric(f"k-threshold", k_metricas["threshold"], fold)
        mlflow.log_metric("k_promedio_recall_optimo", k_metricas_optimo_clasificacion["classification_report"]["macro avg"]["recall"], fold)
        mlflow.log_metric("k_promedio_precision_optimo", k_metricas_optimo_clasificacion["classification_report"]["macro avg"]["precision"], fold)
        mlflow.log_metric("k_promedio_f1score_optimo", k_metricas_optimo_clasificacion["classification_report"]["macro avg"]["f1-score"], fold)
        mlflow.log_metric("k_promedio_recall_equilibrado", k_metricas_equilibrado_clasificacion["classification_report"]["macro avg"]["recall"], fold)
        mlflow.log_metric("k_promedio_precision_equilibrado", k_metricas_equilibrado_clasificacion["classification_report"]["macro avg"]["precision"], fold)
        mlflow.log_metric("k_promedio_f1score_equilibrado", k_metricas_equilibrado_clasificacion["classification_report"]["macro avg"]["f1-score"], fold)


    

    mean_accuracy_optimo = statistics.mean([d['k_limite_optimo_accuracy'] for d in metricas_acumuladas])
    mean_accuracy_equilibrado = statistics.mean([d['k_limite_equilibrado_accuracy'] for d in metricas_acumuladas])
    mean_promedio_recall_optimo = statistics.mean([d['k_promedio_recall_optimo'] for d in metricas_acumuladas])
    mean_promedio_precision_optimo = statistics.mean([d['k_promedio_precision_optimo'] for d in metricas_acumuladas])
    mean_promedio_f1score_optimo = statistics.mean([d['k_promedio_f1score_optimo'] for d in metricas_acumuladas])
    mean_promedio_recall_equilibrado = statistics.mean([d['k_promedio_recall_equilibrado'] for d in metricas_acumuladas])
    mean_promedio_precision_equilibrado = statistics.mean([d['k_promedio_precision_equilibrado'] for d in metricas_acumuladas])
    mean_promedio_f1score_equilibrado = statistics.mean([d['k_promedio_f1score_equilibrado'] for d in metricas_acumuladas])

    stdev_accuracy_optimo = statistics.stdev([d['k_limite_optimo_accuracy'] for d in metricas_acumuladas])
    stdev_accuracy_equilibrado = statistics.stdev([d['k_limite_equilibrado_accuracy'] for d in metricas_acumuladas])
    stdev_promedio_recall_optimo = statistics.stdev([d['k_promedio_recall_optimo'] for d in metricas_acumuladas])
    stdev_promedio_precision_optimo = statistics.stdev([d['k_promedio_precision_optimo'] for d in metricas_acumuladas])
    stdev_promedio_f1score_optimo = statistics.stdev([d['k_promedio_f1score_optimo'] for d in metricas_acumuladas])
    stdev_promedio_recall_equilibrado = statistics.stdev([d['k_promedio_recall_equilibrado'] for d in metricas_acumuladas])
    stdev_promedio_precision_equilibrado = statistics.stdev([d['k_promedio_precision_equilibrado'] for d in metricas_acumuladas])
    stdev_promedio_f1score_equilibrado = statistics.stdev([d['k_promedio_f1score_equilibrado'] for d in metricas_acumuladas])



    mlflow.log_metric("mean_accuracy_limite_optimo", mean_accuracy_optimo)
    mlflow.log_metric("mean_accuracy_limite_equilibrado", mean_accuracy_equilibrado)
    mlflow.log_metric("mean_promedio_recall_optimo", mean_promedio_recall_optimo)
    mlflow.log_metric("mean_promedio_precision_optimo", mean_promedio_precision_optimo)
    mlflow.log_metric("mean_promedio_f1score_optimo", mean_promedio_f1score_optimo)
    mlflow.log_metric("mean_promedio_recall_equilibrado", mean_promedio_recall_equilibrado)
    mlflow.log_metric("mean_promedio_precision_equilibrado", mean_promedio_precision_equilibrado)
    mlflow.log_metric("mean_promedio_f1score_equilibrado", mean_promedio_f1score_equilibrado)

    mlflow.log_metric("stdev_accuracy_limite_optimo", stdev_accuracy_optimo)
    mlflow.log_metric("stdev_accuracy_limite_equilibrado", stdev_accuracy_equilibrado)
    mlflow.log_metric("stdev_promedio_recall_optimo", stdev_promedio_recall_optimo)
    mlflow.log_metric("stdev_promedio_precision_optimo", stdev_promedio_precision_optimo)
    mlflow.log_metric("stdev_promedio_f1score_optimo", stdev_promedio_f1score_optimo)
    mlflow.log_metric("stdev_promedio_recall_equilibrado", stdev_promedio_recall_equilibrado)
    mlflow.log_metric("stdev_promedio_precision_equilibrado", stdev_promedio_precision_equilibrado)
    mlflow.log_metric("stdev_promedio_f1score_equilibrado", stdev_promedio_f1score_equilibrado)

    return metricas_acumuladas

def experinto(dataframe, hiper_parametros):
    X_train = dataframe[dataframe.type_normalized == 0]
    X_test = dataframe[dataframe.type_normalized == 1]

    metricas = experimento_kfold(X_train, X_test, hiper_parametros)


def main():

    # Si no existe, creo el experimento
    experiment_name = "autoencoder-datos-corea-k6"

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name) 

    experiment = mlflow.get_experiment_by_name(experiment_name)

    df_normalizado = get_dataset(PATH_DATOS)
    #df_normalizado = normalizar_dataset(get_dataset(PATH_DATOS))
    #df_normalizado = df_normalizado.drop(columns=["Player", "type"])
    #df_normalizado = df_normalizado.drop(columns=["Type"])

    print(df_normalizado.head())

    hiper_parametros = generar_hiperparametros()
    for _i in range(50):
        for ith, hp in enumerate(hiper_parametros):
            print(f"Hiperparametros: {ith+1}/{len(hiper_parametros)}")
            print(hp)
            with mlflow.start_run(experiment_id = experiment.experiment_id):
                mlflow.log_params(hp)
                experinto(df_normalizado[:], hp)
                gc.collect()

    

if __name__ == "__main__":
    print('Using device:', device)

    main()
