import numpy
import pandas
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import torch
from torch.nn import Module
import mlflow

def entrenar_modelo(model: Module, criterion, optimizer, num_epochs, train_loader, device=torch.device('cpu'), is_debug=False, name_experiment=None):
    history = {}
    history['train_loss'] = []
    history['test_loss'] = []

    for epoch in range(num_epochs):
        h = numpy.array([])
        for data in train_loader:
            # ===================forward=====================
            #data = data.to(device)
            #model.to(device)
            output = model(data)
            loss = criterion(output, data)
            h = numpy.append(h, loss.item())
            
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        
        mean_loss = numpy.mean(h)
        if is_debug and (epoch % 50 == 0 or epoch+1 == num_epochs):
            print(f'epoch [{epoch + 1}/{num_epochs}], loss:{mean_loss}')
        
        if name_experiment:
            mlflow.log_metric(name_experiment, mean_loss, epoch)
        
        history['train_loss'].append(mean_loss)

    return model, history


def evaluar_modelo(model, criterion, test_loader, y_test, device='cpu'):
    pred_losses = {'pred_loss' : []}
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs = data.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs).data.item()
            pred_losses['pred_loss'].append(loss)
            
    reconstructionErrorDF = pandas.DataFrame(pred_losses)
    reconstructionErrorDF['true_class'] = y_test

    return reconstructionErrorDF

def calcular_metricas(reconstructionErrorDF):
    precision, recall, th = precision_recall_curve(reconstructionErrorDF.true_class, reconstructionErrorDF.pred_loss)

    limite_equilibrado = 0
    diferencia_minima = 100
    for iterador, limite in enumerate(th):
        pre = precision[iterador+1]
        rec = recall[iterador+1]

        diferencia = abs(pre-rec)

        if diferencia < diferencia_minima:
            diferencia_minima = diferencia
            limite_equilibrado = limite


    thresholdHumanos = reconstructionErrorDF[reconstructionErrorDF.true_class == 0].quantile(.85).pred_loss
    thresholdBots = reconstructionErrorDF[reconstructionErrorDF.true_class == 1].quantile(.15).pred_loss
    threshold = (thresholdHumanos + thresholdBots)/2

    metricas = {
        "precision":precision,
        "recall":recall,
        "th":th,
        "limite_optimo":(limite_equilibrado + threshold)/2,
        "limite_equilibrado": limite_equilibrado,
        "diferencia_minima":diferencia_minima,
        "thresholdHumanos":thresholdHumanos,
        "thresholdBots":thresholdBots,
        "threshold" : threshold
    }

    return metricas

def generar_predicciones(threshold, reconstructionErrorDF):
    LABELS = ["Human", "Bot"]
    
    y_pred = [1 if e > threshold else 0 for e in reconstructionErrorDF.pred_loss.values]
    reconstructionErrorDF["prediccion"] = y_pred


    tn_n, fp_n, fn_n, tp_n = confusion_matrix(y_true = reconstructionErrorDF.true_class,
                                y_pred = y_pred, 
                                normalize="true").ravel()

    tn, fp, fn, tp = confusion_matrix(y_true = reconstructionErrorDF.true_class,
                                y_pred = y_pred).ravel()

    c_r = classification_report(reconstructionErrorDF.true_class, y_pred, target_names=LABELS, output_dict = True)

    respuesta = {
        "classification_report": c_r,
        "confusion_matrix_normaliazed": {
            "tn_n": tn_n,
            "fp_n": fp_n,
            "fn_n": fn_n,
            "tp_n": tp_n
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    }

    return respuesta