import matplotlib.pyplot as plt

def graficar_entrenamiento(history, title):
    plt.plot(history['train_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    plt.show()

def graficar_metricas(metricas, resultados):
    plt.title('Precision and Recall for different threshold values')
    plt.plot(
        metricas["th"],
        metricas["precision"][1:],
        'b',
        label='Threshold-Precision curve',
        color='r'
    )
    plt.plot(
        metricas["th"],
        metricas["recall"][1:],
        'b',
        label='Threshold-Recall curve',
        color = 'b')
    plt.show()

    plt.figure(figsize=(15,8))
    plt.hist(resultados[resultados.true_class == 0].pred_loss.values, alpha=0.75, label='Humanos')
    plt.hist(resultados[resultados.true_class == 1].pred_loss.values, alpha=0.75, label='Bots')
    plt.xlabel('Pérdidas (MAE - Error Cuadratico Medio)')
    plt.ylabel('Nro. ejemplos')
    plt.axvline(metricas["thresholdHumanos"], 0, 500, c='blue', label='85% Humanos')
    plt.axvline(metricas["thresholdBots"], 0, 500, c='red', label='25% Bots')
    plt.axvline(metricas["threshold"], 0, 500, c='purple', label='Limite Promedio')
    plt.axvline(metricas["limite_optimo"], 0, 500, c='cyan', label='Limite Optimo')
    plt.axvline(metricas["limite_equilibrado"], 0, 500, c='g', label='Limite Equilibrado')
    plt.legend(loc='upper right')
    plt.show()

