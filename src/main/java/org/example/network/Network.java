package org.example.network;

import fi.iki.elonen.NanoHTTPD;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.example.activation.ActivationFunction;
import org.example.activation.IActivationFunction;
import org.example.activation.iml.LeakyReLU;
import org.example.activation.iml.Sigmoid;
import org.example.activation.iml.Swish;
import org.example.activation.iml.TanH;
import org.example.data.MLData;
import org.example.data.MLDataSet;
import org.example.model.Neuron;
import org.example.server.MultiLayerNetworkView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/*
 * Network: Это основной класс, который представляет нейронную сеть.
 * Он содержит слои нейронов (входной, скрытый и выходной), а также параметры обучения, такие как скорость обучения и момент.
 * Neuron: Это класс, который представляет отдельный нейрон в сети. Каждый нейрон имеет свои веса и смещение, а также функцию активации.
 * IActivationFunction: Это интерфейс для функций активации, которые используются нейронами.
 * В коде используются различные функции активации, такие как LeakyReLU, TanH, Sigmoid и Swish.
 * init(): Этот метод инициализирует нейронную сеть, создавая нейроны для каждого слоя.
 * Train(MLDataSet set, int epoch): Этот метод обучает нейронную сеть на основе предоставленного набора данных.
 * Он выполняет прямое и обратное распространение ошибки для каждого эпоха обучения.
 * forward(double[] inputs) и backward(double[] targets): Эти методы выполняют прямое и обратное распространение ошибки в нейронной сети.
 * predict(double… inputs): Этот метод используется для получения прогнозов от нейронной сети на основе входных данных.
 * */
public class Network {

    private static final Logger logger = LogManager.getLogger(Network.class);
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    private final List<Neuron> inputLayer;
    private final List<Neuron> hiddenLayer;
    private final List<Neuron> outputLayer;
    private double learningRate = 0.01;
    private double momentum = 0.5;
    private IActivationFunction iActivationFunction;

    public Network(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.inputLayer = new ArrayList<>();
        this.hiddenLayer = new ArrayList<>();
        this.outputLayer = new ArrayList<>();
    }

    //setLearningRate(double learningRate): Этот метод позволяет установить скорость обучения для нейронной сети.
    // Скорость обучения - это параметр, который определяет, насколько быстро модель обновляет веса в процессе обучения.
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    //setMomentum(double momentum): Этот метод позволяет установить момент для нейронной сети.
    // Момент - это параметр, который помогает ускорить градиентный спуск в правильном направлении и уменьшает колебания.
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    //setActivationFunction(ActivationFunction activationFunction):
    // Этот метод позволяет установить функцию активации для нейронов в нейронной сети.
    // Функция активации определяет выходное значение нейрона на основе его входных значений.
    // В этом коде используется перечисление ActivationFunction, которое содержит различные типы функций активации, такие как LEAKY_RELU, TANH, SIGMOID и SWISH.
    // В зависимости от выбранного значения ActivationFunction, создается соответствующий объект функции активации и присваивается переменной iActivationFunction.
    //
    //Вот подробности:
    //Если выбрано LEAKY_RELU, создается новый объект LeakyReLU и присваивается iActivationFunction.
    //Если выбрано TANH, создается новый объект TanH и присваивается iActivationFunction.
    //Если выбрано SIGMOID, создается новый объект Sigmoid и присваивается iActivationFunction.
    //Если выбрано SWISH, создается новый объект Swish и присваивается iActivationFunction.
    //Таким образом, этот метод позволяет динамически изменять функцию активации нейронов в нейронной сети.
    // Это может быть полезно при настройке и оптимизации нейронной сети.
    public void setActivationFunction(ActivationFunction activationFunction) {
        switch (activationFunction) {
            case LEAKY_RELU -> {
                this.iActivationFunction = new LeakyReLU();
                break;
            }
            case TANH -> {
                this.iActivationFunction = new TanH();
                break;
            }
            case SIGMOID -> {
                this.iActivationFunction = new Sigmoid();
                break;
            }
            case SWISH -> {
                this.iActivationFunction = new Swish();
                break;
            }
        }
    }

    //init(): Этот метод инициализирует нейронную сеть, создавая нейроны для каждого слоя.
    // Создание нейронов для входного слоя:
    // Для каждого входного нейрона создается новый объект Neuron и добавляется в список inputLayer.
    // Количество входных нейронов равно inputSize.
    // Создание нейронов для скрытого слоя:
    // Для каждого скрытого нейрона создается новый объект Neuron,
    // который принимает входной слой и функцию активации в качестве параметров, и добавляется в список hiddenLayer.
    // Количество скрытых нейронов равно hiddenSize.
    // Создание нейронов для выходного слоя:
    // Для каждого выходного нейрона создается новый объект Neuron,
    // который принимает скрытый слой и функцию активации в качестве параметров, и добавляется в список outputLayer.
    // Количество выходных нейронов равно outputSize.
    private void init() {
        for (int i = 0; i < inputSize; i++) {
            this.inputLayer.add(new Neuron());
        }
        for (int i = 0; i < hiddenSize; i++) {
            this.hiddenLayer.add(new Neuron(this.inputLayer, iActivationFunction));
        }
        for (int i = 0; i < outputSize; i++) {
            this.outputLayer.add(new Neuron(this.hiddenLayer, iActivationFunction));
        }
        logger.info("Network initialization.");
    }

    //train(MLDataSet set, int epoch):
    // Этот метод обучает нейронную сеть на основе предоставленного набора данных.
    // Он выполняет прямое и обратное распространение ошибки для каждого эпоха обучения.
    // Инициализация сети: Сначала вызывается метод init(),
    // который инициализирует нейронную сеть, создавая нейроны для каждого слоя.
    // В каждой эпохе происходит следующее:
    // Перемешивание данных: Данные для обучения (set) перемешиваются.
    // Это делается для того, чтобы обучение не было зависимо от порядка данных.
    // Прямое и обратное распространение:
    // Для каждого элемента данных выполняется прямое распространение (forward(datum.getInputs())),
    // где вычисляются выходные значения нейронов, и обратное распространение (backward(datum.getTargets())),
    // где обновляются веса и смещения нейронов на основе ошибки.
    public void train(MLDataSet set, int epoch) {
        this.init();
        logger.info("Training Started");
        for (int i = 0; i < epoch; i++) {
            Collections.shuffle(set.getData());
            for (MLData datum : set.getData()) {
                forward(datum.getInputs());
                backward(datum.getTargets());
            }
        }
        logger.info("Training Finished");
    }

    //backward(double[] targets):
    // Этот метод выполняет обратное распространение ошибки в нейронной сети.
    // Он вычисляет градиенты для каждого нейрона в скрытом и выходном слоях,
    // а затем обновляет веса и смещения нейронов.
    // Вычисление градиента для выходного слоя:
    // Для каждого нейрона в выходном слое вычисляется градиент.
    // Градиент - это производная функции потерь по весам нейрона.
    // Он указывает, в каком направлении нужно изменить веса, чтобы уменьшить ошибку.
    // В этом случае целевые значения (targets) используются для вычисления ошибки.
    // Вычисление градиента для скрытого слоя:
    // Затем для каждого нейрона в скрытом слое также вычисляется градиент.
    // В отличие от выходного слоя, здесь нет целевых значений,
    // поэтому градиент вычисляется на основе градиентов следующего слоя (выходного слоя).
    // Обновление весов и смещений скрытого слоя:
    // После вычисления градиентов для скрытого слоя веса и смещения каждого нейрона в этом слое обновляются.
    // Обновление происходит в направлении, обратном градиенту, с шагом, определяемым скоростью обучения (learningRate).
    // Также используется момент (momentum) для учета предыдущих обновлений весов.
    // Обновление весов и смещений выходного слоя: Наконец, также обновляются веса и смещения каждого нейрона в выходном слое.
    private void backward(double[] targets) {
        int i = 0;
        for (Neuron neuron : outputLayer) {
            neuron.calculateGradient(targets[i++]);
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.calculateGradient();
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.updateConnections(learningRate, momentum);
        }
        for (Neuron neuron : outputLayer) {
            neuron.updateConnections(learningRate, momentum);
        }
    }

    // forward(double[] inputs): Этот метод выполняет прямое распространение в нейронной сети.
    // Он устанавливает входные значения для каждого нейрона во входном слое,
    // а затем вычисляет выходные значения для каждого нейрона в скрытом и выходном слоях.
    // Вот что происходит внутри этого метода:
    //
    // Установка входных значений для входного слоя:
    // Для каждого нейрона во входном слое устанавливается входное значение.
    // Входные значения (inputs) - это данные, которые подаются на вход нейронной сети.
    // Вычисление выходных значений для скрытого слоя:
    // Затем для каждого нейрона в скрытом слое вычисляется выходное значение.
    // Выходное значение нейрона вычисляется как взвешенная сумма его входных значений, пропущенная через функцию активации.
    // Вычисление выходных значений для выходного слоя:
    // Наконец, для каждого нейрона в выходном слое также вычисляется выходное значение.
    // Это делается таким же образом, как и для скрытого слоя.
    private void forward(double[] inputs) {
        int i = 0;
        for (Neuron neuron : inputLayer) {
            neuron.setOutput(inputs[i++]);
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.calculateOutput();
        }
        for (Neuron neuron : outputLayer) {
            neuron.calculateOutput();
        }
    }

    //predict(double… inputs):
    // Этот метод используется для получения прогнозов от нейронной сети на основе входных данных.
    // Он выполняет прямое распространение с входными данными
    // и возвращает выходные значения нейронов в выходном слое.
    // Вот что происходит внутри этого метода:
    //
    // Прямое распространение: Сначала выполняется прямое распространение с входными данными (inputs), вызывая метод forward(inputs).
    // Это проходит входные данные через нейронную сеть и генерирует выходные данные.
    // Сбор выходных данных: Затем выходные значения каждого нейрона в выходном слое собираются в массив output.
    // Логирование: В конце метода выводится сообщение с входными и предсказанными значениями.
    // Возврат предсказаний: Метод возвращает массив output, который содержит предсказанные значения нейронной сети.
    public double[] predict(double... inputs) {
        forward(inputs);
        double[] output = new double[outputLayer.size()];
        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.get(i).getOutput();
        }
        logger.info("Input: " + Arrays.toString(inputs) + " Predicted: " + Arrays.toString(output));
        return output;
    }

    public void runServerAt(int port) throws IOException {
        double[] layers = new double[3];
        layers[0] = inputSize;
        layers[1] = hiddenSize;
        layers[2] = outputSize;
        MultiLayerNetworkView.DATA_NETWORK = Arrays.toString(layers);
        MultiLayerNetworkView multiLayerNetworkView = new MultiLayerNetworkView(port);
        multiLayerNetworkView.start(NanoHTTPD.SOCKET_READ_TIMEOUT,false);
    }
}
