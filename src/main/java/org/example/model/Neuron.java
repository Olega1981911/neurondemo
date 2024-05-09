package org.example.model;

import lombok.Getter;
import lombok.Setter;
import org.example.activation.IActivationFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
/*
* Класс Neuron представляет собой нейрон в нейронной сети. Вот что делает каждый элемент этого класса:
* neuronId: Это уникальный идентификатор для каждого объекта Neuron.
* Он генерируется с помощью UUID.randomUUID(), который создает случайный UUID.
*
* incomingConnections и outgoingConnections: Это списки входящих и исходящих соединений для этого нейрона.
* Входящие соединения - это соединения от других нейронов к этому нейрону, а исходящие соединения - это соединения от этого нейрона к другим нейронам.
*
* bias: Это смещение нейрона, которое добавляется к взвешенной сумме входов перед применением функции активации.
*
* gradient: Это градиент ошибки по отношению к весам и смещениям нейрона.
* Он используется при обратном распространении ошибки для обновления весов и смещений.
*
* output и outputBeforeActivation: Это выход нейрона до и после применения функции активации.
*
* activationFunction: Это функция активации, которая применяется к взвешенной сумме входов и смещения.
*
* calculateOutput(): Этот метод вычисляет выход нейрона, применяя функцию активации к взвешенной сумме входов и смещения.
*
* error(double target): Этот метод вычисляет ошибку между целевым значением и выходом нейрона.
*
* calculateGradient(double target) и calculateGradient(): Эти методы вычисляют градиент ошибки по отношению к весам и смещениям нейрона.
*
* updateConnections(double lr, double mu): Этот метод обновляет веса и смещения нейрона на основе градиента и скорости обучения.
* */
@Getter
@Setter
public class Neuron {
    private UUID neuronId;
    private List<Connection> incomingConnections;
    private List<Connection> outgoingConnections;
    private double bias;
    private double gradient;
    private double output;
    private double outputBeforeActivation;
    private IActivationFunction activationFunction;

    public Neuron() {
        this.neuronId = UUID.randomUUID();
        this.incomingConnections = new ArrayList<>();
        this.outgoingConnections = new ArrayList<>();
        this.bias = 1.0;
    }

    public Neuron(List<Neuron> neurons, IActivationFunction activationFunction) {
        this();
        this.activationFunction = activationFunction;
        for (Neuron neuron : neurons) {
            Connection connection = new Connection(neuron, this);
            neuron.getOutgoingConnections().add(connection);
            this.incomingConnections.add(connection);
        }
    }

    public void calculateOutput() {
        this.outputBeforeActivation = 0.0;
        for (Connection connection : incomingConnections) {
            this.outputBeforeActivation += connection.getSynapticWeight() * connection.getFrom().getOutput();
        }
        this.output = activationFunction.output(this.outputBeforeActivation + bias);
    }

    public double error(double target) {
        return target - output;
    }

    public void calculateGradient(double target) {
        this.gradient = error(target) * activationFunction.outputDerivative(output);
    }

    public void calculateGradient() {
        this.gradient = outgoingConnections.stream().mapToDouble(connection -> connection.getTo().getGradient() * connection.getSynapticWeight()).sum()
                * activationFunction.outputDerivative(output);
    }

    public void updateConnections(double lr, double mu) {
        for (Connection connection : incomingConnections) {
            double prevDelta = connection.getSynapticWeightDelta();
            connection.setSynapticWeightDelta(lr * gradient * connection.getFrom().getOutput());
            connection.updateSynapticWeight(connection.getSynapticWeightDelta() + mu * prevDelta);
        }
    }
}
