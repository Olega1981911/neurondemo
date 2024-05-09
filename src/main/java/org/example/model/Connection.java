package org.example.model;

import lombok.Getter;
import lombok.Setter;
import org.example.util.RandomGenerator;


import java.util.UUID;
/*
* класс Connection,
* который представляет собой связь между двумя нейронами в нейронной сети.
* Вот что делает каждый элемент этого класса:
* connectionId: Это уникальный идентификатор для каждого объекта Connection.
* Он генерируется с помощью UUID.randomUUID(), который создает случайный UUID.
* from и to: Это нейроны, которые связаны этим соединением.
* from - это нейрон, который отправляет сигнал,
* а to - это нейрон, который получает сигнал.
* synapticWeight: Это вес синапса, который определяет силу связи между from и to.
* Он инициализируется случайным значением между -2 и 2 с помощью RandomGenerator.random(-2, 2).
*synapticWeightDelta: Это изменение веса синапса.
* Он не инициализируется в конструкторе, поэтому его начальное значение будет 0.
*
* */

@Getter
@Setter
public class Connection {
    private UUID connectionId;
    private Neuron from;
    private Neuron to;
    private double synapticWeight;
    private double synapticWeightDelta;

    public Connection(Neuron from, Neuron to) {
        this.connectionId = UUID.randomUUID();
        this.from = from;
        this.to = to;
        this.synapticWeight = RandomGenerator.random(-2, 2);
    }

    //Данный метод обновляет вес синапса в нейронной сети.
    // Вес синапса - это значение,
    // которое определяет силу связи между двумя нейронами.
    // Когда нейрон A отправляет сигнал нейрону B,
    // вес синапса между A и B определяет, насколько сильно сигнал будет влиять на нейрон B.
    public void updateSynapticWeight(double synapticWeight) {
        this.synapticWeight += synapticWeight;
    }
}
