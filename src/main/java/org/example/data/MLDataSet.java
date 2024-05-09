package org.example.data;


import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

/*
* Класс MLDataSet представляет собой набор данных для машинного обучения.
*
* inputs и targets: Это двумерные массивы, которые содержат входные данные и целевые значения для набора данных.
* Входные данные - это значения, которые подаются на вход модели машинного обучения,
* а целевые значения - это значения, которые модель должна предсказать.
*
* data: Это список объектов MLData, которые содержат пары входных данных и целевых значений.
*
* Класс MLDataSet также определяет конструктор, который принимает входные данные и целевые значения,
* создает объекты MLData для каждой пары входных данных и целевых значений и добавляет их в список data.
* */
@Getter
@Setter
public class MLDataSet {
    private double[][] inputs;
    private double[][] targets;
    private List<MLData> data;

    public MLDataSet(double[][] inputs, double[][] targets) {
        this.data = new ArrayList<>();
        this.inputs = inputs;
        this.targets = targets;
        for (int i = 0; i < this.inputs.length; i++) {
            this.data.add(new MLData(inputs[i], targets[i]));
        }
    }
}
