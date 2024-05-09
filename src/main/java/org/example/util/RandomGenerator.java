package org.example.util;

public class RandomGenerator {

    //  Этот метод принимает два целочисленных параметра: min и max,
    //  и возвращает случайное число с плавающей точкой между min и max.
    public static double random(int min, int max) {
        //Метод Math.random() генерирует случайное число с плавающей точкой между 0.0 и 1.0.
        // Умножение этого числа на (max - min) дает случайное число в диапазоне от 0 до (max - min).
        // Добавление min к этому числу смещает диапазон к [min, max).
        // Обратите внимание, что max не включается в этот диапазон.
        return min + (max - min) * Math.random();
    }
}
