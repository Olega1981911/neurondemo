package org.example;

import java.util.UUID;

public class Main {
    public static void main(String[] args) {
        UUID uuid = UUID.randomUUID();
        System.out.println("Сгенерированный UUID: " + uuid.toString());
    }
}