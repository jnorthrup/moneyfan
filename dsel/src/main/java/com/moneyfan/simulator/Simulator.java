package com.moneyfan.simulator;

import com.moneyfan.dsel.core.Join;
import static com.moneyfan.dsel.core.enums.Ops.j; // Example of using concise glyph

public class Simulator {
    public static void main(String[] args) {
        System.out.println("MoneyFan Simulator starting...");

        // Example usage of Join and DSEL constructs
        Join<Integer, String> dataPoint = j(1, "ExampleData");
        System.out.println("Initial data point: " + dataPoint);

        Join<String, String> processedPoint = dataPoint
            .mapFst(i -> "ID-" + i)
            .mapSnd(s -> s.toUpperCase());
        System.out.println("Processed data point: " + processedPoint);

        System.out.println("Simulator finished basic DSEL demonstration.");
    }
}
