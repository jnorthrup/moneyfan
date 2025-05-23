package com.moneyfan;

import com.moneyfan.dsel.Join;
import com.moneyfan.dsel.Jn;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Stream;

import static com.moneyfan.dsel.Jn.*; // Static import for shorthand operations

public class Main {

    public static void main(String[] args) {
        System.out.println("DSEL Cursor Demo\n");

        // Original data as a list of Joins
        List<Join<String, Integer>> originalData = List.of(
                jn("apple", 10),
                jn("banana", 20),
                jn("cherry", 5),
                jn("date", 25),
                jn("elderberry", 15)
        );

        System.out.println("Original Data:");
        originalData.forEach(System.out::println);
        System.out.println();

        // 1. Map First (String to Uppercase)
        System.out.println("1. Map First (String to Uppercase):");
        originalData.stream()
                .map(j -> mf(j, StrOps.TO_UPPER.get())) // Using shorthand mf and enum for function
                .forEach(System.out::println);
        System.out.println();

        // 2. Map Second (Integer Doubled)
        System.out.println("2. Map Second (Integer Doubled):");
        originalData.stream()
                .map(j -> ms(j, IntOps.DOUBLED.get())) // Using shorthand ms and enum for function
                .forEach(System.out::println);
        System.out.println();

        // 3. Filter (Integer > 15 on original)
        System.out.println("3. Filter (Integer > 15 on original):");
        fls(originalData.stream(), IntPreds.GREATER_THAN_15.get()) // Using shorthand fls and enum for predicate
                .forEach(System.out::println);
        System.out.println();

        // 4. Map Both (New String, Original Integer)
        System.out.println("4. Map Both (New String, Original Integer):");
        originalData.stream()
                .map(j -> mb(j, s -> s + ":" + j.second(), Function.identity())) // Lambda capturing 'j' for composition
                .forEach(System.out::println);
        System.out.println();

        // 5. Chained: Uppercase First, Filter (Integer > 10), Double Second
        System.out.println("5. Chained: Uppercase First, Filter (Integer > 10), Double Second:");
        originalData.stream()
                .map(j -> mf(j, StrOps.TO_UPPER.get())) // Uppercase First
                .filter(j -> j.second() > 10) // Filter (Integer > 10) - direct lambda for simple predicate
                .map(j -> ms(j, IntOps.DOUBLED.get())) // Double Second
                .forEach(System.out::println);
        System.out.println();

        // 6. Head (Top 2)
        System.out.println("6. Head (Top 2):");
        hd(originalData.stream(), 2) // Using shorthand hd
                .forEach(System.out::println);
        System.out.println();

        // 7. Tail (Last 2 from original) - This requires collecting and then skipping, or using a custom collector.
        // For simplicity and to demonstrate 'skip', we'll show skip first, then explain tail.
        // A true 'tail' would be `skip(count - n)`.
        System.out.println("7. Tail (Last 2 from original):");
        long count = ct(originalData.stream()); // Get count first
        sk(originalData.stream(), count - 3) // Skip all but the last 3 (to get last 3)
                .forEach(System.out::println);
        System.out.println();

        // 8. Skip (First 2 then print remaining)
        System.out.println("8. Skip (First 2 then print remaining):");
        sk(originalData.stream(), 2) // Using shorthand sk
                .forEach(System.out::println);
        System.out.println();

        // 9. Collect to List
        System.out.println("9. Collect to List:");
        List<Join<String, Integer>> collectedList = cl(originalData.stream()); // Using shorthand cl
        collectedList.forEach(System.out::println);
        System.out.println();

        // 10. Count of elements in original cursor:
        System.out.println("10. Count of elements in original cursor:");
        System.out.println("Count: " + ct(originalData.stream())); // Using shorthand ct
        System.out.println();

        // 11. Swap elements:
        System.out.println("11. Swap elements:");
        originalData.stream()
                .map(Jn::sw) // Using method reference for shorthand sw
                .forEach(System.out::println);
        System.out.println();

        System.out.println("--- End of DSEL Cursor Demo ---");
    }
}