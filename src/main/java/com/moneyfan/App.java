package com.moneyfan;

import com.moneyfan.dsel.DSEL_Cursor;
import com.moneyfan.dsel.Join;
import com.moneyfan.dsel.ListBackedCursor;

import java.util.Arrays;
import java.util.List;

/**
 * Main application entry point for demonstrating the DSEL.
 */
public class App {
    public static void main(String[] args) {
        // Example usage for Join and ListBackedCursor
        List<Join<String, Integer>> initialData = Arrays.asList(
            JoinOps.j("Alpha", 100),
            JoinOps.j("Beta", 200),
            JoinOps.j("Gamma", 300),
            JoinOps.j("Delta", 400)
        );
        ListBackedCursor<String, Integer> cursor = ListBackedCursor.of(initialData);

        System.out.println("Original Cursor:");
        cursor.toList().forEach(join -> System.out.println(JoinOps.str(join)));

        // Map first element (String to its length)
        ListBackedCursor<Integer, Integer> mappedFirst = cursor.mapFirst(String::length);
        System.out.println("\nCursor after mapFirst (string to length):");
        mappedFirst.toList().forEach(join -> System.out.println(JoinOps.str(join)));

        // Map second element (Integer to String representation)
        ListBackedCursor<String, String> mappedSecond = cursor.mapSecond(val -> "Value: " + val);
        System.out.println("\nCursor after mapSecond (int to string):");
        mappedSecond.toList().forEach(join -> System.out.println(JoinOps.str(join)));

        // Swap elements
        ListBackedCursor<Integer, String> swapped = cursor.swap();
        System.out.println("\nCursor after swap:");
        swapped.toList().forEach(join -> System.out.println(JoinOps.str(join)));

        // Filter (e.g., only elements where second > 250)
        ListBackedCursor<String, Integer> filtered = cursor.filter(join -> join.second() > 250);
        System.out.println("\nCursor after filter (second > 250):");
        filtered.toList().forEach(join -> System.out.println(JoinOps.str(join)));

        // Demonstrate head and tail (iterating through the cursor)
        System.out.println("\nIterating with head() and tail():");
        DSEL_Cursor<String, Integer> current = cursor;
        while (!current.isEmpty()) {
            System.out.println("Head: " + JoinOps.str(current.head()));
            current = current.tail();
        }
    }
}
