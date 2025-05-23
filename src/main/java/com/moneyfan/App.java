package com.moneyfan;

import com.moneyfan.dsel.DSEL_Cursor;
import com.moneyfan.dsel.Join;
import com.moneyfan.dsel.JoinOps;
import com.moneyfan.dsel.ListBackedCursor;

import java.util.Arrays;
import java.util.List;

/**
 * Main application entry point for demonstrating the DSEL.
 */
public class App {
    public static void main(String[] args) {
        System.out.println("DSEL Cursor Demo");
        
        // Initial data (List of Join records using JoinOps.cj for conciseness)
        List<Join<String, Integer>> initialData = Arrays.asList(
            JoinOps.cj("apple", 10),
            JoinOps.cj("banana", 20),
            JoinOps.cj("cherry", 5),
            JoinOps.cj("date", 25),
            JoinOps.cj("elderberry", 15)
        );
        
        DSEL_Cursor<String, Integer> cursor = new ListBackedCursor<>(initialData);
        
        System.out.println("\nOriginal Data:");
        cursor.print(-1);  // Print all elements

        // 1. Map First (mf): Convert string to uppercase
        System.out.println("\n1. Map First (String to Uppercase):");
        DSEL_Cursor<String, Integer> mappedFirst = cursor.mf(String::toUpperCase);
        mappedFirst.print(-1);

        // 2. Map Second (ms): Double the integer
        System.out.println("\n2. Map Second (Integer Doubled):");
        DSEL_Cursor<String, Integer> mappedSecond = cursor.ms(x -> x * 2);
        mappedSecond.print(-1);

        // 3. Filter (fl): Keep only Joins where integer value > 15
        System.out.println("\n3. Filter (Integer > 15 on original):");
        DSEL_Cursor<String, Integer> filtered = cursor.fl(join -> join.second() > 15);
        filtered.print(-1);

        // 4. Map Both (mb): Combine string and integer into a new string, keep original integer
        System.out.println("\n4. Map Both (New String, Original Integer):");
        DSEL_Cursor<String, Integer> mappedBoth = cursor.mb((str, num) -> JoinOps.cj(str + ":" + num, num));
        mappedBoth.print(-1);

        // 5. Chaining operations: mf, then fl, then ms
        System.out.println("\n5. Chained: Uppercase First, Filter (Integer > 10), Double Second:");
        DSEL_Cursor<String, Integer> chained = cursor
                .mf(String::toUpperCase)
                .fl(join -> join.second() > 10)
                .ms(x -> x * 2);
        chained.print(-1);

        // 6. Head (tk)
        System.out.println("\n6. Head (Top 2):");
        cursor.head(2).print(-1);

        // 7. Tail (tl)
        System.out.println("\n7. Tail (Last 2 from original):");
        cursor.tail(2).print(-1);

        // 8. Skip (sk)
        System.out.println("\n8. Skip (First 2 then print remaining):");
        cursor.skip(2).print(-1);

        // 9. Collect to List (cl)
        System.out.println("\n9. Collect to List:");
        List<Join<String, Integer>> collectedList = cursor.collect();
        collectedList.forEach(System.out::println);

        // 10. Count (ct)
        System.out.println("\n10. Count of elements in original cursor:");
        System.out.println("Count: " + cursor.count());

        // 11. Swap elements using default swp() method
        System.out.println("\n11. Swap elements:");
        DSEL_Cursor<Integer, String> swapped = cursor.swp();
        swapped.print(-1);

        System.out.println("\n--- End of DSEL Cursor Demo ---");
    }
}
