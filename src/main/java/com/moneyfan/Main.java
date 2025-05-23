package com.moneyfan;

import com.moneyfan.dsel.core.Join;
import static com.moneyfan.dsel.core.Types.jn; // Using the shorthand

public class Main {
    public static void main(String[] args) {
        Join<String, Integer> exampleJoin = jn("Hello DSEL", 42);
        System.out.println("Main application started. Example Join: " + exampleJoin);
    }
}
