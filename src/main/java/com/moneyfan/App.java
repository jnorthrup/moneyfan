package com.moneyfan;

import com.moneyfan.dsel.DSEL;
import com.moneyfan.dsel.DSEL_Cursor;
import com.moneyfan.dsel.Join;
import com.moneyfan.dsel.OpUtils;

import java.util.List;

public class App {
    public static void main(String[] args) {
        System.out.println("MoneyFan DSEL Demo Start");

        Join<String, Integer> j1 = Join.of("Alice", 30);
        Join<String, Integer> j2 = Join.of("Bob", 25);
        Join<String, Integer> j3 = Join.of("Charlie", 35);
        Join<String, Integer> j4 = Join.of("David", 20);

        DSEL_Cursor<String, Integer> users = DSEL.INSTANCE.of(j1, j2, j3, j4);
        System.out.println("Initial Users: " + users);

        // Shorthand mapFirst (mfst)
        DSEL_Cursor<Integer, Integer> nameLengths = users.mfst(String::length);
        System.out.println("Name Lengths to Age: " + nameLengths);

        // Shorthand mapSecond (msnd)
        DSEL_Cursor<String, String> userAgesStr = users.msnd(age -> "Age: " + age);
        System.out.println("User Ages as String: " + userAgesStr);

        // Shorthand filter (flt)
        DSEL_Cursor<String, Integer> usersOver25 = users.flt(user -> user.second() > 25);
        System.out.println("Users over 25: " + usersOver25);

        // Shorthand filterFirst (fltFst)
        DSEL_Cursor<String, Integer> usersShortName = users.fltFst(name -> name.length() < 5);
        System.out.println("Users with short names (<5 chars): " + usersShortName);

        // Shorthand swap (swp)
        DSEL_Cursor<Integer, String> ageToUser = users.swp();
        System.out.println("Age to User: " + ageToUser);

        // Chaining with shorthands
        List<Join<Integer, String>> processed = DSEL.INSTANCE.of(
                Join.of("Eve", 28), Join.of("Frank", 40), Join.of("Grace", 22), Join.of("Ivy", 33)
            )
            .fltSnd(age -> age >= 25) // filterSecond: keep users age 25+
            .mbth( // mapBoth
                name -> name.toUpperCase(),
                age -> age * 2
            )
            .swp() // swap: Join<Integer, String> (age*2, NAME)
            .fltFst(doubledAge -> doubledAge < 70) // filterFirst: keep if doubledAge < 70
            .col(); // collect

        System.out.println("Processed & Collected List:");
        processed.forEach(System.out::println);
        System.out.println("MoneyFan DSEL Demo End");
    }
}
