package com.example

fun main(args: Array<String>) {
    println(\"Starting Fusion Trader App...\")
    println(\"Usage: java -jar fusion-trader.jar [mode]\")
    println(\"  - Empty or 'simulation': Run the simulation in console\")
    println(\"  - 'server': Run the web server\")
    
    if (args.isNotEmpty() && args[0] == \"server\") {
        println(\"Starting Fusion Trader Server...\")
        val server = FusionTraderServer()
        server.startServer(7000)
    } else {
        val fusionTraderApp = FusionTraderApp()
        fusionTraderApp.startSimulation()
        println(\"\\nFusion Trader App simulation completed!\")
    }
}