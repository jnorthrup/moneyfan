// launch.main.kts
// This script is intended to be run in an environment where k2script is available
// and the classpath for moneyfan project is correctly set up (e.g., by a Gradle task).

// @file:DependsOn("com.github.jnorthrup:k2script:-SNAPSHOT") // May not be needed if run via Gradle with k2script on classpath

import com.vsiwest.moneyfan.DataIngestionManager
import kotlin.system.exitProcess

println("k2script: Starting DataIngestionManager from launch.main.kts...")

try {
    DataIngestionManager.main(emptyArray())
    println("k2script: DataIngestionManager.main() called. The application should be running.")
    // If DataIngestionManager's main method is blocking and runs indefinitely,
    // this script might appear to hang here, which is expected.
    // If it exits, the script will also exit.
} catch (e: Exception) {
    println("k2script: Error launching DataIngestionManager: ${e.message}")
    e.printStackTrace()
    exitProcess(1)
}
