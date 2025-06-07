plugins {
    kotlin("jvm") version "1.9.22"
    application
    java
}

group = "com.vsiwest"
version = "1.0-SNAPSHOT"

// Configure Java options
java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

repositories {
    mavenCentral()
    // JitPack is already defined in settings.gradle.kts
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("com.github.jnorthrup:k2script:-SNAPSHOT")

    // Dependencies translated from pom.xml
    implementation("commons-codec:commons-codec:1.16.0")
    implementation("org.jetbrains:annotations:24.1.0")

    // HTTP Client
    implementation("org.apache.httpcomponents.client5:httpclient5:5.2.3")
    implementation("org.apache.httpcomponents.core5:httpcore5:5.2.4")

    // JSON Processing
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.15.2")

    // ZeroMQ
    implementation("org.zeromq:jeromq:0.5.4")

    // Logging
    implementation("org.slf4j:slf4j-simple:2.1.0-alpha1")

    // Test dependencies
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.10.2")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.10.2")
    testImplementation("org.mockito:mockito-core:5.12.0")
    testImplementation("org.mockito:mockito-junit-jupiter:5.12.0")
}

application {
    mainClass.set("com.vsiwest.moneyfan.DataIngestionManager")
}

// Configure JUnit Platform for testing
tasks.withType<Test> {
    useJUnitPlatform()
    testLogging {
        events("passed", "skipped", "failed")
    }
}

// Task to run the launch.main.kts script
tasks.register<JavaExec>("runMoneyfan") {
    group = "application"
    description = "Runs the DataIngestionManager via launch.main.kts using k2script (if k2script provides a main runner) or Kotlin's runner."

    // Classpath should include project's compiled classes, resources, and all runtime dependencies
    classpath = sourceSets.main.get().runtimeClasspath

    // The main class to execute.
    // If k2script provides a specific runner for .kts files that are part of a project, use that.
    // For example, if k2script means we treat .kts files as sources and compile them:
    // We might need to configure Kotlin compilation to include the scripts directory.
    // Or, if k2script is more like 'kscript' where it's a command that processes the script:
    // This JavaExec task might need to invoke the k2script command line tool.

    // Option 1: Treat the .kts file as a Kotlin main file (requires it to have a main function or be executable by Kotlin runner)
    // This assumes `launch.main.kts` is compiled to `LaunchMainKts.class` or similar.
    // We might need to adjust sourceSets if `scripts` is not compiled by default.
    // For simplicity, let's assume we want to execute it as a script with Kotlin's script runner.
    // This typically requires the Kotlin compiler and scripting modules on the classpath.

    // Add Kotlin scripting dependencies for running .kts directly via Kotlin's own mechanisms
    // This might be what k2script leverages or replaces.
    // These are often needed if you're not using a dedicated script runner from k2script.
    val kotlinVersion = "1.9.22" // ensure this matches project's kotlin version
    classpath += files(
        "org.jetbrains.kotlin:kotlin-compiler-embeddable:$kotlinVersion",
        "org.jetbrains.kotlin:kotlin-script-runtime:$kotlinVersion",
        "org.jetbrains.kotlin:kotlin-scripting-compiler-embeddable:$kotlinVersion",
        "org.jetbrains.kotlin:kotlin-scripting-jvm:$kotlinVersion"
        // k2script itself should be on the classpath via project dependencies already
    )

    mainClass.set("org.jetbrains.kotlin.cli.jvm.K2JVMCompiler") // Using the Kotlin compiler to run the script

    // Arguments to the Kotlin compiler/runner to execute the script
    // The exact arguments can be tricky and depend on the runner.
    // For K2JVMCompiler to run a script:
    args = listOf(
        "-script",
        file("scripts/launch.main.kts").absolutePath
        // Potentially add classpath arguments here if K2JVMCompiler -script doesn't inherit easily,
        // though JavaExec's classpath should ideally be picked up.
    )

    // If k2script has a specific main class for running scripts, replace mainClass and args:
    // e.g. mainClass.set("com.github.jnorthrup.k2script.Runner") // HYPOTHETICAL
    //      args = listOf(file("scripts/launch.main.kts").absolutePath)

    // Standard input, output, error
    standardInput = System.`in`
    standardOutput = System.out
    errorOutput = System.err
}
