package org.bereft.process

import java.nio.file.Path
import java.nio.file.Paths

/**
 * A thin ProcessBuilder-based replacement for the helper scripts that used to live under `mp/bin/*`.
 *
 * These helpers existed primarily to provide cross-platform shell wrappers around Java main classes
 * (different class-path separators on Windows vs *nix), plus a handful of simple shell pipelines.
 *
 * By moving to Java/Kotlin `ProcessBuilder` we get the same behaviour in a way that is portable and
 * script-agnostic, without having to ship (or maintain) separate `.cmd`/`.sh` files.
 *
 * Only the most frequently used wrappers have been ported for now.  Adding more is as easy as
 * creating another public function that builds the command-line and forwards to [launch].
 */
object MpBinLauncher {

  /* ──────────────────────────────────────────────────────────────────────────── */
  /*  Public helper methods (replacements for the original shell scripts)        */
  /* ──────────────────────────────────────────────────────────────────────────── */

  /**
   * Equivalent of `mp/bin/ApiKeyServer.sh|cmd` – starts the `ApiKeyNodeKt` service.
   *
   * @param apiKeys   path to the api-keys text file (default `cfg/ApiKeys.txt`).
   */
  @JvmStatic
  fun apiKeyServer(apiKeys: Path = Paths.get("cfg", "ApiKeys.txt")): Int =
    runJavaMain(
      mainClass = "org.github.jnorthrup.node.execution.ApiKeyNodeKt",
      args = listOf(apiKeys.toString())
    )

  /**
   * Equivalent of `mp/bin/ShardNode.sh|cmd` – launches a shard node instance.
   *
   * @param shardCfg  path to the shard-configuration file (default `cfg/ShardCfg.txt`).
   */
  @JvmStatic
  fun shardNode(shardCfg: Path = Paths.get("cfg", "ShardCfg.txt")): Int =
    runJavaMain(
      mainClass = "org.github.jnorthrup.node.execution.ShardNodeKt",
      args = listOf(shardCfg.toString())
    )

  /**
   * Replacement for `mp/bin/KeyPair.sh|cmd`.  Generates an ed25519 key-pair by invoking the
   * `PrintKeyPairKt` main class and appends the result to the given config files.
   *
   * The original shell script sprinkled the generated values into two separate files.  Here we keep
   * the same behaviour for compatibility, but do it entirely from Java/Kotlin rather than relying on
   * `bash` array tricks.
   *
   * @param meshId    logical identifier to prefix the config entries with.
   * @param apiKeys   target file (defaults to `cfg/ApiKeys.txt`).
   * @param shardCfg  target file (defaults to `cfg/ShardCfg.txt`).
   */
  @JvmStatic
  fun keyPair(
    meshId: String,
    apiKeys: Path = Paths.get("./cfg", "ApiKeys.txt"),
    shardCfg: Path = Paths.get("./cfg", "ShardCfg.txt")
  ): Int {
    val (priv, pub) = invokePrintKeyPair() ?: return -1

    // Append to api-keys
    apiKeys.toFile().apply { parentFile.mkdirs(); appendText("mesh.id=$meshId\n") }
    apiKeys.toFile().appendText("$meshId.keypair.private=\"$priv\"\n")

    // Append to shard-cfg
    shardCfg.toFile().apply { parentFile.mkdirs(); appendText("mesh.id=$meshId\n") }
    shardCfg.toFile().appendText("$meshId.keypair.public=\"$pub\"\n")

    return 0
  }

  /**
   * Port of `mp/bin/allcachedpairs.sh` – prints the cached pair directories (to stdout).
   *
   * This reproduces the original behaviour by delegating to a small inlined shell pipeline. On
   * systems without a POSIX shell the same effect can be achieved via Java NIO, but invoking `bash`
   * keeps the implementation tiny and stable.
   *
   * @param mpData  root of the mpdata hierarchy (defaults to `~/mpdata`).
   */
  @JvmStatic
  fun allCachedPairs(mpData: Path = Paths.get(System.getProperty("user.home"), "mpdata")): Int {
    val cmd = listOf(
      "bash", "-c",
      "cd ${mpData.toAbsolutePath()} && find import/ -mindepth 4 -maxdepth 4 -type d | cut -d / -f4-5"
    )
    return launch(cmd)
  }

  /**
   * Minimal ProcessBuilder wrapper for `mp/bin/CmcSandboxData.sh`.
   *
   * Uses `curl` to hit the CoinMarketCap sandbox endpoint.  The API key must be supplied by the
   * caller (hard-wiring secrets in source code is bad practice!).
   *
   * @param apiKey   your CMC sandbox API key.
   * @param start    first record (default 1).
   * @param limit    number of records (default 5000).
   * @param convert  fiat symbol to convert against (default "USD").
   */
  @JvmStatic
  fun cmcSandboxData(
    apiKey: String,
    start: Int = 1,
    limit: Int = 5000,
    convert: String = "USD"
  ): Int {
    val cmd = listOf(
      "curl",
      "-H", "X-CMC_PRO_API_KEY: $apiKey",
      "-H", "Accept: application/json",
      "-d", "start=$start&limit=$limit&convert=$convert",
      "-G", "https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    )
    return launch(cmd)
  }

  /* ──────────────────────────────────────────────────────────────────────────── */
  /*  Internal helpers                                                           */
  /* ──────────────────────────────────────────────────────────────────────────── */

  /**
   * Generic helper that constructs the correct class-path for `target/classes` plus every JAR under
   * `target/lib`, taking the platform-specific path separator into account.
   */
  private fun runJavaMain(mainClass: String, args: List<String> = emptyList()): Int {
    val pathSeparator = if (System.getProperty("os.name").startsWith("Windows")) ";" else ":"
    val classPath = "./target/classes${pathSeparator}./target/lib/*"

    val cmdLine = mutableListOf("java", "-classpath", classPath, mainClass).apply { addAll(args) }
    return launch(cmdLine)
  }

  /**
   * Spawns the given command line using [ProcessBuilder], forwarding stdin/stdout/stderr to the
   * current process and returning the exit code.
   */
  private fun launch(cmdLine: List<String>): Int {
    val pb = ProcessBuilder(cmdLine)
      .redirectInput(ProcessBuilder.Redirect.INHERIT)
      .redirectOutput(ProcessBuilder.Redirect.INHERIT)
      .redirectError(ProcessBuilder.Redirect.INHERIT)

    val p = pb.start()
    return p.waitFor()
  }

  /**
   * Calls the `PrintKeyPairKt` utility and parses its stdout into `(priv, pub)`.
   */
  private fun invokePrintKeyPair(): Pair<String, String>? {
    val pathSeparator = if (System.getProperty("os.name").startsWith("Windows")) ";" else ":"
    val classPath = "./target/classes${pathSeparator}./target/lib/*"
    val cmdLine = listOf("java", "-classpath", classPath, "org.github.jnorthrup.runtime.PrintKeyPairKt")

    val pb = ProcessBuilder(cmdLine)
    val proc = pb.start()

    val output = proc.inputStream.bufferedReader().readLines()
    proc.waitFor()

    if (output.size < 2) return null
    return output[0] to output[1]
  }
}