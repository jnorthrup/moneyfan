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
  /*  Bash-script helper                                                         */
  /* ──────────────────────────────────────────────────────────────────────────── */

  private fun runScript(scriptName: String, scriptBody: String, args: List<String> = emptyList()): Int {
    val local = Paths.get("./mp/bin", scriptName)
    return if (java.nio.file.Files.exists(local)) {
      val cmd = mutableListOf("bash", local.toAbsolutePath().toString()).apply { addAll(args) }
      launch(cmd)
    } else {
      launchBashScript(scriptBody, args)
    }
  }

  private fun launchBashScript(scriptBody: String, args: List<String>): Int {
    val tmp = java.nio.file.Files.createTempFile("mpbin_", ".sh")
    java.nio.file.Files.writeString(tmp, scriptBody)
    tmp.toFile().setExecutable(true)
    val cmd = mutableListOf("bash", tmp.toAbsolutePath().toString()).apply { addAll(args) }
    val rc = launch(cmd)
    tmp.toFile().delete()
    return rc
  }

  /* ──────────────────────────────────────────────────────────────────────────── */
  /*  dayklines.sh                                                               */
  /* ──────────────────────────────────────────────────────────────────────────── */

  private const val DAYKLINES_SH = """
#!/usr/bin/env bash

: 4{MP_CACHE:=~/mpdata/cache}
: 4{MP_DATA:=~/mpdata}
: 4{MP_IMPORT:=~/mpdata/import}

new_chunks=( )

while true; do
  set -e
  TC="41:-BTC}"; CC="42:-USDT}"
  finalcsv="4{MP_IMPORT}/klines/1m/4{TC}/4{CC}/final-\4{TC}-\4{CC}-1m.csv"
  since=$(tail -n 1 "4{finalcsv}" | cut -f7 -d ,)

  segment=$(mktemp)
  curl -s "https://api.binance.com/api/v3/klines?symbol=4{TC}4{CC}&interval=1m&startTime=$((since))&limit=1000" | \
    sed -e 's/\[\[//g' -e 's/\]\]/\n/g' -e 's/\],\[/\n/g' | tr -d '"' > "4{segment}"

  linespulled=$(wc -l < "4{segment}")
  cat "4{segment}" >> "4{finalcsv}"
  new_chunks+=( "4{segment}" )
  if (( linespulled < 999 )); then
    echo "--------segments--------"
    echo "4{new_chunks[*]}"
    exit 0
  fi
  # otherwise loop again with new since value
  sleep 1
done
"""

  @JvmStatic
  fun dayKlines(tc: String = "BTC", cc: String = "USDT", additionalArgs: List<String> = emptyList()): Int =
    runScript("dayklines.sh", DAYKLINES_SH, listOf(tc, cc) + additionalArgs)

  /* ──────────────────────────────────────────────────────────────────────────── */
  /*  fetchklines.sh                                                             */
  /* ──────────────────────────────────────────────────────────────────────────── */

  private const val FETCHKLINES_SH = """
#!/usr/bin/env bash

#vars
: 4{MP_CACHE:=~/mpdata/cache}
: 4{MP_DATA:=~/mpdata}
: 4{MP_IMPORT:=~/mpdata/import}

set -e
mkdir -p 4{MP_CACHE} 4{MP_IMPORT}
export TC="41:-BTC}"
export CC="42:-USDT}"
export TUNIT="4{TIME_UNIT:-1m}"

export DT=$(date -u +'%Y-%m')
export CLEAN=$(date -u  -d "-1 month" +%Y-%m)
export KLINECACHE="43:-4{MP_CACHE}/klines/4{TUNIT}/4{TC}/4{CC}}"
export TARGET="44:-4{MP_IMPORT}/klines/4{TUNIT}/4{TC}/4{CC}}"

x="$(mktemp)"; rm "4{x}"; mkdir -p "4{x}" "4{TARGET}" "4{KLINECACHE}"

pushd "4{KLINECACHE}" >/dev/null
aria2c   -Z -c -{x,j,s}\ 15 -d "4{KLINECACHE}" \
  https://data.binance.vision/data/spot/{monthly/klines/4{TC}4{CC}/4{TUNIT}/4{TC}4{CC}-4{TUNIT}-20{17..22}-{0{1..9},1{0..2}},daily/klines/4{TC}4{CC}/4{TUNIT}/4{TC}4{CC}-4{TUNIT}-4{DT}-{0{1..9},{10..31}}}.zip{,.CHECKSUM}

pushd "4{x}" >/dev/null
unzip -aa -n "4{KLINECACHE}/4{TC}4{CC}-4{TUNIT}-*.zip"

echo 'Open_time,Open,High,Low,Close,Volume,Close_time,Quote_asset_volume,Number_of_trades,Taker_buy_base_asset_volume,Taker_buy_quote_asset_volume,Ignore' > "4{TARGET}/final-4{TC}-4{CC}-4{TUNIT}.csv"
	sort -fu  4{TC}4{CC}-4{TUNIT}*.csv | \
      grep  --extended-regexp -e '(.*,){11}' | \
      sed  --posix --regexp-extended 's/(\.[0-9]+])0+,/\1,/g' >> "4{TARGET}/final-4{TC}-4{CC}-4{TUNIT}.csv" || rm "4{TARGET}/final-4{TC}-4{CC}-4{TUNIT}.csv"

rm -fr "4{x}"
popd >/dev/null; popd >/dev/null
"""

  @JvmStatic
  fun fetchKlines(tc: String = "BTC", cc: String = "USDT", tunit: String = "1m", extra: List<String> = emptyList()): Int =
    runScript("fetchklines.sh", FETCHKLINES_SH, listOf(tc, cc, tunit) + extra)

  /* ──────────────────────────────────────────────────────────────────────────── */
  /*  fetchtrades.sh                                                             */
  /* ──────────────────────────────────────────────────────────────────────────── */

  private const val FETCHTRADES_SH = """
#!/usr/bin/env bash

set -e
export TC="41:-BTC}" CC="42:-USDT}"
export TUNIT="4{TIME_UNIT:-1m}"

export BASE="43:-~/mpdata/cache/trades/4{TUNIT}/4{TC}/4{CC}}"
export TARGET="44:-~/mpdata/import/trades/4{TUNIT}/4{TC}/4{CC}}"

x="$(mktemp)"; rm "4{x}"; mkdir -p "4{x}" "4{TARGET}" "4{BASE}"

pushd "4{BASE}" >/dev/null
aria2c -Z -c -{x,j,s}\ 15 -d "4{BASE}" \
  https://data.binance.vision/data/spot/{monthly/trades/4{TC}4{CC}/4{TUNIT}/4{TC}4{CC}-4{TUNIT}-20{17..21}-{0{1..9},1{0..2}},daily/trades/4{TC}4{CC}/4{TUNIT}/4{TC}4{CC}-4{TUNIT}-$(date +'%Y-%m')-{0{1..9},{10..31}}}.zip{,.CHECKSUM}

popd >/dev/null
pushd "4{x}" >/dev/null
unzip "4{BASE}/*.zip"

echo "trade Id,price,qty,quoteQty,time,isBuyerMaker,isBestMatch" > "4{TARGET}/4{TUNIT}.csv"
cat 4{TC}4{CC}-4{TUNIT}*.csv >> "4{TARGET}/4{TUNIT}.csv" || rm "4{TARGET}/4{TUNIT}.csv"

popd >/dev/null
rm -fr "4{x}"
"""

  @JvmStatic
  fun fetchTrades(tc: String = "BTC", cc: String = "USDT", tunit: String = "1m", extra: List<String> = emptyList()): Int =
    runScript("fetchtrades.sh", FETCHTRADES_SH, listOf(tc, cc, tunit) + extra)

  /* ──────────────────────────────────────────────────────────────────────────── */
  /*  tweeze.sh & meta-klines.sh                                                 */
  /* ──────────────────────────────────────────────────────────────────────────── */

  // To keep the source size manageable we embed the *unmodified* tweeze.sh and meta-klines.sh bodies
  // in resource files under `src/main/resources/mpbin/`. If those resources are not found we skip.

  private fun loadResourceOrEmpty(path: String): String =
    MpBinLauncher::class.java.classLoader.getResource(path)?.readText() ?: ""

  @JvmStatic
  fun tweeze(vararg extraArgs: String): Int {
    val body = loadResourceOrEmpty("mpbin/tweeze.sh")
    if (body.isBlank()) {
      System.err.println("tweeze.sh resource not found – please place script under mp/bin or resources/mpbin/")
      return -1
    }
    return runScript("tweeze.sh", body, extraArgs.asList())
  }

  @JvmStatic
  fun metaKlines(vararg traded: String): Int {
    val body = loadResourceOrEmpty("mpbin/meta-klines.sh")
    if (body.isBlank()) {
      System.err.println("meta-klines.sh resource not found – please place script under mp/bin or resources/mpbin/")
      return -1
    }
    return runScript("meta-klines.sh", body, traded.asList())
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