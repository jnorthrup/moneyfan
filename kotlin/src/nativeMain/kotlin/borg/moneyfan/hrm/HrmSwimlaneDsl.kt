@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
package borg.moneyfan.hrm

import kotlinx.cinterop.*

object HrmSwimlaneDsl {
    private val laneRegex = Regex("""^lane\s+(\d+)\s+(.*)$""")
    private const val GRAD_DIM = 4

    // fast, slow, sig, sharp coefficient priors by archetype
    private val archetypeBasis: Map<String, DoubleArray> = mapOf(
        "grid"               to doubleArrayOf(0.90, 1.10, 1.05, 0.90),
        "volatile_breakout"  to doubleArrayOf(1.20, 0.85, 0.95, 1.25),
        "trend"              to doubleArrayOf(1.10, 1.00, 1.00, 1.05),
        "mean_reversion"     to doubleArrayOf(0.95, 1.10, 1.15, 0.90),
        "momentum"           to doubleArrayOf(1.15, 0.95, 0.95, 1.10),
        "liquidity_rotation" to doubleArrayOf(1.00, 1.05, 0.95, 1.00),
    )

    // fast, slow, sig, sharp dampening/boost by risk tier
    private val riskBasis: Map<String, DoubleArray> = mapOf(
        "protective" to doubleArrayOf(0.90, 1.15, 1.15, 0.85),
        "normal"     to doubleArrayOf(1.00, 1.00, 1.00, 1.00),
        "caution"    to doubleArrayOf(0.95, 1.05, 1.10, 0.92),
    )

    /** Read all lines from a file path using POSIX/native I/O (no java.nio). */
    private fun readAllLines(path: String): List<String> {
        val f = platform.posix.fopen(path, "r") ?: error("swimlane manifest not found: $path")
        val lines = mutableListOf<String>()
        try {
            val buf = ByteArray(4096)
            while (true) {
                val read = kotlinx.cinterop.memScoped {
                    platform.posix.fgets(buf.refTo(0), buf.size, f)
                } ?: break
                lines += buf.toKString().trimEnd('\n', '\r')
            }
        } finally {
            platform.posix.fclose(f)
        }
        return lines
    }

    fun load(path: String): List<HrmSwimlaneSpec> {
        val lanes = mutableListOf<HrmSwimlaneSpec>()
        readAllLines(path).forEach { raw ->
            val line = raw.trim()
            if (line.isEmpty() || line.startsWith("#")) return@forEach
            val match = laneRegex.matchEntire(line) ?: return@forEach
            val laneId = match.groupValues[1].toInt()
            val attrsRaw = match.groupValues[2]
            val attrs = parseAttrs(attrsRaw)

            lanes += HrmSwimlaneSpec(
                laneId   = laneId,
                archetype = attrs["archetype"] ?: "generic",
                riskTier  = attrs["risk"] ?: "normal",
                weight    = parseQ(attrs["weight"], 1.0),
                fast      = parseQ(attrs["fast"],   12.0),
                slow      = parseQ(attrs["slow"],   26.0),
                sig       = parseQ(attrs["sig"],     9.0),
                sharp     = parseQ(attrs["sharp"],   1.0),
            )
        }
        return lanes.sortedBy { it.laneId }
    }

    fun requireArchetypes(specs: List<HrmSwimlaneSpec>, required: Set<String>) {
        val normRequired = required.map { it.trim().lowercase() }.toSet()
        if (normRequired.isEmpty()) return
        val present = specs.map { it.archetype.trim().lowercase() }.toSet()
        val missing = normRequired - present
        require(missing.isEmpty()) {
            "missing required swimlane archetypes: ${missing.sorted().joinToString(", ")}"
        }
    }

    /**
     * Build a 4-D categorical gradient glyph (fast, slow, sig, sharp) from
     * archetype + risk priors composed with quant lane parameters.
     */
    fun gradGlyph(spec: HrmSwimlaneSpec): DoubleArray {
        val archetype = spec.archetype.trim().lowercase()
        val risk      = spec.riskTier.trim().lowercase()

        val archetypeVec = archetypeBasis[archetype] ?: doubleArrayOf(1.0, 1.0, 1.0, 1.0)
        val riskVec      = riskBasis[risk] ?: doubleArrayOf(1.0, 1.0, 1.0, 1.0)

        val tempo    = (spec.fast / kotlin.math.max(spec.slow, 1e-9)).coerceIn(0.05, 2.5)
        val sigRatio = (spec.sig / kotlin.math.max(spec.slow, 1e-9)).coerceIn(0.01, 2.5)
        val quantVec = doubleArrayOf(
            spec.weight.coerceIn(0.1, 4.0),
            (1.0 / tempo).coerceIn(0.4, 2.5),
            sigRatio.coerceIn(0.2, 2.5),
            spec.sharp.coerceIn(0.1, 4.0),
        )

        val out = DoubleArray(GRAD_DIM)
        var i = 0
        while (i < GRAD_DIM) {
            out[i] = archetypeVec[i] * riskVec[i] * quantVec[i]
            i += 1
        }
        return out
    }

    /** Human-readable glyph tag for diagnostics/telemetry. */
    fun gradGlyphTag(spec: HrmSwimlaneSpec): String =
        "${spec.archetype.trim().lowercase()}⊗${spec.riskTier.trim().lowercase()}⊗q"

    fun neutralGlyph(): DoubleArray = DoubleArray(GRAD_DIM) { 1.0 }

    private fun parseAttrs(attrsRaw: String): Map<String, String> {
        val out = linkedMapOf<String, String>()
        attrsRaw.split(Regex("""\s+""")).forEach { token ->
            val eq = token.indexOf('=')
            if (eq <= 0) return@forEach
            val k = token.substring(0, eq).trim().lowercase()
            val v = token.substring(eq + 1).trim()
            if (k.isNotEmpty() && v.isNotEmpty()) out[k] = v
        }
        return out
    }

    private fun parseQ(v: String?, fallback: Double): Double {
        val raw = (v ?: "").trim()
        if (raw.isEmpty()) return fallback
        val unwrapped = if (raw.startsWith("q(") && raw.endsWith(")")) {
            raw.substring(2, raw.length - 1)
        } else raw
        return unwrapped.toDoubleOrNull() ?: fallback
    }
}
