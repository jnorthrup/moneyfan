package vec.util

/**
 * Safely convert a nullable [String] to [Double]. If the conversion fails or the value is null
 * the [default] value is returned.
 */
fun todub(value: String?, default: Double = Double.NaN): Double =
    value?.toDoubleOrNull() ?: default