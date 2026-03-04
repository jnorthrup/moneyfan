package borg.moneyfan.hrm

data class HrmSwimlaneSpec(
    val laneId: Int,
    val archetype: String,
    val riskTier: String = "normal",
    val weight: Double = 1.0,
    val fast: Double = 12.0,
    val slow: Double = 26.0,
    val sig: Double = 9.0,
    val sharp: Double = 1.0,
)
