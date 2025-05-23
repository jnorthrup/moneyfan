package com.moneyfan.simulator.model;

import com.moneyfan.dsel.D; // Assuming D contains 'sr' and other DSEL ops
import com.moneyfan.dsel.core.Series;

// Represents an agent's decision for a single asset.
// Using Series<Double> to align with DSEL, size determined by AssetMutation.OUTPUT_SIZE
public interface AssetOutput extends Series<Double> {
    default double get(AssetMutation mutation) {
        return D.seriesGet(this, mutation.ordinal());
    }

    static AssetOutput create(double[] values) {
        if (values.length != AssetMutation.OUTPUT_SIZE) {
            throw new IllegalArgumentException("Output array size must match AssetMutation.OUTPUT_SIZE");
        }
        return (AssetOutput) D.createSeries(values.length, i -> values[i]);
    }

    static AssetOutput createDefault() {
        return (AssetOutput) D.createSeries(AssetMutation.OUTPUT_SIZE, i -> AssetMutation.ALL_MUTATIONS[i].defaultValue);
    }
}
