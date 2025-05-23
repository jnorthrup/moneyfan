package com.moneyfan.dsl;

import com.moneyfan.common.Join;
public record ImmutableJoin<F, S>(F first, S second) implements Join<F, S> {
        @Override
        public F getFirst() {
            return first;
        }

        @Override
        public S getSecond() {
            return second;
        }
    }