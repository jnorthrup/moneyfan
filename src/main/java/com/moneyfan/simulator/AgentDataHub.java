package com.moneyfan.simulator;

import com.moneyfan.dsel.D;
import com.moneyfan.dsel.core.Join;
import com.moneyfan.dsel.core.Series;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * AgentDataHub manages shared data among trading agents, allowing them to publish and access indicator rewards
 * or other relevant data in a thread-safe manner using DSEL structures.
 */
public class AgentDataHub {
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private Map<String, Double> agentRewards = new HashMap<>();

    /**
     * Publishes an agent's indicator reward or data value.
     * @param agentId The unique identifier of the agent.
     * @param reward The reward or data value to publish.
     */
    public void publishReward(String agentId, double reward) {
        lock.writeLock().lock();
        try {
            agentRewards.put(agentId, reward);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Retrieves the shared data as a DSEL Series of agent ID and reward pairs.
     * @return A Series of Join<String, Double> representing agent IDs and their published rewards.
     */
    public Series<Join<String, Double>> getSharedData() {
        lock.readLock().lock();
        try {
            Map<String, Double> snapshot = new HashMap<>(agentRewards);
            return D.sr(snapshot.size(), i -> {
                String agentId = snapshot.keySet().toArray(new String[0])[i];
                return D.jn(agentId, snapshot.get(agentId));
            });
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Clears the shared data, typically at the start of a new simulation cycle.
     */
    public void reset() {
        lock.writeLock().lock();
        try {
            agentRewards.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }
}