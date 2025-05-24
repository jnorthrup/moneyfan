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
    // AgentId -> Reward/DataValue
    private final Map<String, Double> agentData = new HashMap<>();

    /**
     * Publishes an agent's indicator reward or data value.
     * @param agentId The unique identifier of the agent.
     * @param dataValue The data value to publish (e.g., reward, indicator value).
     */
    public void publishData(String agentId, double dataValue) {
        lock.writeLock().lock();
        try {
            agentData.put(agentId, dataValue);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Retrieves the shared data as a DSEL Series of agent ID and data value pairs.
     * @return A Series of Join<String, Double> representing agent IDs and their published data.
     */
    public Series<Join<String, Double>> getSharedData() {
        lock.readLock().lock();
        try {
            // Create an immutable snapshot of the map for the Series generation
            Map<String, Double> snapshot = new HashMap<>(agentData);
            return D.sr(snapshot.size(), i -> {
                String agentId = (String) snapshot.keySet().toArray()[i]; // Convert set to array to get by index
                return D.jn(agentId, snapshot.get(agentId));
            });
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Resets the shared data, typically at the start of a new simulation cycle.
     */
    public void reset() {
        lock.writeLock().lock();
        try {
            agentData.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }
}
