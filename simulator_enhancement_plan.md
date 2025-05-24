# Simulator Enhancement Plan

## High-Level Overview
- **Load Pairs from Archive:** The simulator will access the HTTP archive to retrieve the toplevel index of pairs from Binance data. This data will be loaded into memory, parsed, and integrated into the Simulator class to provide realistic initial data for simulations.
- **Synchronizer for Initial Timestamps:** Implement a synchronizer that aligns the simulation's start time with the timestamps from the loaded archive data, ensuring the simulation begins at the earliest timestamp to maintain accuracy.
- **Ingest Step for Contiguous Timestamp Counts:** Add an ingest process that examines records for contiguous timestamps, checking for sequential ordering without significant gaps. This will help in identifying and handling data inconsistencies during simulation runs.

## Specifications
1. **Loading Pairs:**
   - Use Java's HttpClient to fetch the archive data from the HTTP source.
   - Parse the toplevel index (assumed to be in JSON or CSV format) to extract pairs.
   - Store the pairs in a data structure like a Series of Joins (jn) in the Simulator class for easy access.

2. **Timestamp Synchronizer:**
   - Upon loading data, extract the initial timestamp from the first record.
   - Synchronize by calculating an offset between the archive's timestamp and the system time, then apply this offset to all simulation timestamps to ensure alignment with data timestamps.

3. **Ingest Step:**
   - During data ingestion, iterate through timestamps and verify if the difference between consecutive ones meets the contiguous criteria (e.g., no gaps larger than a predefined threshold like 1 second).
   - If timestamps are not contiguous, log the discrepancy or trigger a fallback mechanism, such as skipping or interpolating the data.

## Recommendations on Where to Add Logic
- **Existing Classes:**
  - Extend Simulator.java with a new method for loading and synchronizing data.
  - Integrate the ingest check into the runSimulation() method or the process() method in SkimmerSwimlane.java to validate data before processing.
- **New Classes:**
  - Introduce a utility class, such as ArchiveDataHandler.java, to encapsulate loading, parsing, synchronization, and contiguous checks. This keeps the code modular and testable, with Simulator.java calling methods from this new class.

## Dependencies and Potential Impacts
- **Dependencies:** Include HTTP client libraries (e.g., java.net.http.HttpClient) and JSON/CSV parsing libraries (e.g., Jackson or OpenCSV) in pom.xml to handle data fetching and parsing.
- **Potential Impacts:** Network dependencies could introduce latency or failures, so add robust error handling. Synchronization changes might affect timing in concurrent executions, potentially requiring updates to locks in Simulator.java. The ingest step could add processing overhead, so optimize for performance, and ensure it doesn't disrupt existing agent decision logic.

## Mermaid Diagram for Plan Visualization
```mermaid
graph TD
    A[Start] --> B[Fetch Archive Data via HTTP]
    B --> C[Parse Toplevel Index of Pairs]
    C --> D[Synchronize Initial Timestamps]
    D --> E[Ingest and Check Contiguous Timestamps]
    E --> F[Update Simulator with Data]
    F --> G[Run Simulation in Swimlanes]
    G --> H[End]

    subgraph Data Preparation
        B --> C
        C --> D
        D --> E
    end

    subgraph Simulation Execution
        E --> F
        F --> G
        G --> H
    end