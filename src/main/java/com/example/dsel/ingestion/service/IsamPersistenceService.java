package com.example.dsel.ingestion.service;

import com.example.dsel.ingestion.config.AppConfig;
// DselSchemas is not directly used here, schema is passed as List<RecordMeta>
// import com.example.dsel.ingestion.schema.DselSchemas; 
import borg.trikeshed.cursor.Cursor;
import borg.trikeshed.cursor.RowVec; // For Cursor, which is Series<RowVec> (though RowVec itself isn't directly used here)
import borg.trikeshed.isam.IsamDataFile;
import borg.trikeshed.isam.IsamMetaFileReader;
import borg.trikeshed.isam.RecordMeta; // Java representation compatible with DselSchemas
import borg.trikeshed.lib.Series;
import borg.trikeshed.lib.StdLibs; // Assuming a utility for List to Series conversion
import borg.trikeshed.type.ColumnMeta; // For the Series<ColumnMeta> type in Kotlin sig

import java.nio.file.Files; // Added for Files.createDirectories
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class IsamPersistenceService {

    private final AppConfig appConfig;
    private final String basePath;

    public IsamPersistenceService(AppConfig appConfig) {
        this.appConfig = appConfig;
        String rawBasePath = appConfig.getMpImportBasePath(); // Using mpImportBasePath as per previous AppConfig
        if (rawBasePath.startsWith("~" + java.io.File.separator) || rawBasePath.equals("~")) {
            this.basePath = System.getProperty("user.home") + rawBasePath.substring(1);
        } else if (rawBasePath.startsWith("~")) {
            // Handle cases where there might not be a separator after ~ but it's not just ~
            this.basePath = System.getProperty("user.home");
        } else {
            this.basePath = rawBasePath;
        }
    }

    private Path getDataFilePath(String assetPair, String interval, String dataType) {
        String sanitizedAssetPair = assetPair.replace("/", "");
        // Assuming data file has a .dat extension or similar, adjust if known
        String dataFileName = dataType + ".dat"; 
        return Paths.get(basePath, dataType, interval, sanitizedAssetPair, dataFileName);
    }

    private Path getMetaFilePath(String assetPair, String interval, String dataType) {
        String sanitizedAssetPair = assetPair.replace("/", "");
        String metaFileName = dataType + ".meta"; // As per IsamDataFile default
        return Paths.get(basePath, dataType, interval, sanitizedAssetPair, metaFileName);
    }
    
    // Helper to convert Java List<RecordMeta> to borg.trikeshed.lib.Series<ColumnMeta>
    // RecordMeta is a ColumnMeta, so this is mainly a collection type conversion.
    private Series<ColumnMeta> convertSchemaToSeries(List<RecordMeta> javaSchema) {
        List<ColumnMeta> columnMetaList = javaSchema.stream()
                               .map(rm -> (ColumnMeta)rm) // Explicit cast if needed, though RecordMeta should be a ColumnMeta
                               .collect(Collectors.toList());
        // This relies on borg.trikeshed.lib.StdLibs.toSeries(List) existing and working as expected.
        return StdLibs.toSeries(columnMetaList); 
    }

    public void saveCursor(Cursor dataCursor, String assetPair, String interval, String dataType, List<RecordMeta> schema) {
        Path dataFilePath = getDataFilePath(assetPair, interval, dataType);
        Path metaFilePath = getMetaFilePath(assetPair, interval, dataType);

        System.out.println("IsamPersistenceService: Saving data for " + assetPair + "/" + interval + "/" + dataType);
        System.out.println("Meta file: " + metaFilePath.toString());
        System.out.println("Data file: " + dataFilePath.toString());

        try {
            // Ensure parent directories exist
            Files.createDirectories(dataFilePath.getParent());
            Files.createDirectories(metaFilePath.getParent());

            // 1. Convert schema (List<RecordMeta>) to Series<ColumnMeta> for IsamMetaFileReader.Companion.write
            Series<ColumnMeta> schemaAsSeries = convertSchemaToSeries(schema);
            
            // This map is for specifying lengths of variable-length string types.
            // The ISAM system might require actual lengths for IoString, IoByteArray here.
            Map<String, Integer> varChars = Collections.emptyMap();

            // 2. Write the .meta file using IsamMetaFileReader.Companion.write
            // The prompt provides this exact static method invocation.
            borg.trikeshed.isam.IsamMetaFileReader.Companion.write(metaFilePath.toString(), schemaAsSeries, varChars, null, 0, null);

            System.out.println("Successfully wrote metadata file: " + metaFilePath);

            // 3. Write the data file using IsamDataFile.Companion.write
            // The prompt provides this exact static method invocation.
            borg.trikeshed.isam.IsamDataFile.Companion.write(dataCursor, dataFilePath.toString(), varChars, null, 0, null);

            System.out.println("Successfully wrote data file: " + dataFilePath);

        } catch (Exception e) {
            System.err.println("Error in IsamPersistenceService for " + assetPair + "/" + interval + "/" + dataType + ": " + e.getMessage());
            e.printStackTrace();
            // Consider re-throwing or specific exception handling
        }
    }
}
