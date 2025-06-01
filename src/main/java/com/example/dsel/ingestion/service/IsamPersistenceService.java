package com.example.dsel.ingestion.service;

import com.example.dsel.ingestion.config.AppConfig;
import borg.trikeshed.cursor.Cursor;
import borg.trikeshed.cursor.RowVec;
import borg.trikeshed.isam.RecordMeta; // User's existing Kotlin type
import borg.trikeshed.nio.IOMemento; // Changed from isam.meta to nio
import borg.trikeshed.lib.Series; // For iterating cursor
import borg.trikeshed.lib.Join; // For accessing elements in RowVec
import borg.trikeshed.nio.ByteFieldSerializer; // Added
import borg.trikeshed.nio.LongFieldSerializer; // Added
import borg.trikeshed.nio.DoubleFieldSerializer; // Added
import borg.trikeshed.nio.IntegerFieldSerializer; // Added
import borg.trikeshed.nio.BooleanFieldSerializer; // Added
import borg.trikeshed.nio.StringFieldSerializer; // Added

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap; // For varChars if used
import java.util.function.Supplier;

// Removed local ByteFieldSerializer interface and its implementations (LongFieldSerializer, DoubleFieldSerializer, etc.)
// They are now imported from borg.trikeshed.nio.*

class RecordRowSerializer { // Renamed from RecordSerializer to avoid conflict if it's a common name
    private final List<AugmentedRecordMeta> augmentedSchema;
    private final Map<IOMemento, ByteFieldSerializer<?>> serializers;

    // Helper class to hold calculated offsets and lengths
    static class AugmentedRecordMeta {
        final RecordMeta originalMeta;
        final int offsetInRecord;
        final int fieldLength;

        AugmentedRecordMeta(RecordMeta originalMeta, int offsetInRecord, int fieldLength) {
            this.originalMeta = originalMeta;
            this.offsetInRecord = offsetInRecord;
            this.fieldLength = fieldLength;
        }
    }

    public RecordRowSerializer(List<AugmentedRecordMeta> augmentedSchema) {
        this.augmentedSchema = augmentedSchema;
        this.serializers = new HashMap<>();
        serializers.put(IOMemento.IoLong, new LongFieldSerializer());
        serializers.put(IOMemento.IoDouble, new DoubleFieldSerializer());
        serializers.put(IOMemento.IoInt, new IntegerFieldSerializer());
        serializers.put(IOMemento.IoBoolean, new BooleanFieldSerializer());
        serializers.put(IOMemento.IoString, new StringFieldSerializer());
        // Add other types as needed (e.g., IoFloat, IoShort, IoByte, IoInstant etc.)
    }

    @SuppressWarnings("unchecked")
    public void serializeRecord(ByteBuffer recordBuffer, RowVec row) {
        recordBuffer.clear(); // Prepare buffer for writing one record
        for (int i = 0; i < augmentedSchema.size(); i++) {
            AugmentedRecordMeta augMeta = augmentedSchema.get(i);
            Object value = ((Join<Object, Supplier<RecordMeta>>) row.get(i)).fst();
            
            @SuppressWarnings("rawtypes")
            ByteFieldSerializer serializer = serializers.get(augMeta.originalMeta.type());
            if (serializer != null) {
                serializer.serialize(recordBuffer, augMeta.offsetInRecord, value, augMeta.fieldLength);
            } else {
                System.err.println("Warning: No serializer for IOMemento type: " + augMeta.originalMeta.type() + " for field " + augMeta.originalMeta.name());
                // Handle unsupported types, e.g., by writing zeros or throwing exception
                // Position buffer to write zeros for the field length
                recordBuffer.position(augMeta.offsetInRecord);
                for(int k=0; k < augMeta.fieldLength; ++k) {
                    recordBuffer.put((byte)0);
                }
            }
        }
        recordBuffer.flip(); // Prepare for writing to channel
    }
}


public class IsamPersistenceService {

    private final AppConfig appConfig;
    private final String basePath;
    private static final int DEFAULT_STRING_LENGTH = 50; // Default fixed length for strings

    public IsamPersistenceService(AppConfig appConfig) {
        this.appConfig = appConfig;
        String rawBasePath = appConfig.getMpImportBasePath();
        if (rawBasePath.startsWith("~" + java.io.File.separator) || rawBasePath.equals("~")) {
            this.basePath = System.getProperty("user.home") + rawBasePath.substring(1);
        } else if (rawBasePath.startsWith("~")) {
            this.basePath = System.getProperty("user.home");
        } else {
            this.basePath = rawBasePath;
        }
    }

    private Path getDataFilePath(String assetPair, String interval, String dataType) {
        String sanitizedAssetPair = assetPair.replace("/", "");
        String dataFileName = dataType + ".isam"; // Changed extension
        return Paths.get(basePath, dataType, interval, sanitizedAssetPair, dataFileName);
    }

    private Path getMetaFilePath(String assetPair, String interval, String dataType) {
        String sanitizedAssetPair = assetPair.replace("/", "");
        String metaFileName = dataType + ".meta";
        return Paths.get(basePath, dataType, interval, sanitizedAssetPair, metaFileName);
    }

    // Calculate offsets and lengths for the schema
    private List<RecordRowSerializer.AugmentedRecordMeta> prepareAugmentedSchema(List<RecordMeta> schema, Map<String, Integer> varCharLengths) {
        List<RecordRowSerializer.AugmentedRecordMeta> augmented = new ArrayList<>();
        int currentOffset = 0;
        for (RecordMeta meta : schema) {
            int fieldLength;
            Integer networkSize = meta.type().networkSize(); // Changed getNetworkSize() to networkSize()
            if (networkSize != null) {
                fieldLength = networkSize;
            } else if (meta.type() == IOMemento.IoString) { // Assuming IoString has null networkSize
                fieldLength = varCharLengths.getOrDefault(meta.name(), DEFAULT_STRING_LENGTH);
            } else {
                System.err.println("Warning: Variable size type " + meta.type() + " for field " + meta.name() + " has no defined length. Using default: " + DEFAULT_STRING_LENGTH);
                fieldLength = DEFAULT_STRING_LENGTH; // Fallback for other var types like IoByteArray
            }
            augmented.add(new RecordRowSerializer.AugmentedRecordMeta(meta, currentOffset, fieldLength));
            currentOffset += fieldLength;
        }
        return augmented;
    }

    private int calculateRecordLength(List<RecordRowSerializer.AugmentedRecordMeta> augmentedSchema) {
        if (augmentedSchema.isEmpty()) return 0;
        RecordRowSerializer.AugmentedRecordMeta lastField = augmentedSchema.get(augmentedSchema.size() - 1);
        return lastField.offsetInRecord + lastField.fieldLength;
    }

    public void saveCursor(Cursor dataCursor, String assetPair, String interval, String dataType, List<RecordMeta> schema) {
        Path dataFilePath = getDataFilePath(assetPair, interval, dataType);
        Path metaFilePath = getMetaFilePath(assetPair, interval, dataType);

        System.out.println("IsamPersistenceService (Pure Java NIO): Saving data for " + assetPair + "/" + interval + "/" + dataType);
        System.out.println("Meta file: " + metaFilePath.toString());
        System.out.println("Data file: " + dataFilePath.toString());

        try {
            Files.createDirectories(dataFilePath.getParent());
            Files.createDirectories(metaFilePath.getParent());

            Map<String, Integer> varCharLengths = new HashMap<>(); 
            // Populate varCharLengths for IoString fields if specific lengths are needed,
            // otherwise DEFAULT_STRING_LENGTH will be used by prepareAugmentedSchema.
            // Example:
            // for (RecordMeta meta : schema) {
            //     if (meta.type() == IOMemento.IoString) {
            //         varCharLengths.put(meta.name(), YOUR_CONFIGURED_LENGTH_FOR_THIS_FIELD); 
            //     }
            // }


            List<RecordRowSerializer.AugmentedRecordMeta> augmentedSchema = prepareAugmentedSchema(schema, varCharLengths);
            int recordLength = calculateRecordLength(augmentedSchema);
            if (recordLength == 0 && dataCursor.size() > 0) {
                 System.err.println("Error: Record length is 0 but cursor has data. Aborting ISAM write for " + assetPair);
                 return;
            }


            // 1. Write .meta file
            try (BufferedWriter writer = Files.newBufferedWriter(metaFilePath, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
                StringBuilder coords = new StringBuilder();
                StringBuilder names = new StringBuilder();
                StringBuilder types = new StringBuilder();
                // int currentFileOffset = 0; // Meta file format is offset in record, not file
                for (RecordRowSerializer.AugmentedRecordMeta augMeta : augmentedSchema) {
                    coords.append(augMeta.offsetInRecord).append(" ").append(augMeta.offsetInRecord + augMeta.fieldLength).append(" ");
                    names.append(augMeta.originalMeta.name()).append(" ");
                    types.append(augMeta.originalMeta.type().name()).append(" ");
                    // currentFileOffset += augMeta.fieldLength; // Not needed for meta structure
                }
                writer.write("# format: coords WS .. EOL names WS .. EOL TypeMememento WS .. [EOL]"); writer.newLine();
                writer.write("# last coord is the recordlen (implicitly)"); writer.newLine();
                writer.write(coords.toString().trim()); writer.newLine();
                writer.write(names.toString().trim()); writer.newLine();
                writer.write(types.toString().trim()); writer.newLine();
                System.out.println("Successfully wrote metadata file: " + metaFilePath);
            }

            // 2. Write data file using NIO mmap
            if (dataCursor.size() == 0) {
                System.out.println("No data to write to ISAM file for " + assetPair);
                if (!Files.exists(dataFilePath)) Files.createFile(dataFilePath); // Create empty data file
                return;
            }
            
            RecordRowSerializer recordSerializer = new RecordRowSerializer(augmentedSchema);
            ByteBuffer recordBuffer = ByteBuffer.allocate(recordLength); 
            recordBuffer.order(ByteOrder.BIG_ENDIAN); 

            try (FileChannel fileChannel = FileChannel.open(dataFilePath, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
                // long totalFileSize = (long)dataCursor.size() * recordLength; // Not needed for direct write
                
                for (int i = 0; i < dataCursor.size(); i++) {
                    RowVec row = dataCursor.get(i);
                    recordSerializer.serializeRecord(recordBuffer, row); 
                    fileChannel.write(recordBuffer); 
                    recordBuffer.clear(); 
                }
                System.out.println("Successfully wrote data file: " + dataFilePath + " with " + dataCursor.size() + " records.");
            }

        } catch (Exception e) {
            System.err.println("Error in IsamPersistenceService (Pure Java NIO) for " + assetPair + "/" + interval + "/" + dataType + ": " + e.getMessage());
            e.printStackTrace();
        }
    }
}
