package com.moneyfan.dsel.dsel;

import com.moneyfan.dsel.TupFrame;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.function.Function;
import java.util.stream.Stream;
import java.util.stream.Collectors; // Added for .collect(Collectors.toList())

public class CsvUtil {

    /**
     * Loads data from a CSV file and transforms it into a TupFrame.
     *
     * @param filePath Path to the CSV file.
     * @param rowParser A function that takes a String array (parsed CSV row) and returns a Join<F,S>.
     * @param skipHeader True if the first line of the CSV is a header and should be skipped.
     * @param <F> Type of the first element in the resulting Join records.
     * @param <S> Type of the second element in the resulting Join records.
     * @return A TupFrame containing the parsed data.
     * @throws IOException If an I/O error occurs.
     */
    public static <F, S> TupFrame<F, S> load(String filePath, Function<String[], Join<F,S>> rowParser, boolean skipHeader) throws IOException {
        Stream<String> lines;
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            // Read all lines into a temporary list to make it a restartable stream for TupFrame
            // In a more advanced implementation, we might handle non-restartable streams differently
            // or TupFrame would consume the stream directly.
            lines = reader.lines().collect(Collectors.toList()).stream(); // Use Collectors.toList()
        }

        if (skipHeader) {
            lines = lines.skip(1);
        }

        Stream<Join<F, S>> joinStream = lines
            .map(line -> line.split(",")) // Simple CSV split, not handling quotes or escaped commas
            .map(rowParser);

        // The TupFrame constructor expects a Stream, but for practical use,
        // it's often better if it can operate on a collection or a re-streamable source.
        // Here, we collect to list first as TupFrame.of expects a List.
        // A more direct fromStream(joinStream) in TupFrame would be ideal if it manages stream consumption.
        // For simplicity and current TupFrame.of, we collect.
        return TupFrame.of(joinStream.collect(Collectors.toList()));
    }
}
