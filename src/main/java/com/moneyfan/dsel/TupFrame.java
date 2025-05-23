package com.moneyfan.dsel;

import com.moneyfan.dsel.dsel.CsvUtil;
import com.moneyfan.dsel.dsel.Join;

import java.io.IOException;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;
import java.util.Iterator;

/**
 * TupFrame: A DSEL for 2-ary tuple record processing.
 * Represents a sequence of Join records, designed for lazy operations.
 * @param <F> type of the first element in Joins
 * @param <S> type of the second element in Joins
 */
public class TupFrame<F, S> implements Iterable<Join<F, S>> {

    private final Stream<Join<F, S>> stream;

    /**
     * Private constructor to enforce use of static factory methods.
     * @param stream The underlying stream of Join records.
     */
    private TupFrame(Stream<Join<F, S>> stream) {
        this.stream = stream;
    }

    /**
     * Creates a TupFrame from a list of Join records.
     * @param data The list of Join records.
     * @param <F> type of the first element
     * @param <S> type of the second element
     * @return A new TupFrame.
     */
    public static <F, S> TupFrame<F, S> of(List<Join<F, S>> data) {
        return new TupFrame<>(data.stream());
    }

    /**
     * Creates a TupFrame by loading data from a CSV file using a provided parser.
     * This is a factory method that would typically use a CsvUtil or similar.
     */
    public static <F, S> TupFrame<F, S> loadFromCsv(String filePath, Function<String[], Join<F,S>> rowParser, boolean skipHeader) throws IOException {
        return CsvUtil.load ( filePath, rowParser, skipHeader);
    }

    /**
     * Creates a TupFrame from an existing Stream of Join records.
     * Used internally for chaining operations.
     * @param stream The stream of Join records.
     * @param <F> type of the first element
     * @param <S> type of the second element
     * @return A new TupFrame.
     */
    public static <F, S> TupFrame<F, S> fromStream(Stream<Join<F, S>> stream) {
        return new TupFrame<>(stream);
    }

    /**
     * Applies a mapping function to each Join record in the TupFrame, creating a new TupFrame
     * with transformed records.
     * @param mapper A function to apply to each Join record.
     * @param <F2> type of the new first element
     * @param <S2> type of the new second element
     * @return A new TupFrame with mapped records.
     */
    public <F2, S2> TupFrame<F2, S2> map(Function<? super Join<F, S>, ? extends Join<F2, S2>> mapper) {
        return new TupFrame<>(stream.map(mapper));
    }

    /**
     * Filters the Join records in the TupFrame based on a predicate, creating a new TupFrame
     * containing only the records that satisfy the predicate.
     * @param predicate A predicate to apply to each Join record.
     * @return A new TupFrame with filtered records.
     */
    public TupFrame<F, S> filter(Predicate<? super Join<F, S>> predicate) {
        return new TupFrame<>(stream.filter(predicate));
    }

    /**
     * Performs a terminal operation, consuming the TupFrame and applying a consumer to each record.
     * @param consumer A consumer to apply to each Join record.
     */
    public void forEach(Consumer<? super Join<F, S>> consumer) {
        stream.forEach(consumer);
    }

    /**
     * Returns an iterator over elements of type {@code Join<F, S>}.
     * This allows TupFrame to be used in enhanced for-loops.
     * Note: Streams are generally single-use. Iterating multiple times on the same TupFrame
     * instance might lead to unexpected behavior if the underlying stream is consumed.
     * For multiple iterations, recreate the TupFrame from its source.
     * @return an Iterator.
     */
    @Override
    public Iterator<Join<F, S>> iterator() {
        return stream.iterator();
    }
}
