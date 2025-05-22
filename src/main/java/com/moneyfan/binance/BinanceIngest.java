package com.moneyfan.binance;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.GridCursor;
import com.moneyfan.io.CSVCursorReader;
import com.moneyfan.io.ISAMWriter;

import java.nio.file.Path;
import java.util.List;

/**
 * Utility for ingesting Binance kline CSV into ISAM format.
 */
public final class BinanceIngest {

    private BinanceIngest() {}

    public static void convertCsvToIsam(Path csvPath, Path isamPath) throws Exception {
        List<Scalar> schema = klineSchema();
        GridCursor grid = CSVCursorReader.read(csvPath, schema);
        ISAMWriter.write(grid, isamPath);
    }

    private static List<Scalar> klineSchema() {
        return List.of(
                Scalar.of(IOMemento.IO_LONG, "open_time"),
                Scalar.of(IOMemento.IO_DOUBLE, "open"),
                Scalar.of(IOMemento.IO_DOUBLE, "high"),
                Scalar.of(IOMemento.IO_DOUBLE, "low"),
                Scalar.of(IOMemento.IO_DOUBLE, "close"),
                Scalar.of(IOMemento.IO_DOUBLE, "volume"),
                Scalar.of(IOMemento.IO_LONG, "close_time"),
                Scalar.of(IOMemento.IO_DOUBLE, "quote_asset_volume"),
                Scalar.of(IOMemento.IO_INT, "number_of_trades"),
                Scalar.of(IOMemento.IO_DOUBLE, "taker_buy_base_volume"),
                Scalar.of(IOMemento.IO_DOUBLE, "taker_buy_quote_volume")
        );
    }

    public static Path downloadDailyKlines(String symbol,String interval, java.time.LocalDate date, Path destDir) throws Exception {
        java.time.format.DateTimeFormatter fmt = java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd");
        String dateStr = fmt.format(date);
        String fileName = symbol+"-"+interval+"-"+dateStr+".zip";
        String url = "https://data.binance.vision/data/spot/daily/klines/"+symbol+"/"+interval+"/"+fileName;
        java.net.http.HttpClient client = java.net.http.HttpClient.newHttpClient();
        java.net.http.HttpRequest request = java.net.http.HttpRequest.newBuilder(java.net.URI.create(url)).build();
        java.net.http.HttpResponse<byte[]> resp = client.send(request, java.net.http.HttpResponse.BodyHandlers.ofByteArray());
        if(resp.statusCode()!=200) throw new RuntimeException("Failed to download: "+url+" status="+resp.statusCode());
        Path zipPath = destDir.resolve(fileName);
        java.nio.file.Files.write(zipPath, resp.body());
        // unzip
        Path csvOutput = null;
        try (java.util.zip.ZipInputStream zis = new java.util.zip.ZipInputStream(java.nio.file.Files.newInputStream(zipPath))) {
            java.util.zip.ZipEntry entry;
            while((entry=zis.getNextEntry())!=null) {
                if(entry.isDirectory()) continue;
                Path out = destDir.resolve(entry.getName());
                java.nio.file.Files.copy(zis, out, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                csvOutput = out;
            }
        }
        return csvOutput;
    }
}