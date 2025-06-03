package com.vsiwest.moneyfan.ingestion;

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.HttpEntity;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.HttpClientResponseHandler; // Added for handling response

import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class BinanceCsvDownloader {

    public void downloadFile(String url, String destinationPath) throws IOException {
        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
            HttpGet httpGet = new HttpGet(url);

            // Using a response handler to simplify resource management
            HttpClientResponseHandler<byte[]> responseHandler = response -> {
                int status = response.getCode();
                if (status >= 200 && status < 300) {
                    HttpEntity entity = response.getEntity();
                    if (entity != null) {
                        return EntityUtils.toByteArray(entity);
                    }
                    return null;
                } else {
                    throw new IOException("Unexpected response status: " + status);
                }
            };

            byte[] fileBytes = httpClient.execute(httpGet, responseHandler);

            if (fileBytes != null) {
                try (FileOutputStream fos = new FileOutputStream(destinationPath)) {
                    fos.write(fileBytes);
                }
            } else {
                // This case might occur if the entity was null but response was 2xx
                // Or could be handled by response handler throwing an exception earlier
                System.err.println("No content found to download from " + url);
            }
        }
    }

    public static void main(String[] args) {
        BinanceCsvDownloader downloader = new BinanceCsvDownloader();
        String testUrl = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv";
        String tempPath = "iris_test_download.csv";

        try {
            System.out.println("Attempting to download from: " + testUrl);
            downloader.downloadFile(testUrl, tempPath);
            if (Files.exists(Paths.get(tempPath)) && Files.size(Paths.get(tempPath)) > 0) {
                System.out.println("File downloaded successfully to: " + tempPath);
                // Optional: Clean up the downloaded file
                // Files.delete(Paths.get(tempPath));
            } else {
                System.err.println("File download failed or file is empty.");
            }
        } catch (IOException e) {
            System.err.println("Error during download: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
