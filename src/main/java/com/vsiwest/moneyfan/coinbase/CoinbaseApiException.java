package com.vsiwest.moneyfan.coinbase;

public class CoinbaseApiException extends Exception {

    public CoinbaseApiException(String message) {
        super(message);
    }

    public CoinbaseApiException(String message, Throwable cause) {
        super(message, cause);
    }
}
