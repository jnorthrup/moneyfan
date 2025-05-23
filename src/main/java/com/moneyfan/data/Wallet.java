package com.moneyfan.data;

import com.moneyfan.dsl.row.DataRow;
import com.moneyfan.dsl.typeevidence.BasicSchemaEvidence;
import com.moneyfan.dsl.typeevidence.FieldTypeEvidence;
import com.moneyfan.dsl.typeevidence.SemanticTokenType;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class Wallet {
    private final String walletId;
    private final Map<String, BigDecimal> balances = new HashMap<>();
    private final Map<String, BigDecimal> lockedBalances = new HashMap<>();

    public static final BasicSchemaEvidence BALANCE_SCHEMA = new BasicSchemaEvidence("WalletBalance", List.of(
            FieldTypeEvidence.fte("asset", SemanticTokenType.STRING, String.class),
            FieldTypeEvidence.fte("free", SemanticTokenType.BIG_DECIMAL, BigDecimal.class),
            FieldTypeEvidence.fte("locked", SemanticTokenType.BIG_DECIMAL, BigDecimal.class)
    ));

    private final List<DataRow> currentBalancesTable = new ArrayList<>();

    public Wallet(String walletId) {
        this.walletId = walletId;
    }

    public String getWalletId() {
        return walletId;
    }

    public BigDecimal getFreeBalance(String asset) {
        return balances.getOrDefault(asset, BigDecimal.ZERO);
    }

    public BigDecimal getLockedBalance(String asset) {
        return lockedBalances.getOrDefault(asset, BigDecimal.ZERO);
    }

    public void setFreeBalance(String asset, BigDecimal freeAmount) {
        if (freeAmount.compareTo(BigDecimal.ZERO) < 0) {
            balances.put(asset, BigDecimal.ZERO); // Cannot be negative
        } else {
            balances.put(asset, freeAmount);
        }
        updateBalancesTable(asset);
    }

    public void lockAmount(String asset, BigDecimal amount) {
        BigDecimal currentFree = getFreeBalance(asset);
        BigDecimal currentLocked = getLockedBalance(asset);
        if (currentFree.compareTo(amount) >= 0) {
            balances.put(asset, currentFree.subtract(amount));
            lockedBalances.put(asset, currentLocked.add(amount));
            updateBalancesTable(asset);
        }
    }

    public boolean unlockAmount(String asset, BigDecimal amount) {
        BigDecimal currentLocked = getLockedBalance(asset);
        if (currentLocked.compareTo(amount) >= 0) {
            lockedBalances.put(asset, currentLocked.subtract(amount));
            balances.put(asset, getFreeBalance(asset).add(amount));
            updateBalancesTable(asset);
            return true;
        }
        return false;
    }

    private void updateBalancesTable(String asset) {
        // Simplified update logic; in a real scenario, this would integrate with Table
        DataRow newRow = new DataRow(BALANCE_SCHEMA, new Object[]{asset, getFreeBalance(asset), getLockedBalance(asset)});
        // Add or update in currentBalancesTable (implementation depends on Table class)
        currentBalancesTable.add(newRow); // Example; replace with actual logic
    }

    public List<DataRow> getCurrentBalancesTable() {
        return Collections.unmodifiableList(currentBalancesTable);
    }
}
