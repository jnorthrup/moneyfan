#!/bin/bash

# This script removes old Java source files after package refactoring.
# It should be run from the root of your 'moneyfan' project.

echo "Deleting old Java source files..."

# Old bbcursive paths
rm -f bbcursive/src/main/java/com/bbcursive/core/ParseResult.java
rm -f bbcursive/src/main/java/com/example/bbcursive/BBAtom.java
rm -f bbcursive/src/main/java/com/example/bbcursive/core/Cursive.java
rm -f bbcursive/src/main/java/com/yourdomain/bbcursive/BBAtom.java
rm -f bbcursive/src/main/java/com/yourdomain/bbcursive/core/ParseResult.java

# Old bikeshed paths
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/core/Join.java
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/core/Series.java
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/dsel/Cursor.java
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/dsel/RowVec.java
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/dsel/Series.java
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/isam/RecordMeta.java
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/trading/AgentInterface.java
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/trading/TickData.java
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/trading/WalletState.java
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/types/ColumnMeta.java
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/types/TypeMemento.java
rm -f bikeshed/src/main/java/com/vsiwest/bikeshed/util/Constants.java

# Remove empty directories if they exist
find bbcursive/src/main/java/com/bbcursive -type d -empty -delete
find bbcursive/src/main/java/com/example -type d -empty -delete
find bbcursive/src/main/java/com/yourdomain -type d -empty -delete
find bikeshed/src/main/java/com/vsiwest/bikeshed -type d -empty -delete

echo "Old files deletion complete. Please manually verify and remove any remaining empty directories if necessary."
