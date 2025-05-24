# MoneyFan

MoneyFan is a Java-based application designed for trading simulations and data analysis. This project provides tools for processing market data, running trading simulations with various agents, and managing data storage with custom ISAM structures.

## Installation

To get started with MoneyFan, ensure you have Java and Maven installed on your system. Follow these steps to build and run the project:

1. Clone the repository to your local machine.
2. Navigate to the project directory:
   ```bash
   cd moneyfan
   ```
3. Build the project using Maven:
   ```bash
   mvn clean install
   ```
4. Run the main application:
   ```bash
   mvn exec:java -Dexec.mainClass="com.moneyfan.Main"
   ```

## Usage

MoneyFan provides several key components for trading simulations and data processing:

- **Trading Agents**: Implementations like `RandomTradingAgent` and `BaseTradingAgent` in `src/main/java/com/moneyfan/agent` and `src/main/java/com/moneyfan/simulator` for simulating trading strategies.
- **Data Processing**: Classes such as `BinanceCsvToIsamConverter` and `CandleProvider` in `src/main/java/com/moneyfan/data` for converting and managing market data.
- **Simulator**: The `Simulator` class in `src/main/java/com/moneyfan/simulator` orchestrates the trading simulation environment with swimlanes for different agent types.
- **DSEL (Data Selection Language)**: Located in `src/main/java/com/moneyfan/dsel`, offers tools for data querying and manipulation, including cursors and joins (jn).

For detailed usage of specific components, refer to the source code documentation and test cases in `src/test/java`.

## Contributing

Contributions to MoneyFan are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue on the GitHub repository.



this java project is enshrined in implementation faithfully to the Trikeshed legacy of type aliases:x

./src/jvmTest/kotlin/gk/kademlia/codec/SmMsgPackTest.kt://typealias ReifiedMessage = Pair<List<Pair<String, String>>, String>
./src/commonMain/kotlin/borg/trikeshed/cursor/ColMeta.kt:typealias ColumnMeta = Join<String, TypeMemento>
./src/commonMain/kotlin/borg/trikeshed/cursor/Cursor.kt:typealias RowVec = Series2<Any?, () -> ColumnMeta>
./src/commonMain/kotlin/borg/trikeshed/cursor/Cursor.kt:typealias Cursor = Series<RowVec>
./src/commonMain/kotlin/borg/trikeshed/tilting/sorting/SortChooser.kt:typealias FlatFileRow = RowVec
./src/commonMain/kotlin/borg/trikeshed/tilting/sorting/SortChooser.kt:typealias Pair<F,S> = Join<F, S>
./src/commonMain/kotlin/borg/trikeshed/parse/json/Json.kt:typealias JsElement = Join<Twin<Int>, Series<Int>> //(openIdx j closeIdx) j commaIdxs
./src/commonMain/kotlin/borg/trikeshed/parse/json/Json.kt:typealias JsIndex = Join<Twin<Int>, Series<Char>> //(twin j src)
./src/commonMain/kotlin/borg/trikeshed/parse/json/Json.kt:typealias JsContext = Join<JsElement, Series<Char>>
./src/commonMain/kotlin/borg/trikeshed/parse/json/Json.kt:typealias JsPathElement = Either<String, Int>
./src/commonMain/kotlin/borg/trikeshed/parse/json/Json.kt:typealias JsPath = Series<JsPathElement>
./src/commonMain/kotlin/borg/trikeshed/parse/DelimitRange.kt:typealias DelimitRange = Twin<UShort> //beginInclusive, endInclusive
./src/commonMain/kotlin/borg/trikeshed/io/PosixFile.kt:typealias PosixOffset = Long
./src/commonMain/kotlin/borg/trikeshed/io/PosixFile.kt:typealias PosixStat = Any
./src/commonMain/kotlin/borg/trikeshed/common/collections/CircularQueue.kt:typealias CircularQueue<T> = CirQlar<T>
./src/commonMain/kotlin/borg/trikeshed/common/collections/HashSeriesSet.kt:typealias Bucket<T> = Series<T>
./src/commonMain/kotlin/borg/trikeshed/acapulco/ObservationBuilder.kt:typealias AgentObservation = Series<RowVec>  // Adjust RowVec as needed based on existing code
./src/commonMain/kotlin/borg/trikeshed/acapulco/AgentInterface.kt:typealias AgentObservation = borg.trikeshed.common.Series<*>  // Adjust as per actual implementation
./src/commonMain/kotlin/borg/trikeshed/acapulco/AgentInterface.kt:typealias AgentAction = DoubleArray  // Represents AssetOutput
./src/commonMain/kotlin/borg/trikeshed/lib/LongSeries.kt:typealias LongSeries<T> = Join<Long, (Long) -> T>
./src/commonMain/kotlin/borg/trikeshed/lib/Predicate.kt:typealias Predicate<T> = (self: T) -> Boolean
./src/commonMain/kotlin/borg/trikeshed/lib/Series2.kt:typealias Series2<A, B> = Series<Join<A, B>>
./src/commonMain/kotlin/borg/trikeshed/lib/Join.kt:typealias Twin<T> = Join<T, T>
./src/commonMain/kotlin/borg/trikeshed/lib/ByteSeries.kt:typealias Series<T> = Join<Int, (Int) -> T>
./src/commonMain/kotlin/borg/trikeshed/reactor/PlatformTypes.kt:typealias Interest = Int
./src/commonMain/kotlin/borg/trikeshed/reactor/AsyncReaction.kt:typealias AsyncReaction = Join<Interest, UnaryAsyncReaction>
./src/commonMain/kotlin/gk/kademlia/include/TypeDefs.kt://typealias Address = URI
./src/commonMain/kotlin/gk/kademlia/include/TypeDefs.kt:typealias Address = String
./src/commonMain/kotlin/gk/kademlia/include/TypeDefs.kt:typealias Route<TNum> = Join<NUID<TNum>, Address>
./src/jvmMain/kotlin/borg/trikeshed/acapulco/runtime/SupportUtils.kt:typealias marketArgTuple = Array<out String>
./src/jvmMain/kotlin/vec/macros/Shim.kt:typealias Pai2<A, B> = Join<A, B>
./src/jvmMain/kotlin/vec/macros/Shim.kt:typealias Vect0r<T> = Series<T>
./src/jvmMain/kotlin/vec/macros/Shim.kt:// typealias RowVec = borg.trikeshed.cursor.RowVec // Assuming it's defined in cursor package
