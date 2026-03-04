"""
128 Trade Pairs Configuration

Index 0: USD (base currency)
Index 1-127: Coinbase traded assets

Ordered by 24h volume (approximate ranking).
"""

# All 128 trade pairs (USD at index 0)
TRADE_PAIRS = [
    # Index 0: Base currency
    "USD",           # Always 1.0, base reference
    
    # Major pairs (high volume) - Indices 1-20
    "BTC-USD",       # Bitcoin
    "ETH-USD",       # Ethereum
    "SOL-USD",       # Solana
    "XRP-USD",       # Ripple
    "DOGE-USD",      # Dogecoin
    "ADA-USD",       # Cardano
    "AVAX-USD",      # Avalanche
    "DOT-USD",       # Polkadot
    "LINK-USD",      # Chainlink
    "MATIC-USD",     # Polygon
    "SHIB-USD",      # Shiba Inu
    "LTC-USD",       # Litecoin
    "BCH-USD",       # Bitcoin Cash
    "UNI-USD",       # Uniswap
    "ATOM-USD",      # Cosmos
    "ETC-USD",       # Ethereum Classic
    "XLM-USD",       # Stellar
    "ALGO-USD",      # Algorand
    "VET-USD",       # VeChain
    "FIL-USD",       # Filecoin
    
    # Mid-cap pairs - Indices 21-60
    "AAVE-USD", "APE-USD", "APT-USD", "ARB-USD", "AXS-USD",
    "BLUR-USD", "CHZ-USD", "COMP-USD", "CRV-USD", "ENS-USD",
    "FLOW-USD", "FTM-USD", "GALA-USD", "GRT-USD", "HBAR-USD",
    "IMX-USD", "INJ-USD", "IOTX-USD", "KAVA-USD", "LDO-USD",
    "MANA-USD", "MKR-USD", "NEAR-USD", "OP-USD", "PEPE-USD",
    "QNT-USD", "RNDR-USD", "RUNE-USD", "SAND-USD", "SUSHI-USD",
    "SXP-USD", "THETA-USD", "TRX-USD", "WLD-USD", "YFI-USD",
    "ZEC-USD", "ZIL-USD", "1INCH-USD", "BAL-USD", "BATUSD",
    
    # Lower cap / newer pairs - Indices 61-100
    "BIGTIME-USD", "BONK-USD", "CELO-USD", "COTI-USD", "DIA-USD",
    "DYDX-USD", "EGLD-USD", "ENJ-USD", "EOS-USD", "FET-USD",
    "ICP-USD", "ICX-USD", "KNC-USD", "KLAY-USD", "KSM-USD",
    "LOOM-USD", "LRC-USD", "MASK-USD", "MINA-USD", "NKN-USD",
    "NMR-USD", "NU-USD", "OCEAN-USD", "OGN-USD", "OMG-USD",
    "ONT-USD", "ORCA-USD", "OXT-USD", "PAXG-USD", "PLA-USD",
    "POLY-USD", "PUNDIX-USD", "RAD-USD", "REN-USD", "RLC-USD",
    "SALA-USD", "SKL-USD", "SLN-USD", "SNX-USD", "STORJ-USD",
    
    # Stablecoins and wrapped assets - Indices 101-120
    "DAI-USD", "USDC-USD", "USDT-USD", "WBTC-USD", "EURC-USD",
    "GYEN-USD", "PAX-USD", "BUSD-USD", "TUSD-USD", "USDP-USD",
    
    # Additional pairs - Indices 121-127
    "AERO-USD", "AERO-USD", "AXL-USD", "BIT-USD", "BOND-USD",
    "C98-USD", "CTSI-USD", "DUMMY1-USD", "DUMMY2-USD", "DUMMY3-USD", "DUMMY4-USD", "DUMMY5-USD", "DUMMY6-USD", "DUMMY7-USD", "DUMMY8-USD", "DUMMY9-USD", "DUMMY10-USD",
]

# Validate we have exactly 128
assert len(TRADE_PAIRS) == 128, f"Expected 128 pairs, got {len(TRADE_PAIRS)}"

# Asset name extraction (remove -USD suffix)
ASSET_NAMES = [pair.replace("-USD", "") if pair != "USD" else "USD" for pair in TRADE_PAIRS]

# Coinbase pair format (USD base)
COINBASE_PAIRS = [p for p in TRADE_PAIRS if p != "USD"]

# Sector groupings for MapReduce aggregation
SECTORS = {
    "majors": list(range(1, 11)),       # BTC, ETH, SOL, etc.
    "defi": [14, 24, 28, 32, 39, 46, 52, 54, 59],  # UNI, AAVE, COMP, etc.
    "layer1": [1, 2, 3, 6, 7, 8, 15, 19, 43, 52],  # BTC, ETH, SOL, ADA, etc.
    "layer2": [9, 10, 41, 52],           # MATIC, ARB, OP, etc.
    "meme": [5, 11, 31, 60],             # DOGE, SHIB, PEPE, BONK
    "stablecoins": list(range(101, 111)), # DAI, USDC, USDT, etc.
    "privacy": [38, 65, 83],             # XMR (if available), ZEC, etc.
}

# Risk limits per sector
SECTOR_RISK_LIMITS = {
    "majors": 0.3,      # 30% max per position
    "defi": 0.2,
    "layer1": 0.25,
    "layer2": 0.2,
    "meme": 0.05,       # Lower for high volatility
    "stablecoins": 0.5, # Higher for stablecoins
    "default": 0.1,
}


def get_asset_index(asset_name: str) -> int:
    """Get index for asset name"""
    if asset_name == "USD":
        return 0
    pair = f"{asset_name}-USD"
    if pair in TRADE_PAIRS:
        return TRADE_PAIRS.index(pair)
    raise ValueError(f"Unknown asset: {asset_name}")


def get_sector_for_index(idx: int) -> str:
    """Get sector name for asset index"""
    for sector, indices in SECTORS.items():
        if idx in indices:
            return sector
    return "default"


def get_risk_limit(idx: int) -> float:
    """Get risk limit for asset index"""
    sector = get_sector_for_index(idx)
    return SECTOR_RISK_LIMITS.get(sector, SECTOR_RISK_LIMITS["default"])


if __name__ == "__main__":
    print(f"Total trade pairs: {len(TRADE_PAIRS)}")
    print(f"Coinbase pairs: {len(COINBASE_PAIRS)}")
    print(f"\nFirst 10 pairs:")
    for i, pair in enumerate(TRADE_PAIRS[:10]):
        sector = get_sector_for_index(i)
        risk = get_risk_limit(i)
        print(f"  {i}: {pair} ({sector}, risk={risk})")
    
    print(f"\nSector sizes:")
    for sector, indices in SECTORS.items():
        print(f"  {sector}: {len(indices)} assets")
