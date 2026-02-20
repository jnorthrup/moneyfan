// pricingModels.js - Black-Scholes and related pricing models

// Error function approximation for use in Black-Scholes
function erf(x) {
  // save the sign of x
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);

  // constants
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;

  // A&S formula 7.1.26
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  
  return sign * y;
}

// Standard normal CDF
function nd(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

// Black-Scholes formula for option pricing
function blackScholes(S, K, T, sigma, r, call = true) {
  if (T <= 0) return call ? Math.max(S - K, 0) : Math.max(K - S, 0);
  
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  
  if (call) {
    return S * nd(d1) - K * Math.exp(-r * T) * nd(d2);
  } else {
    return K * Math.exp(-r * T) * nd(-d2) - S * nd(-d1);
  }
}

// Delta calculation for options
function delta(S, K, T, sigma, r, call = true) {
  if (T <= 0) return call ? S > K ? 1 : 0 : S > K ? 0 : -1;
  
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  
  if (call) {
    return nd(d1);
  } else {
    return nd(d1) - 1;
  }
}

module.exports = {
  blackScholes,
  delta,
  erf,
  nd
};