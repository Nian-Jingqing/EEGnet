% Find the lag with maximal cross-correlation between two signals
function final_lag = compute_lag(signalA,signalB)
[c, lags] = xcorr(signalA, signalB); % stem(lags,c)
[~,lag_idx] = max(c); % Find the index of the lag with maximal cross-correlation
final_lag = lags(lag_idx);