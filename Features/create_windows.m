function [start_idx, end_idx] = create_windows(total_length, window_step, window_length)

start_idx = 1:window_step:total_length; 
end_idx = start_idx + window_length - 1;

while end_idx(end) > total_length
    start_idx=start_idx(1:end-1);
    end_idx=end_idx(1:end-1);
end 