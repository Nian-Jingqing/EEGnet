% EEGLAB history file generated on the 04-Mar-2020
% ------------------------------------------------

EEG.etc.eeglabvers = '2019.1'; % this tracks which version of EEGLAB is being used, you may ignore it
EEG = eeg_checkset( EEG );
pop_eegplot( EEG, 1, 1, 1);
figure; pop_spectopo(EEG, 1, [0  593970], 'EEG' , 'percent', 15, 'freq', [6 10 22], 'freqrange',[2 25],'electrodes','off');
EEG = pop_select( EEG, 'nochannel',{'POz'});
EEG = eeg_checkset( EEG );
pop_eegplot( EEG, 1, 1, 1);
