clear, clc, close all
% define analysis parameters
wlen = 512;                        % window length (recomended to be power of 2)
hop = wlen/4;                       % hop size (recomended to be power of 2)
nfft = 1024;                        % number of fft points (recomended to be power of 2)
% define a common frequency (readings per second)
fs = 10;
% load dataset
dataset_0 = load("Baseline Input Data Reduced Size.mat").X;
y=dataset_0(:,36001);
result=zeros(size(dataset_0,1),513,258);
for index = 1:size(dataset_0,1)
%for index = 1:10

subsample = [dataset_0(index,1+6000*0:6000*1);dataset_0(index,1+6000*1:6000*2);dataset_0(index,1+6000*2:6000*3);dataset_0(index,1+6000*3:6000*4);dataset_0(index,1+6000*4:6000*5);dataset_0(index,1+6000*5:6000*6)];
result_i = zeros(513,258);
for j = 1:6
            x = subsample(j,:);
            
            % perform STFT
            win = blackman(wlen, 'periodic');
            [S, ~, ~] = stft(x, win, hop, nfft, fs);
            % calculate the coherent amplification of the window
            C = sum(win)/wlen;
            
            % take the amplitude of fft(x) and scale it, so not to be a
            % function of the length of the window and its coherent amplification
            S = abs(S)/wlen/C;
            
            % correction of the DC & Nyquist component
            if rem(nfft, 2)                     % odd nfft excludes Nyquist point
                S(2:end, :) = S(2:end, :).*2;
            else                                % even nfft includes Nyquist point
                S(2:end-1, :) = S(2:end-1, :).*2;
            end
            result_i(:,1+43*(j-1):43+43*(j-1))=S;
            
end
        result(index,:,:)=result_i;
        
end
random_index = randperm(size(dataset_0,1));
result = result(random_index,:,:);
y = y(random_index);
save('stft_input_2.mat','result','y')