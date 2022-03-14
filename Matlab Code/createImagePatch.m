function patchData = createImagePatch(hcube,winSize)

padding = floor((winSize-1)/2);
zeroPaddingPatch = padarray(hcube,[padding,padding],0);
[rows,cols,ch] = size(hcube);
patchData = zeros(rows*cols,winSize,winSize,ch);
zeroPaddedInput = size(zeroPaddingPatch);
patchIdx = 1;
for i= (padding+1):(zeroPaddedInput(1)-padding)
    for j= (padding+1):(zeroPaddedInput(2)-padding)
        patch = zeroPaddingPatch(i-padding:i+padding,j-padding:j+padding,:);
        patchData(patchIdx,:,:,:) = patch;
        patchIdx = patchIdx+1;
    end
end

end