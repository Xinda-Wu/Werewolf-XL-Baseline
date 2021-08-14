1. We deleted some audio samples that were less than 0.98s. Because VGGish network only accept the audio samples that longer than 0.98s. if the audio length of the sample is less than 0.98s, we add 0 to full the empty.
2. We also deleted the same instances when we make a decision level fusion.
3. 

Therefore, the number of audio data will be smaller in the method of "VGGish"