# ASR-SG-HMM-GMM
speech recognition of digits based on single Gaussian, Gaussian Mixture, and Hidden Markov Models.  

Training and test data contain 2,464 and 2,486 utterances respectively.  
Each utterance has a unique id (e.g., ac_1a, ac_1b). After the "[", you can find a 39-dimensional feature vector per line. Each line
corresponds to a feature vector in a consecutive frame, with the final frame in an utterance terminated by "]".  
  
Single-Gaussian-based ASR:
1. Estimate the Gaussian distribution (diagonal covariance) for each digit in the training data by using maximum likelihood estimation.  
2. Compute the log likelihood value for each digit for each utterance in the test data by using the distributions estimated above.  
3. Predict the most likely digit for each utterance by selecting the digit with the largest likelihood.  
4. Compute the accuracy (# of correct digits / # of test utterances (=2486) * 100) and report the accuracy.  
  
GMM-based ASR:  
1. Estimate the Gaussian mixture distribution (diagonal covariance) for each digit by using maximum likelihood estimation.  
2. Initialization: Initialize the mean and variance parameters of each mixture from those of the single-Gaussian-based speech recognition model; Each mixture mean vector should be slightly perturbed randomly according to the standard deviation;
Same as 2,3,4 in Single-Gaussian-based ASR. 

HMM-based ASR:  
1. Estimate an HMM for each digit in the training data, with a single (diagonal covariance) Gaussian distribution per state, by maximum likelihood estimation.  
2. Initialization: Use uniform alignments; Initialize the HMM parameters according to this alignment.  
3. Use Baum-Welch algorithm or Viterbi training algorithm.  
Same as 2,3,4 in Single-Gaussian-based ASR.  

Command:  
python submission.py --mode mode train_1digit.feat test_1digit.feat
mode can be sg, gmm and hmm
--debug can be used before --mode if needed.

