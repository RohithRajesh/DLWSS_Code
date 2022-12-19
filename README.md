# DLWSS Code
# Hardware–Software Co-Design of Statistical and Deep-Learning Frameworks for Wideband Sensing on Zynq System on Chip

Rohith Rajesh, 
Sumit J Darak, 
Akshay Jain, 
Shivam Chandhok, 
Animesh Sharma


**[Paper Link](https://ieeexplore.ieee.org/abstract/document/9967764)** 
**[Arxiv Link](https://arxiv.org/pdf/2209.02661.pdf)** 


> **Abstract:**
>*With the introduction of spectrum sharing and het- erogeneous services in next-generation networks, the base stations need to sense the wideband spectrum and identify the spectrum resources to meet the quality-of-service, bandwidth, and latency constraints. Sub-Nyquist sampling (SNS) enables digitization for sparse wideband spectrum without needing Nyquist speed analog-to-digital converters. However, SNS demands additional signal processing algorithms for spectrum reconstruction, such as the well-known orthogonal matching pursuit (OMP) algo- rithm. OMP is also widely used in other compressed sensing applications. The first contribution of this work is efficiently mapping the OMP algorithm on the Zynq system-on-chip (ZSoC) consisting of an ARM processor and FPGA. Experimental analysis shows a significant degradation in OMP performance for sparse spectrum. Also, OMP needs prior knowledge of spectrum sparsity. We address these challenges via deep-learning-based architectures and efficiently map them on the ZSoC platform as second contribution. Via hardware-software co-design, different versions of the proposed architecture obtained by partitioning between software (ARM processor) and hardware (FPGA) are considered. The resource, power, and execution time comparisons for given memory constraints and a wide range of word lengths are presented for these architectures.*



# Novel deep learning framework for wideband spectrum characterization at sub-Nyquist rate

Shivam Chandhok, 
Himani Joshi, 
A V Subramanyam,
Sumit J. Darak


**[Paper Link](https://link.springer.com/article/10.1007/s11276-021-02765-1)** 
**[Arxiv Link](https://arxiv.org/pdf/1912.05255)** 


> **Abstract:**
>*Introduction of spectrum-sharing in 5G and sub- sequent generation networks demand base-station(s) with the capability to characterize the wideband spectrum spanned over licensed, shared and unlicensed non-contiguous frequency bands. Spectrum characterization involves the identification of vacant bands along with center frequency and parameters (energy, modulation, etc.) of occupied bands. Such characterization at Nyquist sampling is area and power-hungry due to the need for high-speed digitization. Though sub-Nyquist sampling (SNS) offers an excellent alternative when the spectrum is sparse, it suffers from poor performance at low signal to noise ratio (SNR) and demands careful design and integration of digital recon- struction, tunable channelizer and characterization algorithms. In this paper, we propose a novel deep-learning framework via a single unified pipeline to accomplish two tasks: 1) Reconstruct the signal directly from sub-Nyquist samples, and 2) Wideband spectrum characterization. The proposed approach eliminates the need for complex signal conditioning between reconstruction and characterization and does not need complex tunable channelizers. We extensively compare the performance of our framework for a wide range of modulation schemes, SNR and channel conditions. We show that the proposed framework outperforms existing SNS based approaches and characterization performance approaches to Nyquist sampling-based framework with an increase in SNR. Easy to design and integrate along with a single unified deep learning framework make the proposed architecture a good candidate for reconfigurable platforms.*


