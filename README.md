# EMBER2
Alignment-free structure prediction using protein language models

# Requirements

The prediction pipeline uses Python3 and requires the following modules:

* numpy
* matplotlib
* torch (1.9.0 recommended)
* transformers (4.6.0 recommended)

# Installation

Clone the repository and additionally download and unpack the ProtT5 language model and EMBER2 model snapshots:

https://rostlab.org/~conpred/EMBER2/models/ProtT5_snapshot.tar.bz2 (3.6 GB)

https://rostlab.org/~conpred/EMBER2/models/EMBER2_snapshot.tar.bz2 (24.0 MB)

# Usage

For a FASTA file containing one or more protein sequences and an output directory of your choice (must already exist), run the pipeline via

`python predict.py -i <FASTA_file> -o <output_directory> --t5_model <ProtT5_directory> --dst_model <ProtT5dst_directory>`

You can trade speed with prediction quality by modifying the cropping stride used during inference (default: 16) with the `--stride` parameter (see publication for details).
If you run out of GPU memory and/or want to compute predictions for long protein sequences, you might want to lower the default batch-size of 200 with the `--batch_size` parameter.

# Authors
Konstantin Weißenow, Michael Heinzinger, Burkhard Rost

Technical University Munich

# References

      Weißenow, K., Heinzinger, M., Rost, B.
	  Protein language model embeddings for fast, accurate, alignment-free protein structure prediction
      bioRxiv 2021.07.31.454572; doi: https://doi.org/10.1101/2021.07.31.454572
