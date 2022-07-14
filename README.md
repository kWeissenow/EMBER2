# EMBER2
![EMBER](https://rostlab.org/~conpred/EMBER_sketch_small.png "EMBER")

Alignment-free structure prediction using protein language models

# Requirements

The prediction pipeline uses Python3 and requires the following modules:

* numpy
* matplotlib
* torch (1.9.0 recommended)
* transformers (4.6.0 recommended)

The adapted [trRosetta](https://github.com/gjoni/trRosetta) folding pipeline additionally requires pyRosetta to be installed.

# Installation

Clone the repository and install the dependencies listed above.

The ProtT5 protein language model will be downloaded automatically on first use.

# Usage

For a FASTA file containing one or more protein sequences and an output directory of your choice, run the pipeline via

`python predict.py -i <FASTA_file> -o <output_directory>`

The ProtT5 model will be downloaded on first use and stored by default in the directory 'ProtT5-XL-U50'. You can change this directory with the `--t5_model` parameter.

You can trade speed with prediction quality by modifying the cropping stride used during inference (default: 16) with the `--stride` parameter (see publication for details).
If you run out of GPU memory and/or want to compute predictions for long protein sequences, you might want to lower the default batch-size of 200 with the `--batch_size` parameter.

You can create a PDB structure from a predicted distogram using the adapted trRosetta folding scripts in the 'folding' directory:

`python trRosetta.py -m 0 -pd 0.05 <distogram_file> <FASTA_file> output.pdb`

Please note that the FASTA file for the folding script should only contain a single sequence corresponding to the distogram.
It is recommended to create multiple decoys with different cutoffs (-pd [0.05, 0.5]) and modes (-m {0,1,2}). Please refer to [trRosetta](https://github.com/gjoni/trRosetta) for additional details on the folding pipeline.

# Predictions for human proteome (<3000)

Predictions for all human proteins smaller than 3000 residues are available at [EMBER2_human](https://github.com/kWeissenow/EMBER2_human).

# Authors
Konstantin Weißenow, Michael Heinzinger, Burkhard Rost

Technical University Munich

# References

      Weissenow, K., Heinzinger, M., Rost, B.
	  Protein language model embeddings for fast, accurate, alignment-free protein structure prediction
      bioRxiv 2021.07.31.454572; doi: https://doi.org/10.1101/2021.07.31.454572
