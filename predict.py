##################################
# ProtT5dst
# Inference pipeline for 2D structure prediction using language models
# Authors: Konstantin Weißenow, Michael Heinzinger
##################################

import argparse
from pathlib import Path
from math import log
import numpy as np
import os
import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, T5Tokenizer
import transformers
import matplotlib.pyplot as plt

from model import *

transformers.logging.set_verbosity_error()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))


class EmberDataset(torch.utils.data.Dataset):
    def __init__(self, input, prot_len, offsets):
        self.input = input
        self.prot_len = prot_len
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, item):
        offset_x, offset_y = self.offsets[item]

        cut_lower_x = max(0, offset_x)
        cut_upper_x = min(self.prot_len, offset_x+64)
        cut_lower_y = max(0, offset_y)
        cut_upper_y = min(self.prot_len, offset_y+64)

        crop_lower_x = max(0, -offset_x)
        crop_upper_x = min(64, self.prot_len - offset_x)
        crop_lower_y = max(0, -offset_y)
        crop_upper_y = min(64, self.prot_len - offset_y)

        cropping_info = {'length': self.prot_len}
        cropping_info['cut_lower_x'] = cut_lower_x
        cropping_info['cut_upper_x'] = cut_upper_x
        cropping_info['cut_lower_y'] = cut_lower_y
        cropping_info['cut_upper_y'] = cut_upper_y
        cropping_info['crop_lower_x'] = crop_lower_x
        cropping_info['crop_upper_x'] = crop_upper_x
        cropping_info['crop_lower_y'] = crop_lower_y
        cropping_info['crop_upper_y'] = crop_upper_y

        input_crop = torch.zeros((105, 64, 64), device=device)
        input_crop[:, crop_lower_x:crop_upper_x, crop_lower_y:crop_upper_y] = self.input[:, cut_lower_x:cut_upper_x, cut_lower_y:cut_upper_y]

        return input_crop, cropping_info


def get_T5_model(t5_dir):
    if 'xxl' in t5_dir.name:
        transformer_link = "Rostlab/prot_t5_xxl_uniref50"
    elif 'xl' in t5_dir.name:
        transformer_link = "Rostlab/prot_t5_xl_uniref50"
    else:
        raise NotImplementedError

    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=t5_dir, output_attentions=True)
    model = model.half()
    model = model.to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, cache_dir=t5_dir )
    return model, vocab


def get_dst_model(dst_model_path):
    dst_model = DeepDilated(input_channels_2d=105, input_channels_1d=0, layers=120, out_channels=42)
    dst_model.load_state_dict(torch.load(dst_model_path))
    dst_model = DataParallel(dst_model)
    dst_model = dst_model.to(device)
    dst_model.eval()
    return dst_model


def read_fasta( fasta_path, split_char, id_field):
    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                sequences[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[ uniprot_id ] += ''.join( line.split() ).upper()

    return sequences


def get_embedding(seq, model, vocab):

    # TOP100 attention heads for detecting distances/contacts according to KW'S logistic regression analysis
    num_heads = 100
    contact_heads = [635, 755, 311, 271, 690, 761, 375, 617, 567, 23, 731, 535, 614, 640, 544,
                     335, 303, 678, 727, 15, 508, 239, 767, 732, 343, 28, 751, 604, 747, 495, 399,
                     702, 367, 725, 591, 407, 748, 661, 754, 759, 435, 247, 471, 16, 175, 697, 645,
                     613, 609, 670, 650, 713, 709, 13, 316, 728, 647, 215, 716, 663, 683, 1, 688, 527,
                     606, 764, 402, 597, 612, 124, 721, 143, 736, 564, 698, 24, 383, 599, 98, 207, 742,
                     439, 79, 111, 695, 722, 272, 97, 655, 699, 88, 516, 144, 156, 757, 625, 122, 284, 657, 109]
    #print("Extracting only the following subset of attention heads: {}".format(contact_heads))

    seq = seq.replace('U','X').replace('Z','X').replace('O','X')
    seq_len = len(seq)
    seq = ' '.join(list(seq))

    token_encoding = vocab.batch_encode_plus([seq], add_special_tokens=True, padding="longest")
    input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
    attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

    with torch.no_grad():
        embedding_repr = model(input_ids, attention_mask=attention_mask)

    # (24 x 32 x L x L) 24=n_layer; 32=n_heads
    #try:
    emb = torch.cat( embedding_repr[1], dim=0 )[:,:,:seq_len,:seq_len]
    emb = torch.reshape( emb,(768,seq_len,seq_len) )[contact_heads,:,:]
    #except RuntimeError:
    #    del(emb)
    #    print("RuntimeError during concatenation of heads for {} (L={})".format(identifier, seq_len))
    #    emb = torch.cat( [ attention.to('cpu') for attention in embedding_repr[1]], dim=0 )[:,:,:seq_len,:seq_len]
    #    emb = torch.reshape( emb,(768,seq_len,seq_len) )[contact_heads,:,:]
    #    del(embedding_repr)

    emb = emb.detach()

    # symmetry
    emb = 0.5 * (emb + torch.transpose(emb, 1, 2))

    # APC
    for i in range(num_heads):
        diag = torch.diag(emb[i,:,:])
        Fi = (torch.sum(emb[i,:,:], dim=0) - diag) / seq_len
        Fj = (torch.sum(emb[i,:,:], dim=1) - diag) / seq_len
        F = (torch.sum(emb[i,:,:]) - torch.sum(diag)) / (seq_len*seq_len - seq_len)
        correction = torch.outer(Fi, Fj) / F
        emb[i,:,:] -= correction

    return emb


def distogram_mode_to_distance_classes(distogram):
    return np.argmax(distogram, axis=0)


def distogram_avg_to_distance_classes(distogram):
    length = distogram.shape[1]
    mul = np.swapaxes(np.tile(np.arange(42), (length, length, 1)), 0, 2)
    return (np.sum(distogram * mul, axis=0)).astype(np.int8)


# Maps distance bin numbers (LxL) to distance map (LxL), assuming minimum distance bin is <=2A and steps of 0.5A
def distance_classes_to_distance_map(distance_classes):
    distance_map = distance_classes * 0.5 + 1.75
    np.fill_diagonal(distance_map, 0.0)
    return distance_map


# Maps distogram (BxLxL) to distance map (LxL) using the mode of the probability mass function
def distogram_mode_to_distance_map(distogram):
    distance_classes = distogram_mode_to_distance_classes(distogram)
    return distance_classes_to_distance_map(distance_classes)


# Maps distogram (BxLxL) to distance map (LxL) using the average of the probability mass function
def distogram_avg_to_distance_map(distogram):
    distance_classes = distogram_avg_to_distance_classes(distogram)
    return distance_classes_to_distance_map(distance_classes)


# Computes the standard deviation of distogram probability distributions, assuming minimum distance bin is <=2A and steps of 0.5A
def distogram_stddev(distogram):
    values = np.arange(1.75, 22.26, 0.5)
    values[0] = 0.0
    values = np.expand_dims(values, axis=1)
    values = np.expand_dims(values, axis=1)

    means = np.sum(values * distogram, axis=0)
    e_x = np.sum(values * values * distogram, axis=0)

    var = e_x - (means * means)
    # Avoid negative values due to floating point rounding errors
    var[var < 0] = 0
    return np.sqrt(var)


def create_arg_parser():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description=('t5_embedder.py creates T5 embeddings for a given text file containing sequence(s) in FASTA-format.') )

    parser.add_argument( '-i', '--input', required=True, type=str, help='A path to a fasta-formatted text file containing protein sequence(s).')
    parser.add_argument('--split_char', type=str, default=' ', help="The character for splitting the FASTA header in order to retrieve the protein identifier. Should be used in conjunction with --id. Default: ' '")
    parser.add_argument('--id', type=int, default=0, help="The index for the uniprot identifier field after splitting the FASTA header after each symbole in ['|', '#', ':', ' ']. Default: 0")
    parser.add_argument('-o', '--output', required=True, type=str, help="A path where predictions are stored")

    parser.add_argument('--t5_model', required=True, type=str, help='A path to a directory holding the checkpoint for the pre-trained ProtT5 model' )
    parser.add_argument('--dst_model', required=True, type=str, help='A path to the checkpoint file for the ProtT5dst model')
    parser.add_argument('--stride', type=int, default=16, help="Cropping stride to use for predictions. Smaller values need exponentially more computation time, but might be more accurate. Default: 16")
    parser.add_argument('--batch_size', type=int, default=200, help="Batch size used for inference. Set lower values if you run out of memory. Default: 200")
    parser.add_argument('--workers', type=int, default=0, help="Number of threads used for data loading. Default: 0")

    return parser


def main():
    parser     = create_arg_parser()
    args       = parser.parse_args()

    # Read sequence(s)
    seq_path   = Path( args.input )
    seq_dict = read_fasta(args.input, args.split_char, args.id)

    # Load models
    t5_model, vocab = get_T5_model(Path(args.t5_model))
    dst_model = get_dst_model(Path(args.dst_model))

    softmax = torch.nn.Softmax(dim=1)

    # Create weighting kernel
    x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1.0, 0.0
    kernel = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    kernel = kernel.astype(np.float32)
    kernel = torch.from_numpy(kernel).to(device)

    for idx,identifier in enumerate(seq_dict.keys()):
        seq = seq_dict[identifier]
        prot_len = len(seq)
        print("{} {} ({} residues)".format(idx, identifier, prot_len))

        if prot_len > 3000:
            print("Skipping sequence >3k residues!")
            continue

        ##### Retrieve attention head embedding
        emb = get_embedding(seq_dict[identifier], t5_model, vocab)

        ##### Additional channels
        # Mask
        mask_2d = torch.ones((1, prot_len, prot_len), device=device)
        input = torch.cat((mask_2d, emb))

        # Protein length ( / 1000.0)
        length = torch.full((1, prot_len, prot_len), prot_len / 1000.0, device=device)
        input = torch.cat((input, length))

        # Neff
        neff = torch.full((1, prot_len, prot_len), log(1 / prot_len), device=device)
        input = torch.cat((input, neff))

        # Crop positions
        linspace = np.linspace(0, 1, prot_len)
        x, y = np.meshgrid(linspace, linspace, indexing='ij')
        x = torch.unsqueeze(torch.from_numpy(x).to(device), dim=0)
        y = torch.unsqueeze(torch.from_numpy(y).to(device), dim=0)
        input = torch.cat((input, x))
        input = torch.cat((input, y))

        ##### Compute crops
        offsets = []
        start_offset_x = -32
        start_offset_y = -32
        crops_per_dimension = ((prot_len - 32) + -start_offset_x) // args.stride + 1
        for x in range(crops_per_dimension):
            for y in range(crops_per_dimension):
                offset_x = start_offset_x + x * args.stride
                offset_y = start_offset_y + y * args.stride
                offsets.append((offset_x, offset_y))

        ##### Create dataset and dataloader
        dataset = EmberDataset(input, prot_len, offsets)
        pred_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        ##### Inference
        predictions = torch.zeros((42, prot_len, prot_len), device=device)
        prediction_counts = torch.zeros((1, prot_len, prot_len), device=device)
        for input_crops, cropping_info in pred_loader:
            print(".", end='')
            # Forward pass
            with torch.no_grad():
                prediction, _, _ = dst_model.forward(input_crops.to(device), None)
                # Apply manual softmax to get distance distribution
                probabilities = softmax(prediction.detach())

                for i in range(probabilities.shape[0]):
                    cut_lower_x = cropping_info['cut_lower_x'][i].item()
                    cut_upper_x = cropping_info['cut_upper_x'][i].item()
                    cut_lower_y = cropping_info['cut_lower_y'][i].item()
                    cut_upper_y = cropping_info['cut_upper_y'][i].item()
                    crop_lower_x = cropping_info['crop_lower_x'][i].item()
                    crop_upper_x = cropping_info['crop_upper_x'][i].item()
                    crop_lower_y = cropping_info['crop_lower_y'][i].item()
                    crop_upper_y = cropping_info['crop_upper_y'][i].item()
                    predictions[:, cut_lower_x:cut_upper_x, cut_lower_y:cut_upper_y] += kernel[crop_lower_x:crop_upper_x, crop_lower_y:crop_upper_y] * probabilities[i, :, crop_lower_x:crop_upper_x, crop_lower_y:crop_upper_y]
                    prediction_counts[:, cut_lower_x:cut_upper_x, cut_lower_y:cut_upper_y] += kernel[crop_lower_x:crop_upper_x, crop_lower_y:crop_upper_y]

        print("")

        predictions /= prediction_counts
        # Symmetry
        predictions = (predictions + predictions.permute(0, 2, 1)) * 0.5
        # Convert to numpy
        distogram = predictions.cpu().numpy()

        ##### Save results
        contact_map = np.sum(distogram[:13, :, :], axis=0)
        distance_map_mode = distogram_mode_to_distance_map(distogram)
        distance_map_avg = distogram_avg_to_distance_map(distogram)
        stddev = distogram_stddev(distogram)

        sample_path = os.path.join(args.output, identifier)
        if not os.path.isdir(sample_path):
            os.makedirs(sample_path)

        np.save("{}/distogram.npy".format(sample_path), distogram)
        np.save("{}/distance_map_mode.npy".format(sample_path), distance_map_mode)
        np.save("{}/distance_map_avg.npy".format(sample_path), distance_map_avg)
        np.save("{}/contact_map.npy".format(sample_path), contact_map)
        np.save("{}/stddev.npy".format(sample_path), stddev)

        img = plt.imshow(contact_map, cmap="CMRmap")
        cb = plt.colorbar()
        cb.set_label("Contact probability")
        plt.savefig("{}/contact_map.png".format(sample_path))
        plt.clf()

        img = plt.imshow(distance_map_avg, cmap="hot_r")
        cb = plt.colorbar()
        cb.set_label("Angstrom distance")
        plt.savefig("{}/distance_map_avg.png".format(sample_path))
        plt.clf()

        img = plt.imshow(distance_map_mode, cmap="hot_r")
        cb = plt.colorbar()
        cb.set_label("Angstrom distance")
        plt.savefig("{}/distance_map_mode.png".format(sample_path))
        plt.clf()

if __name__ == '__main__':
    main()
