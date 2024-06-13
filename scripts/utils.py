import torch
import os
import json
import numpy as np
import pandas as pd
import pyfaidx
import kipoiseq
import functools
from kipoiseq import Interval
import os
import tensorflow as tf
from natsort import natsorted
import glob

import tabix
import pyBigWig
import selene_sdk
from selene_sdk.targets import Target
from selene_sdk.samplers import RandomPositionsSampler
from selene_sdk.samplers.dataloader import SamplerDataLoader

def load_data(root='./DNALongBench/data', task_name = 'regulatory_sequence_activity', organism = 'human', cell_type='HFF', batch_size=16, sequence_length=196608):
    if task_name == 'regulatory_sequence_activity':
        assert organism == "human" or organism == "mouse"
        human_fasta_path = root + 'regulatory_sequence_activity_prediction/human/seqs/hg38.ml.fa'
        mouse_fasta_path = root + 'regulatory_sequence_activity_prediction/mouse/seqs/mm10.ml.fa'
        data_path = root + 'regulatory_sequence_activity_prediction/'
        
        # SEQUENCE_LENGTH = 196608
        # BIN_SIZE = 128
        # TARGET_LENGTH = 896

        fasta_path = human_fasta_path if organism == "human" else mouse_fasta_path

        train_dataset = BasenjiDataSet(data_path, organism, 'train', sequence_length, fasta_path)
        valid_dataset = BasenjiDataSet(data_path, organism, 'valid', sequence_length, fasta_path)
        test_dataset = BasenjiDataSet(data_path, organism, 'test', sequence_length, fasta_path)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
        
    elif task_name == 'contact_map_prediction':
        train_data_path = root + 'contact_map_prediction/targets/train-*.tfr'
        valid_data_path = root + 'contact_map_prediction/targets/valid-*.tfr'
        test_data_path = root + 'contact_map_prediction/targets/test-*.tfr'
        SEQUENCE_LENGTH = 1048576
        TARGET_LENGTH = 99681

        train_dataset = AkitaDataset(train_data_path, cell_type = cell_type, target_length=TARGET_LENGTH)
        valid_dataset = AkitaDataset(valid_data_path, cell_type = cell_type, target_length=TARGET_LENGTH)
        test_dataset = AkitaDataset(test_data_path, cell_type = cell_type, target_length=TARGET_LENGTH)
     
        
        rain_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
        
    elif task_name == 'transcription_initiation_signal_prediction':
        genome, noblacklist_genome = get_genomes(root+"transcription_initiation_signal_prediction/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
        tfeature = GenomicSignalFeatures([root+"transcription_initiation_signal_prediction/targets/agg.plus.bw.bedgraph.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.encodecage.plus.v2.bedgraph.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.encoderampage.plus.v2.bedgraph.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.plus.grocap.bedgraph.sorted.merged.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.plus.allprocap.bedgraph.sorted.merged.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.minus.allprocap.bedgraph.sorted.merged.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.minus.grocap.bedgraph.sorted.merged.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.encoderampage.minus.v2.bedgraph.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.encodecage.minus.v2.bedgraph.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.minus.bw.bedgraph.bw"],
                                       ['cage_plus','encodecage_plus','encoderampage_plus', 'grocap_plus','procap_plus','procap_minus','grocap_minus'
        ,'encoderampage_minus', 'encodecage_minus','cage_minus'],
                                       (100000,),
                                       [root+"transcription_initiation_signal_prediction/targets/blacklists/fantom.blacklist8.plus.bed.gz",root+"transcription_initiation_signal_prediction/targets/blacklists/fantom.blacklist8.minus.bed.gz"],
                                       [0,9], [1,8], [0.61357, 0.61357])

        from selene_sdk.samplers import RandomPositionsSampler
        from selene_sdk.samplers.dataloader import SamplerDataLoader
        
        sampler = RandomPositionsSampler(
                        reference_sequence = genome,
                        target= tfeature,
                        features = [''],
                        test_holdout=['chr8', 'chr9'],
                        validation_holdout= ['chr10'],
                        sequence_length= 100000,
                        center_bin_to_predict= 100000,
                        position_resolution=1,
                        random_shift=0,
                        random_strand=False
        )
        sampler.mode="train"
        train_loader = SamplerDataLoader(sampler, num_workers=1, batch_size=16, seed=3)
        return train_loader, None, None

        
    return train_loader, valid_loader, test_loader

class AkitaDataset(torch.utils.data.IterableDataset):
    def __init__(self, tfr_pattern, cell_type, target_length=99681):
        super(AkitaDataset).__init__()
        self.dataset = self.read_tfr(tfr_pattern)
        self.cell_type = cell_type
        target_ind_dict = {'HFF': 0, 'H1hESC': 1, 'GM12878': 2, 'IMR90': 3, 'HCT116': 4}
        self.target_ind = target_ind_dict[self.cell_type]
        self.target_length=target_length

    
    # Adapted from https://github.com/calico/basenji/blob/498d0b1cd02e1ba11658d273ef257a07f5a69657/bin/tfr_qc.py#L98
    def file_to_records(self, filename):
        return tf.data.TFRecordDataset(filename, compression_type='ZLIB')
    def parse_proto(self, example_protos):
        features = {
            'sequence': tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.string)
          }
        parsed_features = tf.io.parse_example(example_protos, features=features)
        seq = tf.io.decode_raw(parsed_features['sequence'], tf.uint8)
        targets = tf.io.decode_raw(parsed_features['target'], tf.float16)
        return seq, targets

    def read_tfr(self, tfr_pattern):
        tfr_files = natsorted(glob.glob(tfr_pattern))
        if tfr_files:
            dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
        else:
            print('Cannot order TFRecords %s' % tfr_pattern, file=sys.stderr)
            dataset = tf.data.Dataset.list_files(tfr_pattern)
        dataset = dataset.flat_map(self.file_to_records)
        dataset = dataset.map(self.parse_proto)
        dataset = dataset.batch(1)
        return dataset
    def __iter__(self):
        for seq_raw, targets_raw in self.dataset:
            seq = seq_raw.numpy().reshape(-1,4).astype('int8')
            targets = targets_raw.numpy().reshape(self.target_length,-1).astype('float16')
            # yield {
            #       "sequence": seq,
            #       "target": targets[:, self.target_ind],
                
            #   }
            yield seq, targets[:, self.target_ind]



class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

class BasenjiDataSet(torch.utils.data.IterableDataset):
    @staticmethod
    def get_organism_path(data_path, organism):
        return os.path.join(data_path, organism)

    @classmethod
    def get_metadata(cls, data_path, organism):
        # Keys:
        # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
        # pool_width, crop_bp, target_length
        path = os.path.join(cls.get_organism_path(data_path, organism), 'statistics.json')
        with tf.io.gfile.GFile(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def one_hot_encode(sequence):
        return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

    @classmethod
    def get_tfrecord_files(cls, data_path, organism, subset):
        # Sort the values by int(*).
        return sorted(tf.io.gfile.glob(os.path.join(
            cls.get_organism_path(data_path, organism), 'targets', f'{subset}-*.tfr'
        )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

    @property
    def num_channels(self):
        metadata = self.get_metadata(self.data_path, self.organism)
        return metadata['num_targets']

    @staticmethod
    def deserialize(serialized_example, metadata):
        """Deserialize bytes stored in TFRecordFile."""
        # Deserialization
        feature_map = {
            'sequence': tf.io.FixedLenFeature([], tf.string),  # Ignore this, resize our own bigger one
            'target': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_example(serialized_example, feature_map)
        sequence = tf.io.decode_raw(example['sequence'], tf.bool)
        sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
        sequence = tf.cast(sequence, tf.float32)

        target = tf.io.decode_raw(example['target'], tf.float16)
        target = tf.reshape(target,
                            (metadata['target_length'], metadata['num_targets']))
        target = tf.cast(target, tf.float32)

        return {'sequence_old': sequence,
                'target': target}

    @classmethod
    def get_dataset(cls, data_path, organism, subset, num_threads=8):
        metadata = cls.get_metadata(data_path, organism)
        dataset = tf.data.TFRecordDataset(cls.get_tfrecord_files(data_path, organism, subset),
                                          compression_type='ZLIB',
                                          num_parallel_reads=num_threads).map(
            functools.partial(cls.deserialize, metadata=metadata)
        )
        return dataset

    def __init__(self, data_path, organism: str, subset: str, seq_len: int, fasta_path: str, n_to_test: int = -1):
        assert subset in {"train", "valid", "test"}
        assert organism in {"human", "mouse"}
        self.data_path = data_path
        self.organism = organism
        self.subset = subset
        self.base_dir = self.get_organism_path(data_path, organism)
        self.seq_len = seq_len
        self.fasta_reader = FastaStringExtractor(fasta_path)
        self.n_to_test = n_to_test
        with tf.io.gfile.GFile(f"{self.base_dir}/sequences.bed", 'r') as f:
            region_df = pd.read_csv(f, sep="\t", header=None)
            region_df.columns = ['chrom', 'start', 'end', 'subset']
            self.region_df = region_df.query('subset==@subset').reset_index(drop=True)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Only support single process loading"
        # If num_threads > 1, the following will actually shuffle the inputs! luckily we catch this with the sequence comparison
        basenji_iterator = self.get_dataset(self.data_path, self.organism, self.subset, num_threads=1).as_numpy_iterator()
        for i, records in enumerate(basenji_iterator):
            loc_row = self.region_df.iloc[i]
            target_interval = Interval(loc_row['chrom'], loc_row['start'], loc_row['end'])
            sequence_one_hot = self.one_hot_encode(self.fasta_reader.extract(target_interval.resize(self.seq_len)))
            if self.n_to_test >= 0 and i < self.n_to_test:
                old_sequence_onehot = records["sequence_old"]
                if old_sequence_onehot.shape[0] > sequence_one_hot.shape[0]:
                    diff = old_sequence_onehot.shape[0] - sequence_one_hot.shape[0]
                    trim = diff // 2
                    np.testing.assert_equal(old_sequence_onehot[trim:(-trim)], sequence_one_hot)
                elif sequence_one_hot.shape[0] > old_sequence_onehot.shape[0]:
                    diff = sequence_one_hot.shape[0] - old_sequence_onehot.shape[0]
                    trim = diff // 2
                    np.testing.assert_equal(old_sequence_onehot, sequence_one_hot[trim:(-trim)])
                else:
                    np.testing.assert_equal(old_sequence_onehot, sequence_one_hot)
            # yield {
            #     "sequence": sequence_one_hot,
            #     "target": records["target"],
            # }
            yield sequence_one_hot, records["target"]


def get_genomes(path):
    genome = selene_sdk.sequences.Genome(
                    input_path=path,
                    blacklist_regions= 'hg38'
                )
    noblacklist_genome = selene_sdk.sequences.Genome(
                    input_path=path )
    return genome, noblacklist_genome

class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """
    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None, 
        replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

            
        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist)  for blacklist in self.blacklists]
            self.initialized=True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise
        
        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s)-start,0): int(e)-start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(self.blacklists, self.blacklists_indices, self.replacement_indices, self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = wigmat[replacement_indices, np.fmax(int(s)-start,0): int(e)-start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)]=0
        return wigmat
