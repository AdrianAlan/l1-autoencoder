import argparse
import h5py
import numpy as np
import uproot3 as uproot

from tqdm import tqdm


class HDF5Generator:

    def __init__(self,
                 file_path,
                 hdf5_dataset_path,
                 verbose=False):
        self.file_path = uproot.open(file_path)
        self.hdf5_dataset_path = hdf5_dataset_path
        self.verbose = verbose

    def create_hdf5_dataset(self):

        caloTower = self.file_path['l1CaloTowerEmuTree']['L1CaloTowerTree']
        self.hdf5_dataset_size = len(caloTower['nTower'])

        eta_full = caloTower['ieta'].array()
        phi_full = caloTower['iphi'].array()
        et_full = caloTower['iet'].array()
        ratio_full = caloTower['iratio'].array()

        # Create the HDF5 file.
        hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'w')

        hdf5_Eta = hdf5_dataset.create_dataset(
                name='Eta',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        hdf5_Phi = hdf5_dataset.create_dataset(
                name='Phi',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        hdf5_ET = hdf5_dataset.create_dataset(
                name='ET',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

        hdf5_Ratio = hdf5_dataset.create_dataset(
                name='Ratio',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

        i = 0
        if args.verbose:
            progress_bar = tqdm(total=self.hdf5_dataset_size,
                                desc='Processing')

        for i in np.arange(self.hdf5_dataset_size, dtype=int):

            hdf5_Eta[i] = eta_full[i]
            hdf5_Phi[i] = phi_full[i]
            hdf5_ET[i] = et_full[i]
            hdf5_Ratio[i] = ratio_full[i]

            i += 1

            if self.verbose:
                progress_bar.update(1)

        progress_bar.close()
        hdf5_dataset.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert root file data to h5')
    parser.add_argument('in_file', type=str, help='Input path')
    parser.add_argument('out_file', type=str, help='Output path')
    parser.add_argument('-v', '--verbose', action="store_true")

    args = parser.parse_args()

    generator = HDF5Generator(args.in_file, args.out_file, args.verbose)
    generator.create_hdf5_dataset()
