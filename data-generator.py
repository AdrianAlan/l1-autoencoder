import argparse
import numpy as np
import uproot3 as uproot

from scipy import sparse
from tqdm import tqdm


class DatasetGenerator:

    def __init__(self,
                 file_path,
                 dataset_path,
                 data_shape,
                 verbose=False):
        self.root = uproot.open(file_path)
        self.dataset_path = dataset_path
        self.data_shape = data_shape
        self.verbose = verbose

    def create_dataset(self):

        caloTower = self.root['l1CaloTowerEmuTree']['L1CaloTowerTree']
        self.dataset_size = len(caloTower['nTower'].array())

        eta_full = caloTower['ieta'].array()
        phi_full = caloTower['iphi'].array()
        et_full = caloTower['iet'].array()

        if args.verbose:
            progress_bar = tqdm(total=self.dataset_size, desc='Processing')

        dataset = np.zeros((self.dataset_size, 56, 72))

        for i in np.arange(self.dataset_size, dtype=int):

            phis, etas, ets = phi_full[i], eta_full[i], et_full[i]

            # Delete zero column
            etas = np.array([e+1 if e < 0 else e for e in etas])
            # Construct array from sparse indices
            caloImages = sparse.csc_matrix((ets, (etas + 40, phis-1)),
                                           shape=self.data_shape).toarray()
            # Remove the encaps
            caloImages = caloImages[13:-13, :]
            # Normalize
            scaler = np.max(caloImages)
            if scaler:
                caloImages = caloImages / scaler
            # Fill array
            dataset[i, :, :] = caloImages

            if self.verbose:
                progress_bar.update(1)

        np.save(self.dataset_path, dataset)

        if args.verbose:
            progress_bar.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert root data to numpy pickle')
    parser.add_argument('in_file', type=str, help='Input path')
    parser.add_argument('out_file', type=str, help='Output path')
    parser.add_argument('-v', '--verbose', action="store_true")

    args = parser.parse_args()

    generator = DatasetGenerator(args.in_file,
                                 args.out_file,
                                 (82, 72),
                                 args.verbose)
    generator.create_dataset()
