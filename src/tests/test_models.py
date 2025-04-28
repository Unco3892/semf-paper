import unittest
import numpy as np
import torch
from src.semf.models import MultiMLPs, MultiXGBs, MultiETs, QNN

class TestModels(unittest.TestCase):
    def setUp(self):
        """Set up small synthetic datasets for testing."""
        self.n_train = 100
        self.n_infer = 10
        self.k = 2
        self.output_dim = 1
        self.device = "cpu"
        self.n_estimators = 10
        self.n_epochs = 10

        self.data_list = [
            {'inputs': torch.from_numpy(np.random.rand(self.n_train, 1).astype(np.float32)),
             'outputs': torch.from_numpy(np.random.rand(self.n_train, self.output_dim).astype(np.float32)),
             'weights': torch.from_numpy(np.random.rand(self.n_train).astype(np.float32))}
            for _ in range(self.k)
        ]

        self.new_data_list = [
            {'inputs': torch.from_numpy(np.random.rand(self.n_infer, 1).astype(np.float32))}
            for _ in range(self.k)
        ]

    def test_multi_mlps(self):
        """Test MultiMLPs model."""
        model = MultiMLPs(device=self.device, nn_epochs=self.n_epochs)
        model.train_multiple(self.data_list)
        predictions = model.predict_multiple(self.new_data_list)
        self.assertEqual(len(predictions), self.k)
        for pred in predictions:
            self.assertEqual(pred.shape[0], self.n_infer)

    def test_multi_xgbs(self):
        """Test MultiXGBs model."""
        model = MultiXGBs(device=self.device, tree_n_estimators=self.n_estimators)
        model.train_multiple(self.data_list)
        predictions = model.predict_multiple(self.new_data_list)
        self.assertEqual(len(predictions), self.k)
        for pred in predictions:
            self.assertEqual(pred.shape[0], self.n_infer)

    def test_multi_ets(self):
        """Test MultiETs model."""
        model = MultiETs(device=self.device, tree_n_estimators=self.n_estimators)
        model.train_multiple(self.data_list)
        predictions = model.predict_multiple(self.new_data_list)
        self.assertEqual(len(predictions), self.k)
        for pred in predictions:
            self.assertEqual(pred.shape[0], self.n_infer)

    def test_qnn(self):
        """Test QNN model."""
        x = torch.from_numpy(np.random.rand(self.n_train, 1).astype(np.float32)).to(self.device)
        y = torch.from_numpy(np.random.rand(self.n_train, 1).astype(np.float32)).to(self.device)
        qnn = QNN(x.shape[1], 1, device=self.device)
        model = qnn.train_model(x, y, batch_size=10, epochs=10)
        quantiles = [0.1, 0.5, 0.9]
        predictions = qnn.predict(x[:self.n_infer], quantiles)
        self.assertEqual(predictions.shape[0], self.n_infer)
        self.assertEqual(predictions.shape[1], len(quantiles))

if __name__ == '__main__':
    unittest.main()
