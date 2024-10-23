import unittest
import pandas as pd
from poc_transform.data import prepare_data


class TestDataFunctions(unittest.TestCase):

    def test_prepare_data(self):
        texts = [f"text_{i}" for i in range(10)]
        labels = [f"label_{i}" for i in range(10)]
        data = pd.DataFrame({"review": texts, "sentiment": labels})
        prepared_data = prepare_data(data)
        self.assertEqual(prepared_data, (texts, labels), 'Data preparation returned wrong values')

if __name__ == '__main__':
    unittest.main()
