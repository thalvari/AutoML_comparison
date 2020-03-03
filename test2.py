import warnings

warnings.filterwarnings("ignore")

import autokeras as ak
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# seed = 42
# rn.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)
# torch.manual_seed(seed)
# random_state = RandomState(seed)

# def main():
x, y = load_digits(return_X_y=True)
y = y.astype(np.uint8)
x = x.astype(np.uint8)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
image_hw = np.sqrt(len(x[0])).astype(int)
x_train = x_train.reshape((len(x_train), image_hw, image_hw, 1))
model = ak.ImageClassifier(augment=False, verbose=True)
model.fit(x_train, y_train, time_limit=60 * 15)

# if __name__ == "__main__":
#     main()
