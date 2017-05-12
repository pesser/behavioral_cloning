import os, urllib.request, sys, math, csv, pickle
from zipfile import ZipFile
import numpy as np
from multiprocessing.pool import ThreadPool

import keras
import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array

# path where sample training data will be put
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok = True)


def dl_progress(count, block_size, total_size):
    """Progress bar used during download."""
    if total_size == -1:
        if count == 0:
            sys.stdout.write("Unknown size of download.\n")
    else:
        length = 50
        current_size = count * block_size
        done = current_size * length // total_size
        togo = length - done
        prog = "[" + done * "=" + togo * "-" + "]"
        sys.stdout.write(prog)
        if(current_size < total_size):
            sys.stdout.write("\r")
        else:
            sys.stdout.write("\n")
    sys.stdout.flush()


def download_data():
    """Download data."""
    # direct download from google drive is not working but dataset is
    # included in the repository
    data = {
            "data.zip": "https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip",
            "data_curves.zip": "https://drive.google.com/open?id=0B_2YVqPvaFeTSmtBUDlGcHhTWWc"}
    local_data = {}
    for fname, url in data.items():
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            print("Downloading {}".format(fname))
            urllib.request.urlretrieve(url, path, reporthook = dl_progress)
        else:
            print("Found {}. Skipping download.".format(fname))
        local_data[fname] = path
    return local_data


def extract_data(path):
    """Extract zip file if not already extracted."""
    with ZipFile(path) as f:
        targets = dict((fname, os.path.join(data_dir, fname)) for fname in f.namelist())
        if not all([os.path.exists(target) for target in list(targets.values())[:10]]): # only check for first 10 for speed
            print("Extracting {}".format(path))
            f.extractall(data_dir)
        else:
            print("Skipping extraction of {}".format(path))
    return targets


def prepare_data():
    """Collect data from subdirectories of data dir, merge them, split into
    training, validation and  testing splits and pickle to disk."""
    out_path = os.path.join(data_dir, "data.p")
    if os.path.isfile(out_path):
        print("Using {}".format(out_path))
        with open(out_path, "rb") as f:
            data = pickle.load(f)
        return data

    csvfname = "driving_log.csv"
    imgdirname = "IMG"
    datasets = []
    for dir_ in os.listdir(data_dir):
        # paths
        dir_ = os.path.join(data_dir, dir_)
        csvpath = os.path.join(dir_, csvfname)
        imgpath = os.path.join(dir_, imgdirname)
        if not os.path.isdir(dir_) or not os.path.isfile(csvpath) or not os.path.isdir(imgpath):
            continue

        # parse csv file into dictionary
        lines = list()
        with open(csvpath, "r", newline = "") as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                lines.append(line)
        if lines[0][0] == "center":
            keys = lines.pop(0)
        else:
            keys = "center,left,right,steering,throttle,brake,speed".split(",")
        # adapt paths
        for i, line in enumerate(lines):
            for col in range(3):
                fname = os.path.basename(line[col])
                path = os.path.join(imgpath, fname)
                lines[i][col] = path
        data = dict()
        for i, k in enumerate(keys):
            data[k] = list()
            for line in lines:
                data[k].append(line[i])
        datasets.append(data)
    assert(datasets)
    # merge them
    all_data = dict((k, list()) for k in keys)
    for dataset in datasets:
        for k in keys:
            all_data[k] += dataset[k]
    # shuffle
    n_samples = len(all_data[keys[0]])
    for v in all_data.values():
        assert(len(v) == n_samples)
    shuffled_indices = np.random.permutation(n_samples)
    for k in all_data.keys():
        all_data[k] = [all_data[k][i] for i in shuffled_indices]
    # split
    n_train = int(0.7 * n_samples)
    n_valid = int(0.5 * (n_samples - n_train))
    begins = {
            "train": 0,
            "valid": n_train,
            "test": n_train + n_valid}
    ends = {
            "train": n_train,
            "valid": n_train + n_valid,
            "test": n_samples}
    data = dict()
    for split in ["train", "valid", "test"]:
        data[split] = dict()
        for k in keys:
            data[split][k] = all_data[k][begins[split]:ends[split]]
    # pickle to disk
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print("Wrote {}".format(out_path))
    return data


def table_format(row, header = False, width = 10):
    """Format row as markdown table."""
    result = "|" + "|".join(str(entry).center(width) for entry in row) + "|"
    if header:
        sep = "".join(map(lambda x: x if x == "|" else "=", result))
        result = result + "\n" + sep
    return result


def summarize(data):
    """Summarize size of data for each split."""
    keys = sorted(list(data["train"].keys()))
    print("Keys: {}".format(", ".join(keys)))
    headers = ["Split", "Samples", "Height", "Width", "Channels"]
    print(table_format(headers, header = True))
    for split in ["train", "valid", "test"]:
        d = data[split]
        n = len(d[keys[0]])
        sample_image_path = data[split]["center"][0]
        img = load_img(sample_image_path)
        x = img_to_array(img)
        h, w, c = x.shape
        row = [split, n, h, w, c]
        print(table_format(row))


class FileFlow(object):
    """Load batches on the fly."""

    def __init__(self, batch_size, data, augment = False):
        self.data = data
        self.y = data["steering"]
        self.batch_size = batch_size
        self.augment = augment
        self.n = len(self.y)
        self.steps_per_epoch = math.ceil(self.n / self.batch_size)
        # infer shape
        self.img_shape = img_to_array(load_img(self.data["center"][0])).shape
        self.shuffle()


    def __next__(self):
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        batch_fnames = dict()
        for pos in ["center", "left", "right"]:
            batch_fnames[pos] = [self.data[pos][i] for i in batch_indices]
        batch_y = [float(self.y[i]) for i in batch_indices]

        current_batch_size = len(batch_indices)

        batch = np.zeros((current_batch_size,) + self.img_shape, dtype = K.floatx())
        for i in range(current_batch_size):
            if self.augment:
                # choose one of the camera positions at random
                pos = np.random.choice(["center", "left", "right"])
                fname = batch_fnames[pos][i]
                # and adjust target steering angle
                if pos == "left":
                    batch_y[i] += 0.12
                if pos == "right":
                    batch_y[i] -= 0.12
            else:
                fname = batch_fnames["center"][i]
            img = load_img(fname)
            x = img_to_array(img)
            if self.augment:
                # with probability 0.5 flip image and steering angle
                if np.random.randint(2):
                    x = np.fliplr(x)
                    batch_y[i] = -batch_y[i]
            batch[i] = x

        if batch_end > self.n:
            self.shuffle()
        else:
            self.batch_start = batch_end

        return batch, batch_y


    def shuffle(self):
        self.batch_start = 0
        self.indices = np.random.permutation(self.n)


def keras_normalize(x):
    return x / 127.5 - 1.0


def keras_global_average_pooling(x):
    return K.mean(x, [1,2])


class Model(object):
    """CNN for end-to-end steering angle prediction."""

    def __init__(self, img_shape, n_epochs):
        self.img_shape = img_shape
        self.learning_rate = 1e-3
        self.decay = self.learning_rate / n_epochs
        self.dr_rate = 0.25
        self.n_epochs = n_epochs
        self.define_model()
        self.init_model()


    def define_model(self):
        n_features = 32
        kernel_size = 3
        activation = "relu"
        kernel_initializer = "he_normal"

        x = keras.layers.Input(shape = self.img_shape)
        features = keras.layers.Cropping2D(cropping = ((50,25), (0,0)))(x)
        features = keras.layers.Lambda(keras_normalize)(features)

        for i in range(6):
            features = keras.layers.Convolution2D(
                    filters = 2**i * n_features,
                    kernel_size = kernel_size,
                    strides = 2,
                    padding = "SAME",
                    activation = activation,
                    kernel_initializer = kernel_initializer)(features)
            features = keras.layers.Dropout(self.dr_rate)(features)

        features = keras.layers.Convolution2D(
                filters = 1024,
                kernel_size = 1,
                strides = 1,
                padding = "SAME",
                activation = activation,
                kernel_initializer = kernel_initializer)(features)
        features = keras.layers.Dropout(self.dr_rate)(features)
        features = keras.layers.Convolution2D(
                filters = 1,
                kernel_size = 1,
                strides = 1,
                padding = "SAME",
                activation = None,
                kernel_initializer = kernel_initializer)(features)

        output = keras.layers.Lambda(keras_global_average_pooling)(features)
        self.model = keras.models.Model(x, output)


    def init_model(self):
        optimizer = keras.optimizers.Adam(lr = self.learning_rate, decay = self.decay)
        self.model.compile(loss = "mse", optimizer = optimizer)
        ckpt_path = os.path.join(data_dir, "model.h5")
        checkpoint = keras.callbacks.ModelCheckpoint(
                ckpt_path,
                monitor = "val_loss",
                verbose = 1,
                save_best_only = True)
        self.callbacks = [checkpoint]


    def fit(self, batches, valid_batches = None):
        self.history = self.model.fit_generator(
                generator = batches,
                steps_per_epoch = batches.steps_per_epoch,
                epochs = self.n_epochs,
                validation_data = valid_batches,
                validation_steps = valid_batches.steps_per_epoch,
                callbacks = self.callbacks)


    def test(self, batches):
        return self.model.evaluate_generator(
                generator = batches,
                steps = batches.steps_per_epoch)


if __name__ == "__main__":
    files = download_data()
    for fname in files:
        extract_data(files[fname])
    data = prepare_data()
    summarize(data)

    batch_size = 64
    train_batches = FileFlow(batch_size, data["train"], augment = True)
    valid_batches = FileFlow(batch_size, data["valid"])
    test_batches = FileFlow(batch_size, data["test"])

    img_shape = train_batches.img_shape
    n_epochs = 100

    model = Model(img_shape, n_epochs)
    model.fit(train_batches, valid_batches)
    test_loss = model.test(test_batches)
    print("Testing loss: {:.4e}".format(test_loss))
